import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image

from configs.defaults import *
from src.utils import (
    build_inputs_for_G,
    build_input_for_D,
    assert_finite,
    total_variation_loss_fg
)
from src.losses import (
    compute_loss_parts,
    d_hinge_loss,
    g_hinge_loss
)


# =========================
# Metric History
# =========================
def init_metrics_history():
    keys = [
        "loss", "genonly", "adv",
        "l1_fg", "l1_bg", "edge_fg", "hp_fg",
        "intm", "tv_fg", "low",
        "d_loss", "g_loss"
    ]

    return {
        "train": {k: [] for k in keys},
        "val": {k: [] for k in keys if k not in ["d_loss", "g_loss"]}
    }


# =========================
# SEGSYN Score
# =========================
def segsyn_score(val_m, w_gen=1.0, w_intm=3.0, w_edge=1.0, w_hp=1.0):
    return (
        w_gen * val_m["genonly"] +
        w_intm * val_m["intm"] +
        w_edge * val_m["edge_fg"] +
        w_hp * val_m["hp_fg"]
    )


# =========================
# Save Best
# =========================
def save_best_segsyn_overwrite(G, save_path, epoch, score, val_m):
    os.makedirs(save_path, exist_ok=True)

    torch.save({
        "generator_state_dict": G.state_dict(),
        "epoch": epoch,
        "tag": "SEGSYN",
        "metric": "segsyn_score",
        "value": float(score),
        "val_metrics": {k: float(v) for k, v in val_m.items()}
    }, os.path.join(save_path, "best_SEGSYN.pth"))


# =========================
# Resume
# =========================
def resume_training_if_available(G, D, optG, optD, save_path, device):
    ckpt_path = os.path.join(save_path, "checkpoint_latest.pth")

    start_epoch = 0
    metrics_history = None
    best_state = None

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        G.load_state_dict(ckpt["generator_state_dict"])
        D.load_state_dict(ckpt["discriminator_state_dict"])
        optG.load_state_dict(ckpt["g_optimizer_state_dict"])
        optD.load_state_dict(ckpt["d_optimizer_state_dict"])

        start_epoch = ckpt.get("epoch", 0) + 1
        metrics_history = ckpt.get("metrics_history", None)
        best_state = ckpt.get("best_state", None)

        print(f"🔁 Resumed from epoch {start_epoch}")
    else:
        print("🚀 Starting training from scratch.")

    if metrics_history is None:
        metrics_history = init_metrics_history()

    if best_state is None:
        best_state = {"best_segsyn": float("inf")}

    return start_epoch, metrics_history, best_state


# =========================
# Early Stopping
# =========================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode="min"):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best = None
        self.num_bad = 0

    def _is_improved(self, current):
        if self.best is None:
            return True

        if self.mode == "min":
            return current < (self.best - self.min_delta)

        return current > (self.best + self.min_delta)

    def step(self, current):
        if self._is_improved(current):
            self.best = float(current)
            self.num_bad = 0
            return False

        self.num_bad += 1
        return self.num_bad >= self.patience


# =========================
# Train Discriminator Step
# =========================
def train_discriminator_step(G, D, batch, optD, device):
    real = batch["image"].to(device).float()
    mask = batch["mask"].to(device).float()

    with torch.no_grad():
        fake = G(build_inputs_for_G(mask, use_noise=USE_NOISE))

        if fake.shape != real.shape:
            fake = F.interpolate(
                fake,
                size=real.shape[2:],
                mode="bilinear",
                align_corners=False
            )

    assert_finite(fake, "fake")

    pr = D(build_input_for_D(real, mask))
    pf = D(build_input_for_D(fake, mask))

    assert_finite(pr, "pr")
    assert_finite(pf, "pf")

    d_loss = d_hinge_loss(pr, pf)

    assert_finite(d_loss, "d_loss")

    optD.zero_grad()
    d_loss.backward()
    optD.step()

    return float(d_loss.detach().item())


# =========================
# Train Generator Step
# =========================
def train_generator_step(G, D, batch, optG, cur_lambda_adv, device):
    real = batch["image"].to(device).float()
    mask = batch["mask"].to(device).float()

    fake = G(build_inputs_for_G(mask, use_noise=USE_NOISE))

    if fake.shape != real.shape:
        fake = F.interpolate(
            fake,
            size=real.shape[2:],
            mode="bilinear",
            align_corners=False
        )

    assert_finite(fake, "fake")

    adv = (
        g_hinge_loss(D(build_input_for_D(fake, mask)))
        if cur_lambda_adv > 0
        else torch.tensor(0.0, device=device)
    )

    parts = compute_loss_parts(
        fake, real, mask,
        LAMBDA_L1_FG,
        LAMBDA_L1_BG,
        LAMBDA_EDGE_FG,
        LAMBDA_HP_FG,
        LAMBDA_INTM,
        LAMBDA_TV_FG,
        LAMBDA_LOW,
        total_variation_loss_fg
    )

    total = cur_lambda_adv * adv + parts["genonly"]

    assert_finite(total, "total")

    optG.zero_grad()
    total.backward()
    optG.step()

    return {
        "total": float(total.detach().item()),
        "genonly": float(parts["genonly"].detach().item()),
        "adv": float(adv.detach().item())
    }


# =========================
# Evaluation
# =========================
@torch.no_grad()
def evaluate_metrics(G, D, loader, cur_lambda_adv, device):
    G.eval()
    D.eval()

    keys = [
        "loss", "genonly", "adv",
        "l1_fg", "l1_bg", "edge_fg", "hp_fg",
        "intm", "tv_fg", "low"
    ]

    scores = {k: 0.0 for k in keys}
    n = 0

    for batch in loader:
        real = batch["image"].to(device).float()
        mask = batch["mask"].to(device).float()

        fake = G(build_inputs_for_G(mask, use_noise=USE_NOISE))

        if fake.shape != real.shape:
            fake = F.interpolate(
                fake,
                size=real.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        adv = (
            g_hinge_loss(D(build_input_for_D(fake, mask)))
            if cur_lambda_adv > 0
            else torch.tensor(0.0, device=device)
        )

        parts = compute_loss_parts(
            fake, real, mask,
            LAMBDA_L1_FG,
            LAMBDA_L1_BG,
            LAMBDA_EDGE_FG,
            LAMBDA_HP_FG,
            LAMBDA_INTM,
            LAMBDA_TV_FG,
            LAMBDA_LOW,
            total_variation_loss_fg
        )

        total = cur_lambda_adv * adv + parts["genonly"]

        vals = {
            "loss": total,
            "genonly": parts["genonly"],
            "adv": adv,
            "l1_fg": parts["l1_fg"],
            "l1_bg": parts["l1_bg"],
            "edge_fg": parts["edge_fg"],
            "hp_fg": parts["hp_fg"],
            "intm": parts["intm"],
            "tv_fg": parts["tv_fg"],
            "low": parts["low"]
        }

        batch_size = fake.size(0)

        for k in keys:
            scores[k] += float(vals[k].detach().item()) * batch_size

        n += batch_size

    for k in keys:
        scores[k] /= max(1, n)

    return scores


# =========================
# Save Preview Images
# =========================
@torch.no_grad()
def save_predictions_grid(G, val_loader, save_dir, epoch, device):
    if not SAVE_PREVIEWS:
        return

    os.makedirs(save_dir, exist_ok=True)
    G.eval()

    for i, batch in enumerate(val_loader):
        if i >= 1:
            break

        mask = batch["mask"].to(device).float()
        real = batch["image"].to(device).float()

        fake = G(build_inputs_for_G(mask, use_noise=USE_NOISE))

        if fake.shape[-2:] != real.shape[-2:]:
            fake = F.interpolate(
                fake,
                size=real.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        for j in range(min(mask.size(0), 4)):
            mask_rgb = mask[j].repeat(3, 1, 1)
            fake_img = (fake[j] + 1) / 2
            real_img = (real[j] + 1) / 2

            save_image(
                torch.cat([mask_rgb, fake_img, real_img], dim=-1),
                os.path.join(save_dir, f"epoch{epoch+1}_sample{j}.png"),
                normalize=False
            )


# =========================
# Full Training Pipeline
# =========================
def train_gan_pipeline(
    G,
    D,
    train_loader,
    val_loader,
    optG,
    optD,
    save_path,
    device,
    num_epochs=100,
    early_stop_patience=10,
    early_stop_min_delta=0.0
):
    os.makedirs(save_path, exist_ok=True)

    start_epoch, metrics_history, best_state = resume_training_if_available(
        G, D, optG, optD, save_path, device
    )

    early_stopper = EarlyStopping(
        patience=early_stop_patience,
        min_delta=early_stop_min_delta,
        mode="min"
    )

    for epoch in range(start_epoch, num_epochs):
        G.train()
        D.train()

        cur_lambda_adv = 0.0 if epoch < WARMUP_EPOCHS else LAMBDA_ADV

        d_losses = []
        g_losses = []

        for batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs} (λ_adv={cur_lambda_adv})"
        ):
            if cur_lambda_adv > 0:
                d_losses.append(
                    train_discriminator_step(G, D, batch, optD, device)
                )

            g_out = train_generator_step(
                G, D, batch, optG, cur_lambda_adv, device
            )
            g_losses.append(g_out["total"])

        train_m = evaluate_metrics(G, D, train_loader, cur_lambda_adv, device)
        val_m = evaluate_metrics(G, D, val_loader, cur_lambda_adv, device)

        mean_d = float(np.mean(d_losses)) if d_losses else 0.0
        mean_g = float(np.mean(g_losses)) if g_losses else 0.0

        score = segsyn_score(val_m)

        print(f"\n[Epoch {epoch+1}]")
        print(
            f"Val loss={val_m['loss']:.4f} | "
            f"genonly={val_m['genonly']:.4f} | "
            f"intm={val_m['intm']:.4f}"
        )
        print(
            f"edge_fg={val_m['edge_fg']:.4f} | "
            f"hp_fg={val_m['hp_fg']:.4f} | "
            f"SEGSYN={score:.6f}"
        )
        print(f"Train d_loss={mean_d:.4f} | g_loss={mean_g:.4f}\n")

        for k in metrics_history["train"].keys():
            if k == "d_loss":
                metrics_history["train"][k].append(mean_d)
            elif k == "g_loss":
                metrics_history["train"][k].append(mean_g)
            else:
                metrics_history["train"][k].append(train_m.get(k, float("nan")))

        for k in metrics_history["val"].keys():
            metrics_history["val"][k].append(val_m.get(k, float("nan")))

        save_predictions_grid(
            G,
            val_loader,
            os.path.join(save_path, "epoch_outputs"),
            epoch,
            device
        )

        if score < best_state["best_segsyn"]:
            best_state["best_segsyn"] = score
            save_best_segsyn_overwrite(G, save_path, epoch, score, val_m)
            print(f"💾 Saved BEST_SEGSYN: score={score:.6f}")

        torch.save({
            "epoch": epoch,
            "generator_state_dict": G.state_dict(),
            "discriminator_state_dict": D.state_dict(),
            "g_optimizer_state_dict": optG.state_dict(),
            "d_optimizer_state_dict": optD.state_dict(),
            "metrics_history": metrics_history,
            "best_state": best_state,
            "task": "mask_conditioned_gan_cell_synthesis"
        }, os.path.join(save_path, "checkpoint_latest.pth"))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        stop = early_stopper.step(score)

        print(
            f"🧭 EarlyStop monitor=SEGSYN "
            f"value={score:.6f} | "
            f"best={early_stopper.best:.6f} | "
            f"bad_epochs={early_stopper.num_bad}/{early_stopper.patience}"
        )

        if stop:
            print(
                f"⛔ Early stopping at epoch {epoch+1}. "
                f"Best SEGSYN={early_stopper.best:.6f}"
            )
            break

    print("🏁 Training finished.")

    return metrics_history, best_state