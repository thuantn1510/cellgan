# =========================
# Dataset
# =========================
ROOT = "data/BBBC038v1"
IMG_SIZE = (512, 768)

# =========================
# Training
# =========================
BATCH_SIZE = 2
EPOCHS = 100

NUM_WORKERS = 2
PIN_MEMORY = True

# =========================
# Device
# =========================
DEVICE = "cuda"  

# =========================
# Optimizer
# =========================
LR_G = 1e-4
LR_D = 5e-5
BETAS = (0.5, 0.999)

# =========================
# GAN Strategy
# =========================
WARMUP_EPOCHS = 5
LAMBDA_ADV = 0.10

# =========================
# Loss Weights
# =========================
LAMBDA_L1_FG   = 60.0
LAMBDA_L1_BG   = 1.0
LAMBDA_EDGE_FG = 0.50
LAMBDA_HP_FG   = 0.60
LAMBDA_INTM    = 3.00
LAMBDA_LOW     = 0.005
LAMBDA_TV_FG   = 0.00

# =========================
# Noise Injection
# =========================
USE_NOISE = True

# =========================
# Checkpoint / Output
# =========================
SAVE_PATH = "outputs/checkpoints"

# =========================
# Preview (image per epoch)
# =========================
SAVE_PREVIEWS = True

# =========================
# Early Stopping (SEGSYN)
# =========================
EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 0.0

# =========================
# SEGSYN Weights
# =========================
SEGSYN_W_GEN  = 1.0
SEGSYN_W_INTM = 3.0
SEGSYN_W_EDGE = 1.0
SEGSYN_W_HP   = 1.0

# =========================
# Mask Generator
# =========================
MASK_SAVE_PATH = "outputs/mask_checkpoints"

LAMBDA_MASK_BCE = 1.0
LAMBDA_MASK_DICE = 1.0
LAMBDA_MASK_EDGE = 0.30

MASK_THRESHOLD = 0.5

# =========================
# Export
# =========================
EXPORT_ROOT = "outputs/fake_export_train"
EXPORT_LIMIT = None

# =========================
# Segmentation Training
# =========================
SEG_BATCH_SIZE = 4
SEG_EPOCHS = 100
SEG_LR = 3e-4
SEG_WEIGHT_DECAY = 1e-4
SEG_IN_CHANNELS = 3
SEG_THRESHOLD = 0.5

SEG_EARLY_STOP_PATIENCE = 10
SEG_EARLY_STOP_MIN_DELTA = 1e-4

SEG_SAVE_PATH = "outputs/segmentation"