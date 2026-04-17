import torch
from driftrec.model import ScoreModel, DiscriminativeModel

CKPT_PATH = "model.ckpt"
SAVE_PATH = "model_quantized.ckpt"

# ===== STEP 1: LOAD CHECKPOINT =====
ckpt = torch.load(CKPT_PATH, map_location="cpu")

# ===== STEP 2: DETERMINE MODEL TYPE =====
# DriftRec supports two models
if "model_type" in ckpt:
    if ckpt["model_type"] == "score":
        model = ScoreModel(**ckpt.get("model_config", {}))
    else:
        model = DiscriminativeModel(**ckpt.get("model_config", {}))
else:
    # fallback (common case)
    try:
        model = ScoreModel(**ckpt.get("model_config", {}))
    except:
        model = DiscriminativeModel(**ckpt.get("model_config", {}))

# ===== STEP 3: LOAD WEIGHTS =====
if "state_dict" in ckpt:
    model.load_state_dict(ckpt["state_dict"])
else:
    model.load_state_dict(ckpt)

model.eval()

print("✅ Model loaded successfully")

# ===== STEP 4: APPLY QUANTIZATION =====
print("⚡ Applying dynamic quantization...")

model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

print("✅ Quantization applied")

# ===== STEP 5: SAVE AS CKPT =====
quant_ckpt = {
    "state_dict": model.state_dict()
}

# preserve config if exists
if "model_config" in ckpt:
    quant_ckpt["model_config"] = ckpt["model_config"]

if "model_type" in ckpt:
    quant_ckpt["model_type"] = ckpt["model_type"]

torch.save(quant_ckpt, SAVE_PATH)

print(f"✅ Quantized model saved at: {SAVE_PATH}")