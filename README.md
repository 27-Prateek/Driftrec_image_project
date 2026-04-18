# 🚀 Image Restoration using DriftRec

This project focuses on restoring JPEG-compressed images using a pretrained DriftRec diffusion model and evaluating the results using multiple image quality metrics.

---

## 🎥 Video Explanations

* PHASE1- https://drive.google.com/drive/folders/1rORx2xkMpNZ9PUSA-P44Oqy7Ix7tWxpW?usp=sharing
* PHASE2- https://drive.google.com/drive/folders/1SG0L6ZvqNzK6G2UUK7SMOweHpy9VhOdB?usp=drive_link

## ⚙️ 1. Setup

Create and activate the environment:

```bash
python -m venv driftrec_env
driftrec_env\Scriptsctivate
```

Install dependencies:

```bash
pip install torch torchvision numpy pillow tqdm matplotlib scikit-image piq pandas
```

## 📁 2. Project Structure

```text
driftrec/
│
├── model.ckpt                  # Pretrained model
├── test_input/                 # Original images
├── test_corrupted/             # JPEG compressed images
├── enhanced_output_cpu/        # Model output
├── corrected_corrupted/        # Color corrected images
│
├── enhance_folder.py
├── color_correct.py
├── calc_dist_feats.py
├── temp_eval.py
```

## 🧪 3. Workflow Pipeline

`Original Images` → `JPEG Compression` → `DriftRec Model` → `Color Correction` → `Evaluation`

## 🟨 4. Step-by-Step Execution

### 🔹 Step 1 — Prepare Test Images

```bash
mkdir test_input
```

Download sample images (PowerShell):

```powershell
for ($i=1; $i -le 20; $i++) {
    Invoke-WebRequest "https://picsum.photos/256?random=$i" -OutFile "test_input/img$i.jpg"
}
```

### 🔹 Step 2 — Apply JPEG Corruption

```python
from PIL import Image
import os

inp = "test_input"
out = "test_corrupted"
os.makedirs(out, exist_ok=True)

for f in os.listdir(inp):
    img = Image.open(os.path.join(inp, f)).convert("RGB")
    img.save(os.path.join(out, f), "JPEG", quality=10)

print("Done!")
```

### 🔹 Step 3 — Run DriftRec Enhancement

```bash
python enhance_folder.py --ckpt model.ckpt --indir test_corrupted --outdir enhanced_output_cpu --N 50 --batch_size 1
```
> 👉 **N** = number of diffusion steps (lower = faster)

### 🔹 Step 4 — Apply Color Correction

```bash
python color_correct.py --tgt-dir test_input --enh-dir enhanced_output_cpu --out-dir corrected_corrupted
```

## 📊 5. Evaluation

### 🔹 5.1 PSNR / SSIM (Per-image)

```bash
python temp_eval.py
```
Measures:
* Pixel similarity
* Structural similarity

### 🔹 5.2 FID / KID (Perceptual)

```bash
python calc_dist_feats.py --gt-dir test_input --enh-dir test_corrupted
python calc_dist_feats.py --gt-dir test_input --enh-dir enhanced_output_cpu
```

## 📌 6. Metrics Used

| Metric | Description |
| :--- | :--- |
| **PSNR** | Pixel-level similarity |
| **SSIM** | Structural similarity |
| **FID** | Perceptual distribution similarity |
| **KID** | Stable version of FID |

## 📈 7. Results Summary

| Comparison | Observation |
| :--- | :--- |
| **Corrupted vs Original** | High FID (~300) → Poor quality |
| **Enhanced vs Original** | Very low FID (~3–4) → Strong improvement |
| **Color Correction** | Slight improvement |

## 🧠 8. Key Insights

* DriftRec improves perceptual quality significantly.
* PSNR may decrease due to pixel changes.
* FID/KID better reflect visual quality.
* Color correction only adjusts brightness, not structure.

## ⚠️ 9. Limitations

* Small dataset (~20 images)
* FID/KID unstable for small samples
* CPU inference is slow
* Quantized model not directly runnable

## 🚀 10. Optimization Tips

Reduce steps:
```bash
python enhance_folder.py ... --N 20
```

Resize images:
* 256 → 128

*Tip: Use GPU for faster inference.*

## 📦 11. Quantization (Optional)

* 8-bit conversion reduces size.
* Not directly usable for inference.
* Runtime quantization preferred.


## 🔚 Conclusion

DriftRec effectively restores corrupted images, significantly improving perceptual quality (FID), even when traditional metrics like PSNR do not reflect improvement.
