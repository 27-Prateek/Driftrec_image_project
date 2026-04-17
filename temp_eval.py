import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

orig_dir = "test_input"
corr_dir = "test_corrupted"
rest_dir = "enhanced_output2"

def compute_metrics(img1, img2):
    img1 = cv2.resize(img1, (256, 256))
    img2 = cv2.resize(img2, (256, 256))

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    mse = np.mean((img1 - img2) ** 2)
    psnr = 10 * np.log10((255 ** 2) / mse) if mse != 0 else 100
    ssim_val = ssim(img1, img2, channel_axis=2, data_range=255)

    return mse, psnr, ssim_val

print("\n📊 IMAGE QUALITY COMPARISON\n")

psnr_diff = []
ssim_diff = []

for fname in os.listdir(orig_dir):
    try:
        orig = cv2.imread(os.path.join(orig_dir, fname))
        corr = cv2.imread(os.path.join(corr_dir, fname))

        rest_path_png = os.path.join(rest_dir, fname.replace(".jpg", ".png"))
        rest_path_jpg = os.path.join(rest_dir, fname)

        rest = cv2.imread(rest_path_png) if os.path.exists(rest_path_png) else cv2.imread(rest_path_jpg)

        if orig is None or corr is None or rest is None:
            continue

        mse_corr, psnr_corr, ssim_corr = compute_metrics(orig, corr)
        mse_rest, psnr_rest, ssim_rest = compute_metrics(orig, rest)

        print(f"\n🖼 {fname}")
        print(f"Corrupted → PSNR: {psnr_corr:.2f}, SSIM: {ssim_corr:.3f}")
        print(f"Restored  → PSNR: {psnr_rest:.2f}, SSIM: {ssim_rest:.3f}")

        psnr_diff.append(psnr_rest - psnr_corr)
        ssim_diff.append(ssim_rest - ssim_corr)

        if psnr_rest > psnr_corr:
            print("✅ Model IMPROVED the image")
        else:
            print("❌ No improvement (or worse)")

    except Exception as e:
        print(f"Error with {fname}: {e}")

print("\n📈 AVERAGE IMPROVEMENT")
print(f"PSNR Gain: {np.mean(psnr_diff):.2f}")
print(f"SSIM Gain: {np.mean(ssim_diff):.3f}")