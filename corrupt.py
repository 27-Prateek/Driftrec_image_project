from PIL import Image
import os

inp = "test_input"
out = "test_corrupted"
os.makedirs(out, exist_ok=True)

for f in os.listdir(inp):
    img = Image.open(os.path.join(inp, f)).convert("RGB")
    img.save(os.path.join(out, f), "JPEG", quality=20)

print("Done!")