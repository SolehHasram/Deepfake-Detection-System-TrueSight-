"""
Offline Data Augmentation
==========================
Buang imej HF yang mungkin silap label.
Generate imej baharu dari 101 imej original → ~400 per kelas.
Label dijamin betul kerana semua dari sumber yang sama.
"""
import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
import shutil

TARGET_REAL    = os.path.join("deepfake-and-real-images", "Real")
TARGET_FAKE    = os.path.join("deepfake-and-real-images", "Fake")
TARGET_PER_CLASS = 400   # jumlah akhir per kelas
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("=" * 55)
print("  Offline Data Augmentation")
print("=" * 55)

# ── Langkah 1: Buang imej HF (mungkin silap label) ────────────────────────────
removed = 0
for folder in [TARGET_REAL, TARGET_FAKE]:
    for f in os.listdir(folder):
        if f.startswith("hf_"):
            os.remove(os.path.join(folder, f))
            removed += 1
print(f"\n🗑️  Removed {removed} HuggingFace images (possible label mismatch)")

# Count original images
orig_real = sorted([f for f in os.listdir(TARGET_REAL)
                    if f.lower().endswith(('.jpg','.jpeg','.png'))])
orig_fake = sorted([f for f in os.listdir(TARGET_FAKE)
                    if f.lower().endswith(('.jpg','.jpeg','.png'))])
print(f"✅ Original: Real={len(orig_real)} | Fake={len(orig_fake)}\n")


# ── Langkah 2: Fungsi augmentasi ──────────────────────────────────────────────
def augment_image(img: Image.Image) -> Image.Image:
    """Apply random augmentation to a PIL image."""
    ops = []

    # Flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Random rotation -25 to +25
    if random.random() > 0.3:
        angle = random.uniform(-25, 25)
        img = img.rotate(angle, expand=False, fillcolor=(128, 128, 128))

    # Brightness
    if random.random() > 0.3:
        factor = random.uniform(0.6, 1.4)
        img = ImageEnhance.Brightness(img).enhance(factor)

    # Contrast
    if random.random() > 0.3:
        factor = random.uniform(0.7, 1.3)
        img = ImageEnhance.Contrast(img).enhance(factor)

    # Saturation
    if random.random() > 0.4:
        factor = random.uniform(0.7, 1.3)
        img = ImageEnhance.Color(img).enhance(factor)

    # Sharpness
    if random.random() > 0.5:
        factor = random.uniform(0.5, 2.0)
        img = ImageEnhance.Sharpness(img).enhance(factor)

    # Slight blur
    if random.random() > 0.6:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    # Random crop + resize (simulates zoom)
    if random.random() > 0.4:
        w, h = img.size
        margin = int(min(w, h) * 0.15)
        left   = random.randint(0, margin)
        top    = random.randint(0, margin)
        right  = w - random.randint(0, margin)
        bottom = h - random.randint(0, margin)
        img = img.crop((left, top, right, bottom)).resize((w, h), Image.BILINEAR)

    return img


# ── Langkah 3: Generate augmented images ──────────────────────────────────────
def augment_folder(orig_files, folder, label, target_total):
    current = len([f for f in os.listdir(folder)
                   if f.lower().endswith(('.jpg','.jpeg','.png'))])
    need = target_total - current
    if need <= 0:
        print(f"  {label}: Already {current} images, skipping.")
        return

    print(f"  {label}: Generating {need} augmented images ({current} → {target_total})...")
    generated = 0

    while generated < need:
        # Pick a random original image
        src_fname = random.choice(orig_files)
        src_path  = os.path.join(folder, src_fname)

        try:
            img = Image.open(src_path).convert("RGB").resize((224, 224))
            aug = augment_image(img)
            out_name = f"aug_{label.lower()}_{generated:04d}.jpg"
            aug.save(os.path.join(folder, out_name), "JPEG", quality=88)
            generated += 1
            if generated % 100 == 0:
                print(f"    {generated}/{need}")
        except Exception as e:
            print(f"    Skip {src_fname}: {e}")
            continue

    print(f"  ✅ {label} done: {target_total} images total")


print("Generating augmented images...")
augment_folder(orig_real, TARGET_REAL, "Real", TARGET_PER_CLASS)
augment_folder(orig_fake, TARGET_FAKE, "Fake", TARGET_PER_CLASS)

# ── Ringkasan ──────────────────────────────────────────────────────────────────
final_real = len(os.listdir(TARGET_REAL))
final_fake = len(os.listdir(TARGET_FAKE))
print(f"\n✅ Augmentation complete!")
print(f"   Real: {final_real} images")
print(f"   Fake: {final_fake} images")
print(f"\n>>> Seterusnya: python train_model_dl.py")
