import cv2, numpy as np, glob, os, random

img_dir = "converted/yolo/val/images"
lbl_dir = "converted/yolo/val/labels"
out_dir = "debug_vis"
os.makedirs(out_dir, exist_ok=True)

# wie viele zufällig?
N = 3
random.seed(42)  # optional: für reproduzierbare Auswahl

# alle Bilder (png/jpg) sammeln
img_paths = sorted(glob.glob(f"{img_dir}/*.png") + glob.glob(f"{img_dir}/*.jpg") + glob.glob(f"{img_dir}/*.jpeg"))

# nur solche nehmen, zu denen auch ein Label existiert
img_paths_with_lbl = []
for p in img_paths:
    name = os.path.splitext(os.path.basename(p))[0]
    if os.path.exists(os.path.join(lbl_dir, f"{name}.txt")):
        img_paths_with_lbl.append(p)

# zufällige Auswahl
pick = random.sample(img_paths_with_lbl, k=min(N, len(img_paths_with_lbl)))

for img_path in pick:
    name = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = os.path.join(lbl_dir, f"{name}.txt")
    img = cv2.imread(img_path)
    H, W = img.shape[:2]

    with open(lbl_path, "r") as f:
        for line in f:
            vals = [float(v) for v in line.strip().split()]
            cls = int(vals[0])
            coords = vals[1:]
            if len(coords) < 6:  # mindestens 3 Punkte
                continue
            pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= W
            pts[:, 1] *= H
            pts = pts.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    out_path = os.path.join(out_dir, f"{name}.png")
    cv2.imwrite(out_path, img)
    print("wrote:", out_path)
