"""Dermatology inference app — loads best_model.pt and displays diagnosis in a GUI window."""

import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import ttk

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "outputs" / "best_model.pt"
CFG_PATH   = BASE_DIR / "outputs" / "config.json"

# ---------------------------------------------------------------------------
# Load config + model
# ---------------------------------------------------------------------------

def load_model_and_config():
    with open(CFG_PATH) as f:
        cfg_dict = json.load(f)

    # Import project modules
    sys.path.insert(0, str(BASE_DIR))
    from config import Config
    from models.builder import build_model

    cfg = Config.from_dict(cfg_dict)
    cfg.pretrained = False  # weights come from checkpoint
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(cfg)
    checkpoint = torch.load(MODEL_PATH, map_location=cfg.device)

    # Support both raw state-dict and wrapped checkpoints
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.to(cfg.device).eval()

    return model, cfg


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(image_path: str, model, cfg):
    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(cfg.device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0]

    top_idx   = int(probs.argmax())
    top_label = cfg.class_names[top_idx]
    top_conf  = float(probs[top_idx]) * 100

    all_scores = [
        (cfg.class_names[i], float(probs[i]) * 100)
        for i in range(len(cfg.class_names))
    ]
    all_scores.sort(key=lambda x: x[1], reverse=True)

    return img, top_label, top_conf, all_scores


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

RISK_HIGH   = {"Melanoma", "Squamous cell carcinoma", "Actinic keratosis"}
RISK_MEDIUM = {"Basal cell carcinoma", "Dermatofibroma"}


def risk_color(label: str) -> str:
    if label in RISK_HIGH:
        return "#e53935"   # red
    if label in RISK_MEDIUM:
        return "#fb8c00"   # orange
    return "#43a047"       # green


def show_window(image_path: str, img: Image.Image, label: str, conf: float, scores: list):
    root = tk.Tk()
    root.title("Dermotology — Diagnosis")
    root.configure(bg="#1e1e2e")
    root.resizable(False, False)

    PAD = 16

    # ---- Left: image panel ----
    img_display = img.copy()
    img_display.thumbnail((380, 380))

    # Annotate with top diagnosis
    draw = ImageDraw.Draw(img_display)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    color = risk_color(label)
    text  = f"{label}  {conf:.1f}%"
    bbox  = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([4, 4, tw + 12, th + 12], fill=(0, 0, 0, 180))
    draw.text((8, 6), text, fill=color, font=font)

    from PIL import ImageTk
    tk_img = ImageTk.PhotoImage(img_display)

    left_frame = tk.Frame(root, bg="#1e1e2e")
    left_frame.pack(side=tk.LEFT, padx=PAD, pady=PAD)

    img_label = tk.Label(left_frame, image=tk_img, bg="#1e1e2e")
    img_label.pack()

    # ---- Right: diagnosis panel ----
    right_frame = tk.Frame(root, bg="#282840", padx=PAD, pady=PAD)
    right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, PAD), pady=PAD)

    tk.Label(right_frame, text="DIAGNOSIS", font=("Segoe UI", 11, "bold"),
             fg="#aaaacc", bg="#282840").pack(anchor="w")

    tk.Label(right_frame, text=label, font=("Segoe UI", 20, "bold"),
             fg=risk_color(label), bg="#282840", wraplength=260, justify="left").pack(anchor="w", pady=(4, 0))

    tk.Label(right_frame, text=f"Confidence: {conf:.1f}%",
             font=("Segoe UI", 12), fg="#ccccdd", bg="#282840").pack(anchor="w", pady=(2, 12))

    tk.Frame(right_frame, height=1, bg="#44446a").pack(fill=tk.X, pady=(0, 10))

    tk.Label(right_frame, text="All classes", font=("Segoe UI", 10, "bold"),
             fg="#aaaacc", bg="#282840").pack(anchor="w")

    for cls_name, cls_conf in scores:
        row = tk.Frame(right_frame, bg="#282840")
        row.pack(fill=tk.X, pady=1)

        name_lbl = tk.Label(row, text=cls_name, width=28, anchor="w",
                            font=("Segoe UI", 9), fg="#ccccdd", bg="#282840")
        name_lbl.pack(side=tk.LEFT)

        bar_bg = tk.Frame(row, bg="#44446a", height=12, width=120)
        bar_bg.pack(side=tk.LEFT, padx=(4, 4))
        bar_bg.pack_propagate(False)

        bar_w = max(2, int(cls_conf / 100 * 120))
        bar_fg = tk.Frame(bar_bg, bg=risk_color(cls_name), height=12, width=bar_w)
        bar_fg.place(x=0, y=0)

        tk.Label(row, text=f"{cls_conf:5.1f}%", font=("Segoe UI", 9),
                 fg="#aaaacc", bg="#282840").pack(side=tk.LEFT)

    tk.Frame(right_frame, height=1, bg="#44446a").pack(fill=tk.X, pady=(12, 8))

    tk.Label(right_frame, text=f"Image: {Path(image_path).name}",
             font=("Segoe UI", 8), fg="#666688", bg="#282840").pack(anchor="w")

    root.mainloop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Dermatology inference")
    parser.add_argument("image", help="Path to skin lesion image")
    args = parser.parse_args()

    image_path = args.image
    if not Path(image_path).exists():
        sys.exit(f"Error: file not found: {image_path}")

    print("Loading model...")
    model, cfg = load_model_and_config()

    print(f"Running inference on {image_path} ...")
    img, label, conf, scores = predict(image_path, model, cfg)

    print(f"Diagnosis: {label}  ({conf:.1f}%)")
    show_window(image_path, img, label, conf, scores)


if __name__ == "__main__":
    main()
