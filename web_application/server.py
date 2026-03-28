"""FastAPI inference server — loads best_model.pt and exposes POST /predict.

Run from the project root:
    uvicorn web_application.server:app --host 0.0.0.0 --port 8000

Or from inside web_application/:
    uvicorn server:app --host 0.0.0.0 --port 8000

Then open http://<your-pc-ip>:8000 on your phone (same Wi-Fi).
"""

import io
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WEB_DIR     = Path(__file__).parent
PROJECT_DIR = WEB_DIR.parent
MODEL_PATH  = PROJECT_DIR / "outputs" / "best_model.pt"
CFG_PATH    = PROJECT_DIR / "outputs" / "config.json"
STATIC_DIR  = WEB_DIR / "static"

sys.path.insert(0, str(PROJECT_DIR))

from utils.preproccess import preprocess

# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------
RISK_HIGH   = {"Melanoma", "Squamous cell carcinoma", "Actinic keratosis"}
RISK_MEDIUM = {"Basal cell carcinoma", "Dermatofibroma"}


def risk_level(label: str) -> str:
    if label in RISK_HIGH:
        return "high"
    if label in RISK_MEDIUM:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# App + model state
# ---------------------------------------------------------------------------
app = FastAPI(title="Dermotology API")

_model = None
_cfg   = None


@app.on_event("startup")
async def _load_model():
    global _model, _cfg

    if not CFG_PATH.exists():
        raise RuntimeError(f"Config not found: {CFG_PATH}")
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}")

    from config import Config
    from models.builder import build_model

    with open(CFG_PATH) as f:
        cfg_dict = json.load(f)

    _cfg = Config.from_dict(cfg_dict)
    _cfg.pretrained = False
    _cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    _model = build_model(_cfg)
    checkpoint = torch.load(MODEL_PATH, map_location=_cfg.device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    _model.load_state_dict(state)
    _model.to(_cfg.device).eval()

    print(f"[server] Model ready on {_cfg.device}  |  classes: {_cfg.class_names}")


# ---------------------------------------------------------------------------
# Inference endpoint
# ---------------------------------------------------------------------------
_transform_cache: transforms.Compose | None = None


def _get_transform() -> transforms.Compose:
    global _transform_cache
    if _transform_cache is None:
        _transform_cache = transforms.Compose([
            transforms.Resize((_cfg.image_size, _cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return _transform_cache


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(503, "Model not loaded yet — try again in a moment.")
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "Uploaded file must be an image.")

    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(422, "Could not decode image.")

    img = preprocess(img)

    tensor = _get_transform()(img).unsqueeze(0).to(_cfg.device)

    with torch.no_grad():
        probs = F.softmax(_model(tensor), dim=1)[0]

    top_idx   = int(probs.argmax())
    top_label = _cfg.class_names[top_idx]
    top_conf  = float(probs[top_idx]) * 100

    all_scores = [
        {
            "label":      _cfg.class_names[i],
            "confidence": round(float(probs[i]) * 100, 2),
            "risk":       risk_level(_cfg.class_names[i]),
        }
        for i in range(len(_cfg.class_names))
    ]
    all_scores.sort(key=lambda x: x["confidence"], reverse=True)

    return JSONResponse({
        "label":      top_label,
        "confidence": round(top_conf, 1),
        "risk":       risk_level(top_label),
        "all_scores": all_scores,
    })


# ---------------------------------------------------------------------------
# Serve the frontend
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))
