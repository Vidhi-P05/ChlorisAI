import io
import json
import logging
import traceback
from pathlib import Path
from typing import Dict
from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI(title="Flower Recognition API", version="2.1")

app.mount("/static", StaticFiles(directory="templates/static"), name="static")
templates = Jinja2Templates(directory="templates")

# --------------------------------------------------
# Globals
# --------------------------------------------------
model = None
class_names = None
flower_database: Dict[str, Dict[str, str]] = {}
device = None

# --------------------------------------------------
# Model definition
# --------------------------------------------------
class FlowerClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# --------------------------------------------------
# Load model & metadata
# --------------------------------------------------
def load_model():
    global model, class_names, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load class names
    with open("data/class_names.json", "r", encoding="utf-8") as f:
        class_names = json.load(f)

    # Load model
    model = FlowerClassifier(num_classes=len(class_names))

    # Download the model from Hugging Face if not present locally
    HF_REPO_ID = "Vidhi-Pateliya-01/ChlorisAI-model"  # <-- your repo
    MODEL_FILENAME = "best_model.pth"
    CHECKPOINT_DIR = Path("checkpoints")
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    MODEL_PATH = CHECKPOINT_DIR / MODEL_FILENAME

    if not MODEL_PATH.exists():
        logger.info("Downloading model from Hugging Face...")
        MODEL_PATH = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME, cache_dir=str(CHECKPOINT_DIR))

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(
        checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    )

    model.to(device)
    model.eval()
    logger.info("Model loaded successfully")

def load_flower_database():
    global flower_database
    with open("data/flower_database.json", "r", encoding="utf-8") as f:
        flower_database = json.load(f)
    logger.info(f"Loaded flower database with {len(flower_database)} entries")

# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    return transform(image).unsqueeze(0).to(device)

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def get_flower_info(name: str) -> Dict[str, str]:
    key = name.lower()
    return flower_database.get(
        key,
        {
            "scientific_name": f"Scientific name of {name}",
            "description": f"Description of {name}.",
            "medicinal_use": "No medicinal data available.",
            "habitat": "Habitat information not available."
        }
    )

# --------------------------------------------------
# Startup
# --------------------------------------------------
@app.on_event("startup")
async def startup_event():
    load_model()
    load_flower_database()

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid image file")

        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        if image.width < 32 or image.height < 32:
            raise HTTPException(status_code=400, detail="Image too small")

        input_tensor = preprocess_image(image)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, idx = torch.max(probs, 1)

        flower_name = class_names[idx.item()]
        info = get_flower_info(flower_name)

        return {
            "flower_name": flower_name,
            "confidence": confidence.item(),
            **info
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

# --------------------------------------------------
# Entry point
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port)

