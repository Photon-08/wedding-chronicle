import io
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from torchvision import transforms

from training.models.baseline import BaselineModel
from gradcam import generate_gradcam


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "model.pth"
INDEX_PATH = BASE_DIR / "index.html"

# Must match the ImageFolder class_to_idx mapping used during training.
CLASS_NAMES = ["Both", "Indu", "Neel", "Others"]
IMAGE_SIZE = 224

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


# --------------------------------------------------
# Load model and checkpoint
# --------------------------------------------------

# pretrained=False is intentional: model.pth already contains the trained
# ResNet18 weights, so the app does not need to download ImageNet weights.
model = BaselineModel(num_classes=len(CLASS_NAMES))

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )

try:
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True
    )
except TypeError:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

# Support checkpoints saved with DataParallel.
if state_dict and all(key.startswith("module.") for key in state_dict):
    state_dict = {
        key.replace("module.", "", 1): value
        for key, value in state_dict.items()
    }

model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(title="Wedding Photo Classifier with Grad-CAM")


@app.get("/")
def home():
    if not INDEX_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="index.html was not found beside app.py."
        )
    return FileResponse(INDEX_PATH)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_loaded": True
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    allowed_types = {"image/jpeg", "image/png", "image/webp"}

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG, PNG or WEBP image."
        )

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")

        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # The Grad-CAM function performs a gradient-enabled forward pass.
        # Do NOT wrap this call in no_grad() or inference_mode().
        target_layer = model.feature_extractor.feature_extractor.layer4[-1]

        logits, gradcam_base64 = generate_gradcam(
            model=model,
            input_tensor=input_tensor,
            target_layer=target_layer,
            target_size=(IMAGE_SIZE, IMAGE_SIZE)
        )

        probabilities = F.softmax(logits, dim=1)[0].detach().cpu()
        confidence, predicted_idx = torch.max(probabilities, dim=0)
        predicted_idx = int(predicted_idx.item())

        probability_dict = {
            CLASS_NAMES[i]: round(float(probabilities[i].item()) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        return {
            "predicted_class": CLASS_NAMES[predicted_idx],
            "confidence": round(float(confidence.item()) * 100, 2),
            "probabilities": probability_dict,
            "gradcam": f"data:image/png;base64,{gradcam_base64}"
        }

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be read as an image."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
