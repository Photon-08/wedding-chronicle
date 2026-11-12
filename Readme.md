# Wedding Chronicle
### স্মৃতির অ্যালবাম · A Collection of Our Moments

*A photograph preserves a moment. A collection of photographs preserves a story.*

## The Idea

A wedding is experienced by many people, and everyone captures it differently.

Friends take photographs of the couple, family members capture candid moments, relatives record group pictures, and guests often leave with hundreds of photographs stored on their individual phones. Eventually, these memories become scattered across messaging applications, personal galleries and devices.

**Wedding Chronicle aims to bring all these photographs together into one shared digital wedding album.**

The idea is simple: provide guests with a common platform where they can upload the photographs they capture. Once uploaded, an AI-powered image classification system analyses each photograph and automatically categorises it based on the people appearing in it.

The initial classification system recognises four categories:

- **Neel:** Photographs containing Neel.
- **Indu:** Photographs containing Indu.
- **Both:** Photographs containing both Neel and Indu.
- **Others:** Photographs that do not belong to the above categories.

This creates the foundation for a personalised wedding photo collection, where photographs can eventually be organised, searched and revisited without manually sorting through thousands of images.

The longer-term vision is to move beyond image-level classification towards face detection and recognition, allowing the system to identify individuals in group photographs and organise memories around the people in them.

At its heart, Wedding Chronicle is an attempt to combine a personal family archive with practical applications of computer vision and deep learning.

---

## Overview

Wedding Chronicle currently consists of two primary components:

1. **Frontend:** A responsive, Bengali-inspired interface designed for uploading photographs and viewing classification results.
2. **Backend:** A Python-based image classification service powered by PyTorch, ResNet18 and FastAPI. It performs image inference and generates Grad-CAM visualisations to provide insight into the model's predictions.

The current implementation performs image-level classification. It predicts the most appropriate category for an entire photograph rather than identifying and recognising individual faces.

---

## Design Philosophy

The visual identity draws inspiration from Bengali wedding traditions and the aesthetic of a printed family album.

The interface uses:

- Warm, paper-inspired backgrounds.
- Deep vermilion accents.
- Muted gold decorative elements.
- Editorial typography combining English and Bengali typefaces.
- Subtle ornamental details inspired by traditional alpana patterns.
- Minimal interactions that keep photographs at the centre of the experience.

The intention is to create something that feels personal and timeless rather than like a conventional file-upload application.

---

## Technical Architecture

### Frontend

Built using HTML5, CSS3 and vanilla JavaScript.

Features include:

- Image selection and preview.
- Communication with the prediction API.
- Predicted category and confidence display.
- Class-wise probability distribution.
- Grad-CAM visualisation.

### Backend

| Component | Technology |
|---|---|
| API framework | FastAPI |
| Deep learning framework | PyTorch |
| Model architecture | ResNet18 |
| Image processing | Pillow |
| Numerical operations | NumPy |
| Visualisation | Matplotlib |
| Inference device | Apple Metal (MPS), when available; otherwise CPU |

### Model Architecture

The initial classification model uses a ResNet18 backbone.

The architecture consists of:

1. A ResNet18 feature extraction network.
2. A fully connected classification layer.
3. A four-class output representing the categories defined above.

The current implementation uses the ResNet18 ImageNet output representation (1,000 dimensions) as input to the final classification layer.

The model produces four logits, which are converted into class probabilities using softmax.

---

## Model Explainability: Grad-CAM

Wedding Chronicle incorporates **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visualise image regions that contribute positively to a selected class prediction.

The implementation uses the final convolutional block of ResNet18:

`layer4[-1]`

The Grad-CAM pipeline:

1. Captures feature activations from the target convolutional layer.
2. Computes gradients of the predicted class score with respect to those activations.
3. Uses globally averaged gradients as channel-wise importance weights.
4. Produces a class activation map using the weighted feature maps.
5. Applies ReLU and normalises the resulting activation map.
6. Resizes the heatmap to the input image dimensions.
7. Overlays the heatmap on the original photograph.

The visualisation provides an indication of which spatial regions influenced the model's prediction.

**Important:** Grad-CAM is an interpretability technique, not proof that the model has correctly identified a person.

---

## Project Structure

```text
wedding-chronicle/
│
├── app.py
├── gradcam.py
├── index.html
├── requirements.txt
├── .gitignore
│
├── training/
│   └── models/
│       └── baseline.py
│
├── checkpoints/
│   └── model.pth
│
└── README.md
```

The model checkpoint is required for inference but should not be committed to the public repository.

---

## Getting Started

### Prerequisites

- Python 3.10 or later.
- Git.
- A trained model checkpoint.
- A compatible PyTorch installation.

### 1. Clone the repository

```bash
git clone https://github.com/Photon-08/wedding-chronicle.git
cd wedding-chronicle
```

### 2. Create a virtual environment

```bash
python3 -m venv .env
source .env/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Core dependencies include:

```bash
pip install torch torchvision
pip install fastapi "uvicorn[standard]"
pip install python-multipart pillow numpy matplotlib
```

### 4. Add the model checkpoint

Place the trained model checkpoint at:

```text
checkpoints/model.pth
```

The checkpoint must contain weights compatible with the `BaselineModel` implementation.

### 5. Start the backend

```bash
uvicorn app:app --reload
```

The application will be available at:

`http://127.0.0.1:8000`

Interactive API documentation:

`http://127.0.0.1:8000/docs`

---

## API Reference

### `GET /`

Serves the frontend application.

### `POST /predict`

Accepts an image and returns its classification result.

**Example response:**

```json
{
  "predicted_class": "Neel",
  "confidence": 91.2,
  "probabilities": {
    "Both": 2.1,
    "Indu": 1.5,
    "Neel": 91.2,
    "Others": 5.2
  },
  "gradcam": "data:image/png;base64,..."
}
```

The values are illustrative. Actual results depend on the trained model and input photograph.

### `GET /health`

Provides a basic health-check endpoint.

---

## Deployment

### Frontend: GitHub Pages

The static frontend can be hosted through GitHub Pages.

1. Push the frontend files to the repository.
2. Open repository Settings.
3. Navigate to Pages.
4. Select the `main` branch.
5. Select the root directory.
6. Save the configuration.

### Backend: Separate Deployment

GitHub Pages only serves static files. It cannot execute a Python application or host a FastAPI inference service.

The backend therefore needs a separate Python hosting environment.

For a separate frontend and backend deployment:

- Configure the frontend to send prediction requests to the deployed API URL.
- Configure appropriate CORS policies.
- Enable HTTPS.
- Implement upload size restrictions and input validation.
- Consider rate limiting and inference resource requirements.
- Define secure handling and retention policies for uploaded photographs.

---

## Data and Privacy

Wedding photographs are personal and potentially sensitive.

The project follows a simple principle: photographs should not be published as part of the source code.

Dataset images, model checkpoints, local environment files and other private assets should remain outside version control.

The `.gitignore` file excludes common image formats, environment files, Python cache directories and local development artefacts.

Any production deployment should explicitly define how uploaded photographs are stored, processed, retained and deleted.

---

## Current Scope and Limitations

Wedding Chronicle is an evolving computer vision project.

The current implementation has the following limitations:

- Classification operates at the image level.
- It does not yet perform individual face detection or identity verification.
- Performance depends on dataset quality, class balance and the diversity of training photographs.
- Grad-CAM provides visual explanations but does not guarantee reliable model reasoning.
- Confidence scores should not be interpreted as calibrated certainty.
- Inference requires a compatible model checkpoint.

---

## Future Development

The project is intended to evolve beyond a basic four-class image classifier.

### 1. Face Detection

Detect individual faces within group photographs and extract face-level regions for downstream processing.

### 2. Face Embeddings and Recognition

Explore deep metric learning and face representation techniques.

Potential approaches include:

- FaceNet.
- ArcFace.
- AdaFace.
- Contrastive learning.
- Triplet loss.

### 3. Multi-Person Image Understanding

Identify multiple known individuals in a photograph and aggregate face-level predictions into image-level categories.

### 4. Cloud-Based Photo Collection

Introduce cloud-backed uploads, object storage, automated classification and organised photo retrieval.

### 5. Personalised Photo Gallery

Build a shared gallery where photographs can be browsed and filtered by person, event and other relevant attributes.

### 6. Improved Model Evaluation

Explore:

- Class-wise precision and recall.
- Confusion matrices.
- F1-score.
- Generalisation to unseen photographs.
- Robustness under different lighting conditions.
- Confidence calibration.

---

## Contributing

This is primarily a personal project developed around a wedding and the memories associated with it.

Technical discussions, suggestions and ideas for improving the computer vision pipeline are welcome.

Feel free to open an issue or submit a pull request.

---

## A Note Behind the Project

Some photographs are planned. Others happen unexpectedly.

A family member captures a moment, a friend takes a photograph, and someone else records something that might otherwise have been forgotten.

Wedding Chronicle is an attempt to bring those scattered moments together into one place.

It is a small intersection of personal memories, Bengali wedding aesthetics and an interest in building practical applications with machine learning.

**স্মৃতির অ্যালবাম — Our Moments, Together.**

---

**Built with care by Neel.**