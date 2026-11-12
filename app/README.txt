# Wedding Photo Classifier with Grad-CAM

## Files
- `app.py`: FastAPI application and model loading.
- `gradcam.py`: Grad-CAM computation and overlay generation.
- `index.html`: Existing wedding-themed frontend, updated to display Grad-CAM.
- `requirements.txt`: Python dependencies.
- `training/models/baseline.py`: Keep your existing model architecture here.

## Important
This app expects the trained checkpoint at `checkpoints/model.pth`.
It expects your `BaselineModel` to expose the ResNet18 as
`model.feature_extractor`, with the final convolutional block at
`model.feature_extractor.layer4[-1]`.

The `CLASS_NAMES` order must match the class-to-index mapping used during
training. For an ImageFolder dataset with folders `Both`, `Indu`, `Neel`,
`Others`, the alphabetical order is:
`Both: 0, Indu: 1, Neel: 2, Others: 3`.

## Install and run
Install the requirements in your existing virtual environment:

```bash
python3 -m pip install -r requirements.txt
uvicorn app:app --reload
```

Then open http://127.0.0.1:8000
