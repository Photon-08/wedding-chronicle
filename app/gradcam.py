import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def generate_gradcam(model, input_tensor, target_layer, target_size=(224, 224)):
    """
    Generate a Grad-CAM overlay for the class predicted by the model.

    Returns:
        logits: detached model logits, shape [1, num_classes]
        encoded_png: base64-encoded PNG overlay
    """
    model.eval()
    captured = {}

    def forward_hook(_module, _inputs, output):
        captured["activations"] = output

    hook_handle = target_layer.register_forward_hook(forward_hook)

    try:
        # Gradients must be enabled even though this is an inference endpoint.
        with torch.enable_grad():
            x = input_tensor.detach().clone()

            logits = model(x)
            predicted_class = int(logits.argmax(dim=1).item())

            activations = captured["activations"]

            gradients = torch.autograd.grad(
                outputs=logits[:, predicted_class].sum(),
                inputs=activations,
                retain_graph=False,
                create_graph=False
            )[0]

            # One importance weight per feature-map channel.
            channel_weights = gradients.mean(
                dim=(2, 3),
                keepdim=True
            )

            cam = torch.relu(
                (channel_weights * activations).sum(dim=1, keepdim=True)
            )

            cam_min = cam.amin(dim=(2, 3), keepdim=True)
            cam_max = cam.amax(dim=(2, 3), keepdim=True)
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

            cam = F.interpolate(
                cam,
                size=target_size,
                mode="bilinear",
                align_corners=False
            )

            heatmap = cam[0, 0].detach().cpu().numpy()
            logits = logits.detach()

    finally:
        hook_handle.remove()

    # Convert the normalized heatmap into a JET color map.
    heatmap_rgb = plt.get_cmap("jet")(
        np.clip(heatmap, 0.0, 1.0)
    )[..., :3]

    # Undo ImageNet normalization to reconstruct the displayed model input.
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image_array = input_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
    image_array = np.clip(image_array * std + mean, 0.0, 1.0)

    overlay = np.clip(
        0.55 * image_array + 0.45 * heatmap_rgb,
        0.0,
        1.0
    )

    overlay_image = Image.fromarray(
        (overlay * 255).astype(np.uint8)
    )

    buffer = io.BytesIO()
    overlay_image.save(buffer, format="PNG")

    encoded_png = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return logits, encoded_png
