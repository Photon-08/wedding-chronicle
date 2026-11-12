import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def visualize_predictions(model, test_loader, device, num_images=5):

    model.eval()

    images, labels = next(iter(test_loader))

    images = images[:num_images].to(device)
    labels = labels[:num_images]

    with torch.no_grad():
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1).cpu()

    images = images.cpu()

    # ImageFolder class mapping
    class_names = test_loader.dataset.classes

    # Undo ImageNet normalization for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    images = images * std + mean
    images = torch.clamp(images, 0, 1)

    fig, axes = plt.subplots(1, num_images, figsize=(18, 4))

    for i in range(num_images):

        image = images[i].permute(1, 2, 0).numpy()

        pred_label = class_names[predictions[i].item()]
        true_label = class_names[labels[i].item()]

        axes[i].imshow(image)
        axes[i].set_title(
            f"Predicted: {pred_label}\nTrue: {true_label}",
            color="green" if pred_label == true_label else "red"
        )
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()