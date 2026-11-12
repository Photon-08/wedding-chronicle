import re
import os
import torch 
import torch.nn as nn
from torchvision import transforms
from preprocess import resnet_18_preprocess, train_loader, test_loader
from models.baseline import BaselineModel
from visualize import visualize_predictions

if torch.backends.mps.is_available():
  device = torch.device("mps")
elif torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

print(f"Using device: {device}")

model = BaselineModel(num_classes=4).to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.001)
loss = nn.CrossEntropyLoss()

epochs = 20
loss_history = []
for epoch in range(epochs):
    running_loss = 0.0
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optim.zero_grad()
        outputs = model(images)
        loss_value = loss(outputs, labels)
        running_loss += loss_value.item()
        loss_value.backward()
        optim.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
    loss_history.append(running_loss/len(train_loader))

print("Training complete.")
print("Loss history:", loss_history)

chkpoint_path = r"/Users/indranilbhattacharyya/Documents/photo-album-project/app/checkpoints"
torch.save(model.state_dict(), os.path.join(chkpoint_path, "model.pth"))

visualize_predictions(
    model=model,
    test_loader=test_loader,
    device=device,
    num_images=10
)