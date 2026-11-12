from torchvision import transforms, datasets
from torch.utils.data import DataLoader

resnet_18_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load training dataset
train_dataset = datasets.ImageFolder(
    root="/Users/indranilbhattacharyya/Documents/photo-album-project/app/photo app_split/Train",
    transform=resnet_18_preprocess
)

# Load testing dataset
test_dataset = datasets.ImageFolder(
    root="/Users/indranilbhattacharyya/Documents/photo-album-project/app/photo app_split/Test",
    transform=resnet_18_preprocess
)



BATCH_SIZE = 16

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)