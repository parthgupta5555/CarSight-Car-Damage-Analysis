
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image


# The order must match the class-index order used during training.
CLASS_NAMES = [
    "Front Breakage",
    "Front Crushed",
    "Front Normal",
    "Rear Breakage",
    "Rear Crushed",
    "Rear Normal",
]


# Preprocessing must match the preprocessing used during training.
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        # The checkpoint contains the trained weights, so no need to
        # download ImageNet weights again when creating this architecture.
        self.model = models.resnet50(weights=None)

        # Keep the same layer-freezing setup as the training code.
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # This must match the final layer used when the checkpoint was saved.
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def load_model(checkpoint_path="saved_model.pth"):
    """
    Load the trained model checkpoint.

    Returns:
        model: model in evaluation mode
        device: torch.device("cuda") or torch.device("cpu")
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CarClassifierResNet(num_classes=len(CLASS_NAMES))

    # map_location allows a GPU-trained checkpoint to load on a CPU machine.
    try:
        state_dict = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        state_dict = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, device


def predict_image(image, model, device):
    """
    Predict the car-damage class for a PIL image.

    Returns a dictionary containing:
        class_name
        class_index
        confidence
        probabilities (scores for all six classes)
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    image = image.convert("RGB")
    image_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())
    scores = probabilities.detach().cpu().tolist()

    return {
        "class_name": CLASS_NAMES[predicted_index],
        "class_index": predicted_index,
        "confidence": float(scores[predicted_index]),
        "probabilities": {
            name: float(score)
            for name, score in zip(CLASS_NAMES, scores)
        },
    }


def predict(image_path):
    """
    Backward-compatible function matching the original helper:
    predict(image_path) returns only the predicted class name.
    """
    model, device = load_model("saved_model.pth")
    image = Image.open(image_path).convert("RGB")
    result = predict_image(image, model, device)
    return result["class_name"]
