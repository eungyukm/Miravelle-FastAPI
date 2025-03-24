import torch
from torchvision import transforms
from PIL import Image
from models.common import SimpleCNN

CLASS_NAMES = {
    "texture": ["전", "후"],
    "grotesque": ["정형", "비정형"],
    "object": ["비사물", "사물"],
    "style": ["실사", "카툰"]
}

MODEL_FILES = {
    "texture": "src/model/texture_classifier.pth",
    "grotesque": "src/model/grotesque_classifier.pth",
    "object": "src/model/object_classifier.pth",
    "style": "src/model/style_classifier.pth"
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_model(task: str):
    model = SimpleCNN()
    model.load_state_dict(torch.load(MODEL_FILES[task], map_location="cpu"))
    model.eval()
    return model

def predict(task: str, image: Image.Image):
    image_tensor = transform(image).unsqueeze(0)
    model = load_model(task)
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).item()
    return CLASS_NAMES[task][pred]