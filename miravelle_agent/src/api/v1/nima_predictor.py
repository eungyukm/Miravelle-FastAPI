import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class NIMARegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.75),
            nn.Linear(base_model.last_channel, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.regressor(x)
        return x.squeeze()

# 모델 로드 함수
def load_nima_model(path="nima_regression_model/pytorch_model.bin"):
    model = NIMARegressionModel()
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model