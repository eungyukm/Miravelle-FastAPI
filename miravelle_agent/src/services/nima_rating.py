# services/nima_rating.py
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# NIMA 모델 정의
class NIMA(nn.Module):
    def __init__(self, base_model):
        super(NIMA, self).__init__()
        self.base_model = base_model
        self.fc = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc(x)
        return torch.softmax(x, dim=1)

# NIMA 모델 로드
nima_base = resnet50(weights=ResNet50_Weights.DEFAULT)
nima_model = NIMA(nima_base).to(device)
nima_model.eval()

def calculate_nima_score(image):
    """
    입력 이미지에 대한 NIMA 점수를 계산합니다.
    """
    try:
        # 이미지 전처리
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)

        # 점수 계산
        with torch.no_grad():
            scores = nima_model(image_tensor).cpu().numpy()[0]
        mean_score = np.dot(scores, np.arange(1, 11))

        # 점수를 소수점 두 자리로 반올림
        return round(mean_score, 2)
    except Exception as e:
        raise Exception(f"NIMA 점수 계산 실패: {str(e)}")
