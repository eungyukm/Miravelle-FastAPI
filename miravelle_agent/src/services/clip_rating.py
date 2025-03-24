# services/clip_rating.py
import torch
from PIL import Image
import clip

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP 모델 로드
clip_model, preprocess = clip.load("ViT-B/32", device=device)

def calculate_clip_score(image, input_text="default"):
    """
    입력 이미지와 텍스트에 대한 CLIP 점수를 계산합니다.
    """
    try:
        text_features = clip_model.encode_text(clip.tokenize(input_text).to(device))
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        image_features = clip_model.encode_image(image_tensor)

        similarity = torch.cosine_similarity(text_features, image_features).item()
        clip_score = similarity * 5 + 5

        # 점수를 1과 10 사이로 제한
        return max(1, min(10, clip_score))
    except Exception as e:
        raise Exception(f"CLIP 점수 계산 실패: {str(e)}")
