from fastapi import FastAPI, UploadFile, File
import os
import torch
import ImageReward as reward
from PIL import Image

# FastAPI 앱 생성
app = FastAPI()

# 이미지 저장 폴더 설정
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🔥 ImageReward 모델 로드 확인
print("🔥 ImageReward 모델 로드 중...")
try:
    model = reward.load("ImageReward-v1.0")
    print("✅ 모델 로드 완료")
except Exception as e:
    print(f"❌ 모델 로드 실패: {str(e)}")

# ✅ 1️⃣ 여러 개 이미지 비교 → 가장 선호도 높은 이미지 반환
@app.post("/compare/")
async def compare_images(files: list[UploadFile] = File(...)):
    img_paths = []

    for file in files:
        img_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(img_path, "wb") as buffer:
            buffer.write(await file.read())

        # 🔍 PIL을 사용해 이미지가 정상적으로 열리는지 확인
        try:
            img = Image.open(img_path)
            img.verify()  # 손상된 파일 검사
        except Exception as e:
            return {"error": f"❌ 올바르지 않은 이미지 파일입니다: {file.filename}, {str(e)}"}

        img_paths.append(img_path)

    # ✅ 업로드된 파일 확인
    print(f"📂 업로드된 파일 목록: {img_paths}")

    # 🔥 최소 2개 이상의 이미지를 업로드해야 비교 가능
    if not img_paths or len(img_paths) < 2:
        return {"error": "❌ 최소 2개의 이미지를 업로드해야 비교할 수 있습니다."}

    prompt = "a high-quality and aesthetically pleasing image"

    # ✅ 모델이 파일을 정상적으로 읽을 수 있는지 확인
    for img_path in img_paths:
        try:
            with open(img_path, "rb") as f:
                f.read()  # 파일이 정상적으로 열리는지 확인
        except Exception as e:
            return {"error": f"❌ ImageReward가 파일을 읽을 수 없습니다: {img_path}, {str(e)}"}

    # 🔥 모델 실행
    try:
        with torch.no_grad():
            ranking, scores = model.inference_rank(prompt, img_paths)
        print(f"🔥 모델 평가 결과: ranking={ranking}, scores={scores}")
    except Exception as e:
        return {"error": f"❌ ImageReward 평가 실패: {str(e)}"}

    # 🔥 ranking이 비어있는지 확인
    if not ranking:
        return {"error": "❌ 이미지 평가에 실패했습니다. 파일이 손상되지 않았는지 확인하세요."}

    best_img = img_paths[ranking[0]]

    return {
        "best_image": best_img,
        "scores": {img_paths[i]: scores[i] for i in range(len(img_paths))}
    }


# ✅ 2️⃣ 단일 이미지 평가 → 선호도 점수 반환
@app.post("/evaluate/")
async def evaluate_image(file: UploadFile = File(...)):
    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(img_path, "wb") as buffer:
        buffer.write(await file.read())

    # 🔍 PIL을 사용해 이미지가 정상적으로 열리는지 확인
    try:
        img = Image.open(img_path)
        img.verify()  # 손상된 파일 검사
    except Exception as e:
        return {"error": f"❌ 올바르지 않은 이미지 파일입니다: {file.filename}, {str(e)}"}

    # 평가 프롬프트 (좋은 이미지 표현)
    prompt = "a high-quality and aesthetically pleasing image"

    # 🔥 모델 실행
    try:
        with torch.no_grad():
            score = model.score(prompt, img_path)
        print(f"🔥 {img_path} 평가 점수: {score:.2f}")
    except Exception as e:
        return {"error": f"❌ ImageReward 평가 실패: {str(e)}"}

    return {"image": img_path, "score": score}