# services/utils.py

import requests
from fastapi import HTTPException

def get_evaluation_image_random():
    """
    랜덤 이미지를 다운로드합니다.
    """
    url = "https://picsum.photos/200/300"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.content
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to download random image")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Random image download failed: {str(e)}")
