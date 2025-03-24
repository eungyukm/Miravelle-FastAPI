import requests
import httpx
import urllib.parse
import re

def get_evaluation_image_random():
    url = "https://picsum.photos/500"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.content
        else:
            print(f"Failed to load image: Status code {response.status_code}")
            return None
    except Exception as e:
        print(f"Error while fetching image: {e}")
        return None


def get_image_from_miravell():
    url = "https://miravelle-appservice-dsecega7bbhvefem.koreacentral-01.azurewebsites.net/api/v1/vision/image/evaluate/"
    try:
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            response = client.get(url)
            if response.status_code == 200:
                data = response.json()

                # id 값 추출
                item_id = data.get("id")

                # image_path 값 추출 및 디코딩 처리
                image_path = data.get("image_path")
                if image_path:
                    # URL 디코딩 처리
                    decoded_image_url = urllib.parse.unquote(image_path)

                    # 정규식을 통해 중복된 프로토콜 자동 수정
                    decoded_image_url = re.sub(r'https?://+', 'https://', decoded_image_url)

                    # '/media/' 제거 후 URL 형식 유지
                    cleaned_url = decoded_image_url.replace("/media/", "")

                    # URL 형식 보정 후 중복 제거 처리
                    cleaned_url = re.sub(r"https://https://", "https://", cleaned_url)

                    # 슬래시 하나가 빠진 경우 자동 보정
                    cleaned_url = cleaned_url.replace("https:/", "https://")

                    # id와 URL 반환
                    return {
                        "id": item_id,
                        "image_url": cleaned_url
                    }
                else:
                    raise Exception("image_path가 없습니다.")
            else:
                raise Exception(f"API 호출 실패: 상태 코드 {response.status_code}")
    except Exception as e:
        raise Exception(f"API 호출 실패: {str(e)}")