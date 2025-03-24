import requests
import httpx
import urllib.parse

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
        with httpx.Client() as client:
            response = client.get(url)
            if response.status_code == 200:
                data = response.json()
                image_path = data.get("image_path")
                if image_path:
                    # URL 디코딩 처리 (예: https%3A → https://)
                    decoded_image_url = urllib.parse.unquote(image_path)

                    # "/media/" 제거
                    cleaned_url = decoded_image_url.replace("/media/", "")

                    return cleaned_url
                else:
                    raise Exception("image_path가 없습니다.")
            else:
                raise Exception(f"API 호출 실패: 상태 코드 {response.status_code}")
    except Exception as e:
        raise Exception(f"API 호출 실패: {str(e)}")