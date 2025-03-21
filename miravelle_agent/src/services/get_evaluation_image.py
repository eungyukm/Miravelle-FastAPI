import requests

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