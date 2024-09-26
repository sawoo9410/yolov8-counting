"""
사용자가 간편하게 yolo 모델의 weight 파일을 다운로드 받을 수 있도록 만든 py file
installer.bat 파일을 통해 download.py가 실행
"""

import os
from requests import get  

def download(url, file_name = None):
    """
    주어진 URL에서 파일을 다운로드하여 로컬 시스템에 저장
    
    Parameters:
    - url (str): 다운로드할 파일의 URL.
    - file_name (str, 선택적): 파일을 저장할 때 사용할 이름. 기본값은 URL의 마지막 부분입니다.

    return:
    None
    """
    if not file_name:
        file_name = url.split('/')[-1]

    with open(os.path.join('./models',file_name), "wb") as file:   
        response = get(url)               
        file.write(response.content)     

if not os.path.isdir('./models'):
    os.makedirs('./models', exist_ok=True)

if __name__ == '__main__':
    base_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
    models = ["yolov8n", "yolov8s", "yolov8m", "yolov8l"]
    for model in models:
        url = base_url + model + '.pt'
        if os.path.isfile('./models/' + model + '.pt'):
            pass
        else:
            download(url)
            print('Download '+ url)
