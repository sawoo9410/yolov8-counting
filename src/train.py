import os
import shutil
from ultralytics import YOLO


def pretrain_yolov8(DATASET_NAME, MODEL_NAME, EPOCHS, BATCH_SIZE, IMG_SIZE):
    """
    YOLOv8 모델을 커스텀 데이터셋으로 파인튜닝하고, .pt / .onnx 형식으로 저장합니다.

    Args:
        DATASET_NAME (str): 데이터셋 이름 (datasets/ 하위 폴더명)
        MODEL_NAME (str): 사전학습 모델 이름 (yolov8n, yolov8s, yolov8m, yolov8l)
        EPOCHS (int): 학습 에폭 수
        BATCH_SIZE (int): 배치 크기
        IMG_SIZE (int): 입력 이미지 크기
    """
    yaml_path = os.path.abspath(f"./datasets/{DATASET_NAME}/{DATASET_NAME}.yaml")

    # 사전학습 모델 로드 및 학습
    model = YOLO(f'./models/{MODEL_NAME}.pt')
    model.train(data=yaml_path, epochs=EPOCHS, batch=BATCH_SIZE, imgsz=IMG_SIZE, workers=1)

    # ONNX 형식으로 내보내기
    model.export(format='onnx', dynamic=True, opset=17)

    # 학습 결과를 results/ 디렉토리로 복사
    weights_dir = os.path.join(model.metrics.__dict__['save_dir'], 'weights')
    results_save_dir = os.path.join('./results', f'{DATASET_NAME}/{MODEL_NAME}')
    os.makedirs(results_save_dir, exist_ok=True)

    shutil.copy(os.path.join(weights_dir, 'best.pt'), results_save_dir)
    shutil.copy(os.path.join(weights_dir, 'best.onnx'), results_save_dir)

    # ultralytics가 생성한 runs/ 폴더 정리
    runs_dir = os.path.abspath("./runs")
    if os.path.exists(runs_dir):
        shutil.rmtree(runs_dir)
        print(f"Deleted {runs_dir} successfully.")
