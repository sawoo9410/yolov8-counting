import json
import os
import pathlib
from pathlib import Path
import glob
import shutil
from ultralytics import YOLO

def pretrain_yolov8(DATASET_NAME, MODEL_NAME, EPOCHS, BATCH_SIZE, IMG_SIZE):

    yaml_path = os.path.abspath(f"./datasets/{DATASET_NAME}/{DATASET_NAME}.yaml")

    # 모델과 학습 관련 파라미터 선택
    model = YOLO('./models/{}.pt'.format(MODEL_NAME))
    model.train(data=yaml_path, epochs=EPOCHS, batch=BATCH_SIZE, imgsz=IMG_SIZE, workers=1)
    model.export(format='onnx', dynamic=True, opset=17) 

    # 모델 저장 
    torch_save_dir = os.path.join(model.metrics.__dict__['save_dir'], 'weights/best.pt')
    onnx_save_dir = os.path.join(model.metrics.__dict__['save_dir'], 'weights/best.onnx')
    results_save_dir = os.path.join('./results', f'{DATASET_NAME}/{MODEL_NAME}')

    os.makedirs(results_save_dir, exist_ok=True)

    shutil.copy(torch_save_dir, results_save_dir)
    shutil.copy(onnx_save_dir, results_save_dir)

    # 모델 학습 후 runs 폴더 삭제
    runs_dir = os.path.abspath("./runs")
    if os.path.exists(runs_dir):
        shutil.rmtree(runs_dir)
        print(f"Deleted {runs_dir} successfully.")
    else:
        print(f"No runs directory found at {runs_dir}.")

# if __name__ == '__main__':
#     yolov8(DATASET_NAME, MODEL_NAME, EPOCHS, BATCH_SIZE, IMG_SIZE)
