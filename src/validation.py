import os
import glob
import shutil
import time
from tqdm import tqdm
import cv2
from ultralytics import YOLO


def validation_yolov8(DATASET_NAME, MODEL_NAME):
    """
    학습된 YOLOv8 모델로 이미지 추론을 수행하고, mAP50-95 성능을 평가합니다.

    Args:
        DATASET_NAME (str): 데이터셋 이름
        MODEL_NAME (str): 모델 이름
    """
    MODEL_PATH = os.path.abspath(f"./results/{DATASET_NAME}/{MODEL_NAME}/best.pt")
    TEST_RESULTS_DIR = os.path.abspath(f"./results/{DATASET_NAME}/{MODEL_NAME}/output_images")

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # 이전 결과 디렉토리 초기화
    if os.path.isdir(TEST_RESULTS_DIR):
        shutil.rmtree(TEST_RESULTS_DIR)
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

    # 모델 로드
    model = YOLO(MODEL_PATH)

    # 검증 이미지 목록 로드
    image_list = glob.glob(f'./datasets/{DATASET_NAME}/images/train/*.jpg', recursive=True)

    # 이미지별 추론 및 결과 저장
    # RuntimeError/ONNXRuntimeError 발생 시 추론 간 sleep을 추가하여 재시도
    start_time = time.time()
    infer_times = []
    sleep_interval = 0

    try:
        for i in tqdm(range(len(image_list)), desc="Inference"):
            time.sleep(sleep_interval)
            s_time = time.perf_counter()
            result = model.predict(image_list[i], conf=0.3)[0]
            e_time = time.perf_counter()

            if i != 0:
                infer_times.append(e_time - s_time)

            image_name = os.path.basename(result.path)
            res_plotted = result.plot()
            cv2.imwrite(os.path.join(TEST_RESULTS_DIR, image_name), res_plotted)

    except (RuntimeError, Exception) as e:
        # GPU 메모리 부족 또는 ONNX 런타임 에러 시, sleep 간격을 두고 재시도
        if isinstance(e, RuntimeError):
            print("RuntimeError detected, retrying with delay...")
            sleep_interval = 0.01
        elif 'ONNXRuntimeError' in str(e):
            print("ONNXRuntimeError detected, retrying with delay...")
            sleep_interval = 0.001
        else:
            raise

        for i in tqdm(range(len(image_list)), desc="Inference (retry)"):
            time.sleep(sleep_interval)
            s_time = time.perf_counter()
            result = model.predict(image_list[i], conf=0.3)[0]
            e_time = time.perf_counter()

            if i != 0:
                infer_times.append(e_time - s_time)

            image_name = os.path.basename(result.path)
            res_plotted = result.plot()
            cv2.imwrite(os.path.join(TEST_RESULTS_DIR, image_name), res_plotted)

    end_time = time.time()
    print(f"Total inference time: {end_time - start_time:.2f}s")
    if infer_times:
        print(f"Average inference time per image: {sum(infer_times)/len(infer_times):.4f}s")

    # mAP50-95 평가
    model.val()

    # 메트릭 출력
    names = model.metrics.__dict__['names']
    maps = model.metrics.__dict__['box'].maps

    header = '\n' + '-' * 20 + '\n     mAP50-95\n' + '-' * 20
    print(header)
    for i in range(len(names)):
        print(f'{names[i]:>10} : {maps[i]:.3f}')
    print('-' * 20)

    # 메트릭 파일 저장
    metrics_path = os.path.join(os.path.dirname(MODEL_PATH), 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f'     {MODEL_NAME}')
        f.write(header + '\n')
        for i in range(len(names)):
            f.write(f'{names[i]:>10} : {maps[i]:.3f}\n')

    # ultralytics가 생성한 runs/ 폴더 정리
    runs_dir = os.path.abspath("./runs")
    if os.path.exists(runs_dir):
        shutil.rmtree(runs_dir)
        print(f"Deleted {runs_dir} successfully.")
