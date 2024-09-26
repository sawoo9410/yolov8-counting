from ultralytics import YOLO

import json
import os
import pathlib
import glob
import subprocess
import shutil
from tqdm import tqdm
import cv2
import time
import yaml


def validation_yolov8(DATASET_NAME, MODEL_NAME):
    # 저장된 모델의 경로, 데이터셋 이름, 모델 이름, 테스트 결과를 저장할 디렉터리를 가져옴
    # model_path, dataset_name, model_name, test_results_dir = get_saved_model()

    MODEL_PATH = os.path.abspath(f"./results/{DATASET_NAME}/{MODEL_NAME}/best.pt")
    TEST_RESULTS_DIR = os.path.abspath(f"./results/{DATASET_NAME}/{MODEL_NAME}/output_images")

    # 모델 경로가 비어있는 경우 스크립트 종료
    if MODEL_PATH == '':
        exit()
    
    # 테스트 결과를 저장할 디렉터리가 이미 존재하는 경우 삭제
    if os.path.isdir(TEST_RESULTS_DIR):
        shutil.rmtree(TEST_RESULTS_DIR)

    # 테스트 결과를 저장할 디렉터리 생성
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

    # YOLO 모델 로드
    start_time = time.time()
    model = YOLO(MODEL_PATH)
    
    # validation 이미지 리스트에서 처음 100개만 가져오기
    image_list = glob.glob(f'./datasets/{DATASET_NAME}/images/train/*.jpg', recursive=True)#[:100]

    try:
        print("not error")

        list_time = []
 
        # 모델을 사용해 이미지 리스트에 대한 예측 수행
        for i in tqdm(range(len(image_list))): 
            # 시작
            s_time = time.perf_counter()
            result = model.predict(image_list[i], conf=0.3)[0]
            
            # 종료
            e_time = time.perf_counter()
            print('inf time:', e_time-s_time)
            if i != 0:
                list_time.append(e_time-s_time)
            image_name = os.path.basename(result.path)
            res_plotted = result.plot()
            cv2.imwrite(os.path.join(TEST_RESULTS_DIR, image_name), res_plotted)

        print("Average inf time: ", sum(list_time)/len(list_time))

    except RuntimeError:
        print("Runtime Error")
        # 예외 발생 시 이미지를 하나씩 예측
        for i in tqdm(range(len(image_list))): 
            time.sleep(0.01)
            # 시작
            s_time = time.time()
            result = model.predict(image_list[i], conf=0.3)[0]
            # 종료
            e_time = time.time()
            print('inf time:', e_time-s_time)
            image_name = os.path.basename(result.path)
            res_plotted = result.plot()
            cv2.imwrite(os.path.join(TEST_RESULTS_DIR, image_name), res_plotted)

    except Exception as e:
        # ONNXRuntimeError 예외 발생 시 이미지를 하나씩 예측
        print("ONNX Runtime Error")
        if 'ONNXRuntimeError' in str(e):
            for i in tqdm(range(len(image_list))): 
                time.sleep(0.001)
                s_time = time.time()
                result = model.predict(image_list[i], conf=0.3)[0]
                e_time = time.time()
                print('inf time:', e_time-s_time)

                image_name = os.path.basename(result.path)
                res_plotted = result.plot()
                cv2.imwrite(os.path.join(TEST_RESULTS_DIR, image_name), res_plotted)
                
    # 모델의 validation metrics 계산
    end_time = time.time()
    print('total inf time:', end_time-start_time)
    print(end_time, start_time)

    model.val()

    # 계산된 메트릭을 화면에 출력
    head = '\n'+'-'*20+'\n     mAP50-95\n'+'-'*20
    print(head)
    for i in range(len(model.metrics.__dict__['names'])):
        print('{0:>10} : {1:0.3f}'.format(model.metrics.__dict__['names'][i], model.metrics.__dict__['box'].maps[i]))
    print('-'*20)

    # 계산된 메트릭을 파일에 저장
    with open(os.path.join(os.path.dirname(MODEL_PATH), 'metrics.txt'), 'w') as f:
        f.write('     ' + MODEL_NAME)
        f.write(head+'\n')
        for i in range(len(model.metrics.__dict__['names'])):
            message = '{0:>10} : {1:0.3f}\n'.format(model.metrics.__dict__['names'][i], model.metrics.__dict__['box'].maps[i])
            f.write(message)

    # 모델 학습 후 runs 폴더 삭제
    runs_dir = os.path.abspath("./runs")
    if os.path.exists(runs_dir):
        shutil.rmtree(runs_dir)
        print(f"Deleted {runs_dir} successfully.")
    else:
        print(f"No runs directory found at {runs_dir}.")

    return


# if __name__ == '__main__':
#     pass