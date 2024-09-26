# src/video_inference.py

import os
import cv2
import numpy as np
from PIL import Image as Img
from src.utils.object_counter import ObjectCounter, CrossObjectCounter
from boxmot import BoTSORT
from tqdm import tqdm
from ultralytics import YOLO
import torch


def counter_ob(region_points, class_names, ordered, descript):
    """
    데이터 스트림에서 객체를 추적하고 계산하는 기능을 설정하기 위한 함수입니다.

    Args:
        region_points (list): 객체 계산을 위한 지역을 정의하는 점들의 목록입니다.
        class_names (list): 추적할 객체의 클래스 이름입니다.
        ordered (bool): 객체 수를 보고할 때 순서를 지정할지 여부를 결정하는 플래그입니다.
        descript (str): 지역 또는 기능에 대한 설명입니다.

    Returns:
        ObjectCounter: 설정된 매개변수로 초기화된 ObjectCounter 객체를 반환합니다.

    이 함수는 다양한 시각적 매개변수를 설정하여 ObjectCounter의 동작을 사용자 정의할 수 있게 합니다.
    주요 설정에는 이미지 표시 여부, 트랙 그리기 활성화, 계산된 텍스트의 두께 등이 포함됩니다.
    """
    counter = ObjectCounter()
    counter.set_args(view_img=False,  # 실시간 비디오 스트림을 표시할지 여부
                    reg_pts=region_points,  # 객체 계산 지역 정의 점
                    classes_names=class_names,  # 추적할 클래스의 이름
                    count_reg_color=(255, 255, 255),  # 객체 계산 지역 색상
                    draw_tracks=True,  # 트랙을 그릴지 여부
                    view_in_counts=True,  # 들어오는 객체 수를 표시할지 여부
                    view_out_counts=True,  # 나가는 객체 수를 표시할지 여부
                    count_txt_thickness=1,  # 계산된 텍스트의 두께
                    line_thickness=2,  # 바운딩 박스 라인의 굵기
                    track_thickness=2,  # 트랙 중심 점의 굵기
                    region_thickness=3,  # 계산 지역의 라인 굵기
                    ordered=ordered,  # 결과의 순서를 지정
                    descript=descript)  # 지역 또는 기능 설명

    return counter


def cross_counter_ob(region_points_1, region_points_2, class_names, ordered, descript, count_reg_color=(255, 255, 255)):
    """
    데이터 스트림에서 객체를 추적하고 두 개의 선을 교차하는 객체를 계산하는 기능을 설정하기 위한 함수입니다.

    Args:
        region_points_1 (list): 첫 번째 선을 정의하는 점들의 목록입니다.
        region_points_2 (list): 두 번째 선을 정의하는 점들의 목록입니다.
        class_names (list): 추적할 객체의 클래스 이름입니다.
        ordered (bool): 객체 수를 보고할 때 순서를 지정할지 여부를 결정하는 플래그입니다.
        descript (str): 지역 또는 기능에 대한 설명입니다.

    Returns:
        CrossObjectCounter: 설정된 매개변수로 초기화된 CrossObjectCounter 객체를 반환합니다.

    이 함수는 다양한 시각적 매개변수를 설정하여 CrossObjectCounter의 동작을 사용자 정의할 수 있게 합니다.
    주요 설정에는 이미지 표시 여부, 트랙 그리기 활성화, 계산된 텍스트의 두께 등이 포함됩니다.
    """
    counter = CrossObjectCounter()
    counter.set_args(view_img=False,  # 실시간 비디오 스트림을 표시할지 여부
                     reg_pts_1=region_points_1,  # 첫 번째 객체 계산 선 정의 점
                     reg_pts_2=region_points_2,  # 두 번째 객체 계산 선 정의 점
                     classes_names=class_names,  # 추적할 클래스의 이름
                     count_reg_color=count_reg_color,  # 객체 계산 지역 색상
                     draw_tracks=True,  # 트랙을 그릴지 여부
                     view_in_counts=True,  # 들어오는 객체 수를 표시할지 여부
                     view_out_counts=True,  # 나가는 객체 수를 표시할지 여부
                     count_txt_thickness=1,  # 계산된 텍스트의 두께
                     line_thickness=2,  # 바운딩 박스 라인의 굵기
                     track_thickness=2,  # 트랙 중심 점의 굵기
                     region_thickness=3,  # 계산 지역의 라인 굵기
                     line_dist_thresh=30,  # 선 거리 임계값
                     ordered=ordered,  # 결과의 순서를 지정
                     descript=descript)  # 지역 또는 기능 설명

    return counter

    
# def video_object_counting(video_filename, model_name, dataset_name, line_points, cross_mode=False):
#     """
#     동영상에서 객체를 검출, 추적하고 카운팅하여 결과를 저장합니다.

#     Args:
#         video_filename (str): 입력 동영상 파일 이름 (./datasets/디렉토리에 위치).
#         model_name (str): 사용할 YOLOv8 모델 이름 (e.g., 'yolov8n').
#         dataset_name (str): 데이터셋 이름.
#         line_points (list of tuples): 카운팅 라인의 두 점 [(x1, y1), (x2, y2)].
#         cross_mode (bool): 두 선을 사용한 교차 카운팅 모드 활성화 여부.
#     """
#     # 비디오 파일 경로 지정
#     video_path = os.path.abspath(f"./datasets/{dataset_name}/{video_filename}")
#     if not os.path.exists(video_path):
#         print(f"Error: Video file {video_path} does not exist.")
#         return

#     # 모델 로드 및 디바이스 설정
#     MODEL_PATH = os.path.abspath(f"./results/{dataset_name}/{model_name}/best.pt")

#     model = YOLO(MODEL_PATH)
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model.to(device)

#     # 클래스 이름 로드
#     class_names = model.names

#     # BoTSORT 트래커 초기화
#     tracker = BoTSORT(
#         reid_weights=None,      # ReID 모델 가중치 경로 (ReID 사용하지 않으므로 None 또는 "")
#         half=False,             # FP16 사용 여부 (GPU 사용 시 True)
#         device=device,          # 연산 장치 ('cpu' 또는 'cuda')
#         with_reid=False,        # ReID 사용 여부
#         match_thresh=0.6,       # 매칭 임계값
#     )


#     # ObjectCounter 또는 CrossObjectCounter 초기화 및 설정
#     if cross_mode:
#         if len(line_points) != 4:
#             print("Error: For cross_mode, provide two lines with eight integers (x1 y1 x2 y2 x3 y3 x4 y4).")
#             return
#         # 두 개의 선을 언팩
#         line_p1, line_p2, line_p3, line_p4 = line_points
#         counter = cross_counter_ob([line_p1, line_p2], [line_p3, line_p4], class_names, ordered=1, descript="Line1->Line2")

#     else:
#         if len(line_points) != 2:
#             print("Error: For single line mode, provide two points (x1 y1) and (x2 y2).")
#             return
#         # 두 개의 선을 언팩
#         line_p1, line_p2 = line_points
#         counter = cross_counter_ob([line_p1, line_p2], class_names, ordered=1, descript="Counting Line")

#     # 비디오 캡처
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error: Unable to open video file {video_path}")
#         return

#     # 비디오 속성 가져오기
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     # 출력 비디오 경로 지정
#     output_directory = os.path.abspath(f"./results/{dataset_name}/{model_name}/video_counting")
#     os.makedirs(output_directory, exist_ok=True)
#     output_video_path = os.path.join(output_directory, f'output_counting_{video_filename}')

#     # 비디오 라이터 초기화
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

#     print(f"Processing video... Total frames: {total_frames}")

#     # 프레임 처리
#     for _ in tqdm(range(total_frames), desc="Processing Frames"):
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # PIL 이미지로 변환
#         pil_image = Img.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#         # 객체 검출
#         results = model.predict(pil_image, conf=0.4, device=device)  # 신뢰도 임계값 조정 가능

#         # 검출 결과를 BoTSORT에 입력할 형식으로 변환
#         detections = []
#         for result in results:
#             for box in result.boxes:
#                 xmin, ymin, xmax, ymax = box.xyxy.tolist()[0]
#                 score = box.conf.item()
#                 cls = int(box.cls.item())
#                 detections.append([xmin, ymin, xmax, ymax, score, cls])

#         detections = np.array(detections)

#         # BoTSORT를 사용한 객체 추적
#         tracked_objects = tracker.update(detections, frame)

#         # 추적 결과를 ObjectCounter 또는 CrossObjectCounter에 전달
#         track_dict = {
#             'boxes': tracked_objects[:, :4],       # 바운딩 박스 좌표 (NumPy 배열)
#             'class': tracked_objects[:, 5].astype(int).tolist(),  # 클래스 인덱스
#             'track_id': tracked_objects[:, 4].astype(int).tolist() # 트랙 ID
#         }


#         # 객체 카운팅 및 주석 추가
#         annotated_frame = counter.start_counting(frame, track_dict)

#         # 결과 프레임 저장
#         out.write(annotated_frame)

#     # 자원 해제
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

#     if cross_mode:
#         print(f"Processing complete. Output saved to {output_video_path}")
#         print(f"Total objects crossed: {counter.cross_counts}")
#     else:
#         print(f"Processing complete. Output saved to {output_video_path}")
#         print(f"Total objects counted: {counter.in_counts} In, {counter.out_counts} Out")

def video_object_counting(video_filename, model_name, dataset_name, line_points_list):
    video_path = os.path.abspath(f"./datasets/{dataset_name}/{video_filename}")
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist.")
        return

    MODEL_PATH = os.path.abspath(f"./results/{dataset_name}/{model_name}/best.pt")
    model = YOLO(MODEL_PATH)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    class_names = model.names
    tracker = BoTSORT(reid_weights=None,
                      half=False,
                      device=device,
                      with_reid=False,
                      match_thresh=0.6)

    # 여러 카운터 초기화
    counters = []
    line_count = 1
    for i, line_points in enumerate(line_points_list):
        if len(line_points) == 8:
            line_p1 = (line_points[0], line_points[1])
            line_p2 = (line_points[2], line_points[3])
            line_p3 = (line_points[4], line_points[5])
            line_p4 = (line_points[6], line_points[7])
            counters.append(cross_counter_ob([line_p1, line_p2], [line_p3, line_p4], class_names, ordered=i+1, descript=f"Line{line_count}->Line{line_count+1}"))
            line_count += 2
        else:
            line_p1 = (line_points[0], line_points[1])  # 리스트의 첫 번째 두 좌표를 튜플로 변환
            line_p2 = (line_points[2], line_points[3])  # 리스트의 두 번째 두 좌표를 튜플로 변환
            counters.append(counter_ob([line_p1, line_p2], class_names, ordered=i+1, descript=f"Counting Line{i+1}"))
            line_count += 1

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_directory = os.path.abspath(f"./results/{dataset_name}/{model_name}/video_counting")
    os.makedirs(output_directory, exist_ok=True)
    output_video_path = os.path.join(output_directory, f'output_counting_{video_filename}')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print(f"Processing video... Total frames: {total_frames}")

    for _ in tqdm(range(total_frames), desc="Processing Frames"):
        ret, frame = cap.read()
        if not ret:
            break

        pil_image = Img.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = model.predict(pil_image, conf=0.4, device=device)

        detections = []
        for result in results:
            for box in result.boxes:
                xmin, ymin, xmax, ymax = box.xyxy.tolist()[0]
                score = box.conf.item()
                cls = int(box.cls.item())
                detections.append([xmin, ymin, xmax, ymax, score, cls])

        detections = np.array(detections)
        tracked_objects = tracker.update(detections, frame)

        track_dict = {
            'boxes': tracked_objects[:, :4],
            'class': tracked_objects[:, 5].astype(int).tolist(),
            'track_id': tracked_objects[:, 4].astype(int).tolist()
        }

        for counter in counters:
            frame = counter.start_counting(frame, track_dict)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # for i, counter in enumerate(counters):
    #     print(f"Counter {i+1}: In: {counter.in_counts}, Out: {counter.out_counts}")
