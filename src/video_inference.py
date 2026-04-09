import os
import cv2
import numpy as np
from PIL import Image as Img
from tqdm import tqdm
from ultralytics import YOLO
from boxmot import BoTSORT
import torch

from src.utils.object_counter import ObjectCounter, CrossObjectCounter


def create_single_counter(region_points, class_names, ordered, descript):
    """
    단일 라인 기반 ObjectCounter를 생성합니다.

    Args:
        region_points (list): 카운팅 라인을 정의하는 2개의 점 [(x1,y1), (x2,y2)]
        class_names (dict): 클래스 이름 딕셔너리
        ordered (int): 화면에 표시할 카운트 텍스트의 순서
        descript (str): 라인 설명 텍스트

    Returns:
        ObjectCounter: 설정된 ObjectCounter 객체
    """
    counter = ObjectCounter()
    counter.set_args(
        view_img=False,
        reg_pts=region_points,
        classes_names=class_names,
        count_reg_color=(255, 255, 255),
        draw_tracks=True,
        view_in_counts=True,
        view_out_counts=True,
        count_txt_thickness=1,
        line_thickness=2,
        track_thickness=2,
        region_thickness=3,
        ordered=ordered,
        descript=descript,
    )
    return counter


def create_cross_counter(region_points_1, region_points_2, class_names, ordered, descript):
    """
    두 라인 교차 기반 CrossObjectCounter를 생성합니다.

    Args:
        region_points_1 (list): 첫 번째 라인을 정의하는 2개의 점
        region_points_2 (list): 두 번째 라인을 정의하는 2개의 점
        class_names (dict): 클래스 이름 딕셔너리
        ordered (int): 화면에 표시할 카운트 텍스트의 순서
        descript (str): 라인 설명 텍스트

    Returns:
        CrossObjectCounter: 설정된 CrossObjectCounter 객체
    """
    counter = CrossObjectCounter()
    counter.set_args(
        view_img=False,
        reg_pts_1=region_points_1,
        reg_pts_2=region_points_2,
        classes_names=class_names,
        count_reg_color=(255, 255, 255),
        draw_tracks=True,
        view_in_counts=True,
        view_out_counts=True,
        count_txt_thickness=1,
        line_thickness=2,
        track_thickness=2,
        region_thickness=3,
        line_dist_thresh=30,
        ordered=ordered,
        descript=descript,
    )
    return counter


def video_object_counting(video_filename, model_name, dataset_name, line_points_list):
    """
    영상에서 객체를 탐지 + 추적 + 카운팅하여 결과 영상을 저장합니다.

    Args:
        video_filename (str): 입력 영상 파일명 (datasets/<dataset_name>/ 내 위치)
        model_name (str): 사용할 YOLOv8 모델 이름
        dataset_name (str): 데이터셋 이름
        line_points_list (list): 카운팅 라인 좌표 목록
            - 4개 좌표: single line 모드 (In/Out 카운팅)
            - 8개 좌표: cross line 모드 (두 라인 사이 통과 감지)
    """
    # 영상 파일 경로 확인
    video_path = os.path.abspath(f"./datasets/{dataset_name}/{video_filename}")
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist.")
        return

    # 모델 로드 및 디바이스 설정
    MODEL_PATH = os.path.abspath(f"./results/{dataset_name}/{model_name}/best.pt")
    model = YOLO(MODEL_PATH)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    class_names = model.names

    # BoTSORT 트래커 초기화
    tracker = BoTSORT(
        reid_weights=None,
        half=False,
        device=device,
        with_reid=False,
        match_thresh=0.6,
    )

    # 카운팅 라인별 카운터 초기화
    # 좌표 4개 → single line, 8개 → cross line (두 선 사이 통과 감지)
    counters = []
    line_count = 1
    for i, line_points in enumerate(line_points_list):
        if len(line_points) == 8:
            # Cross line 모드: 8개 좌표를 두 라인으로 분리
            line_p1 = (line_points[0], line_points[1])
            line_p2 = (line_points[2], line_points[3])
            line_p3 = (line_points[4], line_points[5])
            line_p4 = (line_points[6], line_points[7])
            counters.append(create_cross_counter(
                [line_p1, line_p2], [line_p3, line_p4],
                class_names, ordered=i + 1,
                descript=f"Line{line_count}->Line{line_count + 1}",
            ))
            line_count += 2
        else:
            # Single line 모드: 4개 좌표를 하나의 라인으로 사용
            line_p1 = (line_points[0], line_points[1])
            line_p2 = (line_points[2], line_points[3])
            counters.append(create_single_counter(
                [line_p1, line_p2],
                class_names, ordered=i + 1,
                descript=f"Counting Line{i + 1}",
            ))
            line_count += 1

    # 영상 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 출력 영상 설정
    output_directory = os.path.abspath(f"./results/{dataset_name}/{model_name}/video_counting")
    os.makedirs(output_directory, exist_ok=True)
    output_video_path = os.path.join(output_directory, f'output_counting_{video_filename}')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print(f"Processing video... Total frames: {total_frames}")

    # 프레임별 처리: 탐지 → 추적 → 카운팅
    for _ in tqdm(range(total_frames), desc="Processing Frames"):
        ret, frame = cap.read()
        if not ret:
            break

        pil_image = Img.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = model.predict(pil_image, conf=0.4, device=device)

        # 탐지 결과를 BoTSORT 입력 형식으로 변환
        detections = []
        for result in results:
            for box in result.boxes:
                xmin, ymin, xmax, ymax = box.xyxy.tolist()[0]
                score = box.conf.item()
                cls = int(box.cls.item())
                detections.append([xmin, ymin, xmax, ymax, score, cls])

        detections = np.array(detections)

        # 객체 추적
        tracked_objects = tracker.update(detections, frame)

        # 추적 결과를 카운터에 전달
        track_dict = {
            'boxes': tracked_objects[:, :4],
            'class': tracked_objects[:, 5].astype(int).tolist(),
            'track_id': tracked_objects[:, 4].astype(int).tolist(),
        }

        # 각 카운터로 카운팅 수행 및 프레임에 시각화
        for counter in counters:
            frame = counter.start_counting(frame, track_dict)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Processing complete. Output saved to {output_video_path}")
