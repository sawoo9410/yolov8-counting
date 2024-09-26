import numpy as np
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                InputVStreamParams, OutputVStreamParams, FormatType)

import cv2

from PIL import Image as Img
import os
import yaml
import time

from utils import object_counter
from boxmot import BoTSORT

from tkinter import filedialog, Button
import tkinter.ttk as ttk
from tkinter import *
import tkinter as tk
from tkinter import Tk

target = None
target = VDevice()

DL_TYPE = 'object_counting' # Set inference type


##################################
############ GUI CODE ############
##################################
def get_yaml_file():
    """
    파일 다이얼로그를 사용하여 YAML 파일을 선택하고, 선택된 파일의 경로와 데이터셋 이름을 반환합니다.
    
    반환값:
    - yaml_path (str): 선택된 YAML 파일의 경로.
    - datasets_name (str): 선택된 파일이 위치한 디렉토리의 이름.
    """
    root = Tk()  # Tkinter 루트 윈도우 생성
    root.withdraw()  # 빈 윈도우 숨기기
    yaml_path = filedialog.askopenfilename(initialdir='./datasets', 
                                           title="데이터를 선택해주세요")  # 파일 다이얼로그 열기, 초기 디렉토리 설정 및 제목 지정
    datasets_name = os.path.dirname(yaml_path).split('/')[-1]  # 선택된 파일의 디렉토리 이름 가져오기
    root.destroy()  # Tkinter 루트 윈도우 파괴
    return yaml_path, datasets_name  # 파일 경로와 데이터셋 이름 반환


class HailoGUI:
    def __init__(self):
        self.user_input = {'video_path': None, 'model_name': None}
        self.root = tk.Tk()
        self.root.title("HAILO GUI")
        self.root.geometry("700x200+400+400")
        self.root.configure(bg='white')
        
        self.frame = tk.Frame(self.root, bg='white')
        self.frame.place(relx=0.5, rely=0.5, anchor='center')

        self.create_widgets()
        self.root.mainloop()

    def create_widgets(self):
        # '모델' 라벨 추가
        tk.Label(self.frame, text='모델', font=('맑은 고딕', 16, 'bold'), bg='white', width=12, height=1).grid(column=0, row=0, padx=5, pady=10)

        # 모델 선택 콤보박스 추가
        model_values = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l']
        self.model_combobox = ttk.Combobox(self.frame, values=model_values, state='readonly', width=15, font=('맑은 고딕', 12))
        self.model_combobox.grid(column=1, row=0, padx=5, pady=10)
        self.model_combobox.current(1)
        
        # 설명 라벨 추가
        tk.Label(self.frame, text='분석할 모델을 선택하세요.', font=('맑은 고딕', 10), bg='white', anchor='w').grid(column=3, row=0, sticky='w')

        # '비디오 파일' 라벨 추가
        tk.Label(self.frame, text='비디오 파일', font=('맑은 고딕', 16, 'bold'), bg='white', width=12, height=1).grid(column=0, row=1, padx=5, pady=10)

        # 비디오 파일 선택 버튼 추가
        self.video_path_entry = tk.Entry(self.frame, width=15, font=('맑은 고딕', 12))
        self.video_path_entry.grid(column=1, row=1, padx=5, pady=15)
        tk.Button(self.frame, text='파일 선택', font=('맑은 고딕', 12), command=self.select_file).grid(column=2, row=1, padx=5, pady=10)
        
        # 설명 라벨 추가
        tk.Label(self.frame, text='분석할 비디오 파일을 선택하세요.', font=('맑은 고딕', 10), bg='white', anchor='w').grid(column=3, row=1, sticky='w')

        # '제출' 버튼 추가 및 배치
        tk.Button(self.frame, text='제출', font=('맑은 고딕', 14, 'bold'), bg='skyblue', fg='white', height=1, command=self.on_submit).grid(column=1, row=2, padx=5, pady=20)

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if file_path:
            self.video_path_entry.delete(0, tk.END)
            self.video_path_entry.insert(0, file_path)
            self.user_input['video_path'] = file_path

    def on_submit(self):
        self.user_input['model_name'] = self.model_combobox.get()
        self.root.quit()

    def get_user_input(self):
        return self.user_input['video_path'], self.user_input['model_name']


##################################


def create_directory_if_not_exists(directory_path):
    """
    주어진 경로에 디렉토리가 없으면 생성합니다.

    매개변수:
    - directory_path (str): 생성할 디렉토리 경로
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    return


def letterbox_image(image, size):
    """
    주어진 크기에 맞게 이미지의 비율을 유지하면서 이미지를 리사이즈하고 패딩을 적용하는 함수입니다.
    만약 이미지가 지정된 크기보다 작을 경우, 주변을 색상 (114, 114, 114)으로 채웁니다.

    매개변수:
        image (PIL.Image.Image): 리사이즈할 PIL 이미지 객체.
        size (tuple): 리사이즈 및 패딩을 적용할 대상 크기 (너비, 높이).

    반환값:
        PIL.Image.Image: 비율을 유지하며 리사이즈하고 패딩된 이미지.
    """

    # 이미지의 원래 크기 추출
    img_w, img_h = image.size
    model_input_w, model_input_h = size

    # 이미지의 비율을 유지하면서 리사이즈할 스케일 계산
    scale = min(model_input_w / img_w, model_input_h / img_h)
    scaled_w = int(img_w * scale)
    scaled_h = int(img_h * scale)

    # 지정된 스케일로 이미지 리사이즈
    image = image.resize((scaled_w, scaled_h), resample=Img.Resampling.BICUBIC)

    # 패딩을 적용할 새 이미지 생성
    new_image = Img.new('RGB', size, (114, 114, 114))

    # 리사이즈된 이미지를 새 이미지 중앙에 붙여넣기
    new_image.paste(image, ((model_input_w - scaled_w) // 2, (model_input_h - scaled_h) // 2))

    del image
    
    return new_image


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
    counter = object_counter.ObjectCounter()
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


def post_process_track(detections, raw_width, raw_height, width, height, num_of_classes, min_score=0.5, scale_factor=1):
    """
    검출된 객체의 데이터를 후처리하여 화면 크기에 맞게 조정하고 필터링하는 함수입니다.

    Args:
        detections (list): 객체 검출 결과 리스트로, 각 클래스별로 검출된 객체의 바운딩 박스와 점수가 포함되어 있습니다.
        raw_width (int): 원본 이미지의 너비입니다.
        raw_height (int): 원본 이미지의 높이입니다.
        width (int): 출력 화면의 너비입니다.
        height (int): 출력 화면의 높이입니다.
        num_of_classes (int): 검출 가능한 클래스의 총 개수입니다.
        min_score (float, optional): 객체를 유효한 것으로 간주하기 위한 최소 점수입니다. 기본값은 0.5입니다.
        scale_factor (float, optional): 확대/축소 비율을 조정하는 인자입니다. 기본값은 1입니다.

    Returns:
        tuple: 처리된 결과를 두 개의 리스트로 반환합니다.
            - pred_dict (dict): 조정된 바운딩 박스, 클래스 라벨, 점수가 포함된 사전입니다.
            - pred_list (list): 조정되고 필터링된 바운딩 박스와 해당 점수, 클래스 인덱스를 포함하는 리스트입니다.

    각 바운딩 박스는 화면 크기에 맞게 조정되며, 설정된 `min_score` 이상의 점수를 가진 객체만 결과에 포함됩니다.
    """
    
    # 기본적인 화면 크기 조정 계수를 계산합니다.
    scale = min(width / raw_width, height / raw_height)
    x_scale = raw_width * scale
    y_scale = raw_height * scale

    # 이미지의 패딩 값을 계산합니다.
    x_padding = (width - x_scale) / 2
    y_padding = (height - y_scale) / 2

    pred_list = []
    pred_dict = {
        "boxes": [],
        "labels": [],
        "scores": []
    }

    # 각 클래스 별로 검출된 객체를 확인하고 처리합니다.
    for cls_idx in range(num_of_classes):
        if len(detections[cls_idx]) == 0:
            print("This image is not detected.")
        else:
            for idx in range(len(detections[cls_idx])):
                if detections[cls_idx][idx][4] > min_score:
                    # 검출된 바운딩 박스를 화면 크기에 맞게 조정합니다.
                    scaled_box = [x * width if i % 2 else x * height for i, x in enumerate(detections[cls_idx][idx][:4])]
                    scaled_box[0], scaled_box[1], scaled_box[2], scaled_box[3] = scaled_box[1], scaled_box[0], scaled_box[3], scaled_box[2]

                    # 조정된 좌표를 원래 이미지의 크기로 복원합니다.
                    original_x1 = (scaled_box[0] - x_padding) * (raw_width / x_scale)
                    original_y1 = (scaled_box[1] - y_padding) * (raw_height / y_scale)
                    original_x2 = (scaled_box[2] - x_padding) * (raw_width / x_scale)
                    original_y2 = (scaled_box[3] - y_padding) * (raw_height / y_scale)

                    # 결과 리스트에 추가합니다.
                    pred_list.append([original_x1, original_y1, original_x2, original_y2,
                                      detections[cls_idx][idx][4],
                                      cls_idx])
                    
                    pred_dict["boxes"].append(scaled_box)
                    pred_dict["labels"].append(cls_idx)
                    pred_dict["scores"].append(detections[cls_idx][idx][4])

    return pred_dict, pred_list


class HailoInferences:
    def __init__(self, path, DL_TYPE):
        if not path: 
            return

        # load hef file
        model = HEF(path) # get_hef(path)
        print(path)
        print(DL_TYPE)

        configure_params = ConfigureParams.create_from_hef(hef=model, interface=HailoStreamInterface.PCIe)
        self.network_group = target.configure(model, configure_params)[0]
        self.network_group_params = self.network_group.create_params()

        self.input_vstreams_params = InputVStreamParams.make(self.network_group, quantized=False, format_type=FormatType.AUTO)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group, quantized=False, format_type=FormatType.FLOAT32)

        self.input_vstream_info = model.get_input_vstream_infos()
        self.output_vstream_info = model.get_output_vstream_infos()     

        self.height, self.width, _ = self.input_vstream_info[0].shape

        del configure_params


    def object_detection(self, image, yaml_info, insp_option, min_score=0.5):
        """
        주어진 이미지에서 객체를 탐지하고 추적 준비 과정을 수행하는 함수입니다.

        Args:
            image (PIL.Image or np.array): 입력 이미지, PIL 이미지 객체 또는 numpy 배열 형식일 수 있습니다.
            yaml_info (dict): 모델 및 클래스 관련 정보가 담긴 YAML 파일 정보입니다.
            insp_option (Any): 검사 옵션, 이 함수에서는 구체적으로 사용되지 않습니다.
            min_score (float, optional): 객체 탐지를 유효하게 간주하기 위한 최소 점수입니다. 기본값은 0.5입니다.

        Returns:
            tuple: 탐지된 객체 정보, 처리된 프레임 이미지, 단일 추론 시간, 전체 추론 시간을 포함하는 튜플을 반환합니다.

        이 메서드는 입력 이미지를 받아서 객체 탐지 모델을 통해 객체를 탐지하고, 탐지된 객체에 대해 후처리를 수행합니다.
        처리된 이미지와 탐지 정보를 DeepSort 추적기에 업데이트하기 위해 준비합니다.
        """

        num_of_classes = yaml_info['nc']  # YAML 파일에서 클래스 수를 가져옵니다.

        start_full = time.time()  # 전체 추론 시작 시간을 기록합니다.
        
        # 입력 이미지가 PIL.Image 객체가 아닐 경우, PIL.Image 객체로 변환합니다.
        if not isinstance(image, Img.Image):
            image = Img.fromarray(image)
        
        print(image.size)
        raw_width, raw_height = image.size

        # 입력 이미지를 모델 입력 크기에 맞게 조정합니다.
        processed_image = letterbox_image(image, (self.height, self.width))

        start = time.time()
        
        # 모델 추론 파이프라인을 설정하고 추론을 수행합니다.
        with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            input_data = {self.input_vstream_info[0].name: np.expand_dims(processed_image, axis=0).astype(np.uint8)}
            with self.network_group.activate(self.network_group_params):
                raw_detection = infer_pipeline.infer(input_data)
            del infer_pipeline  # 추론 파이프라인 해제
        end = time.time()
        infer_time = end - start  # 단일 추론 시간을 계산합니다.

        # 추론 결과에서 필요한 데이터를 추출합니다.
        for key in raw_detection.keys():
            if 'yolov8_nms_postprocess' in key:
                detections = raw_detection[key][0]

        # 추론 결과를 후처리합니다.
        pred_dict, pred_list = post_process_track(detections, raw_width, raw_height, self.width, self.height, num_of_classes, min_score=min_score)

        # 추적을 위해 프레임과 탐지 결과를 준비합니다.
        frame = np.array(image)
        pred_results = np.array(pred_list)

        end_full = time.time()
        full_infer_time = end_full - start_full  # 전체 추론 시간을 계산합니다.

        return pred_results, frame, infer_time, full_infer_time  # 결과 반환


def object_counting(tracker, total_counter, frame, pred_results):
    """
    프레임 내 객체를 추적하고, 해당 객체들을 세는 함수입니다.

    Args:
        tracker (object): 객체를 추적하는 트래커 객체입니다.
        total_counter (list of counters): 각 객체에 대해 카운트를 수행하는 카운터 객체의 리스트입니다.
        frame (np.array): 현재 처리하고 있는 프레임의 이미지 데이터입니다.
        pred_results (list): 객체 탐지 결과를 포함하는 리스트, 각 객체의 위치와 클래스 정보가 포함되어 있습니다.

    Returns:
        np.array: 추적 결과가 추가된 이미지 프레임을 반환합니다.

    이 함수는 주어진 프레임에 대해 객체 탐지 결과를 사용하여 추적을 수행하고,
    추적된 각 객체에 대한 정보를 추출하여 카운터 객체를 통해 객체를 세는 작업을 진행합니다.
    추적 결과는 프레임에 어노테이션으로 추가되고, 각 카운터를 통해 세어진 결과로 프레임을 업데이트한 후 반환합니다.
    """

    # 트래커를 사용하여 탐지 결과에 따라 객체를 추적합니다.
    tracker_results = tracker.update(pred_results, frame)  # 각 객체에 대한 추적 정보를 얻습니다.

    # 추적 결과를 저장할 딕셔너리를 초기화합니다.
    track_dict = {
        'boxes': [],  # 객체의 바운딩 박스
        'class': [],  # 객체의 클래스
        'track_id': [],  # 추적 ID
    }

    # 추적된 각 객체에 대해 정보를 딕셔너리에 저장합니다.
    for track in tracker_results:
        track_dict["boxes"].append(track[:4])  # 바운딩 박스 좌표
        track_dict["class"].append(int(track[6]))  # 객체의 클래스 인덱스
        track_dict["track_id"].append(int(track[4]))  # 추적 ID

    # 각 카운터 객체를 사용하여 현재 프레임에 객체 카운팅을 수행합니다.
    for counter in total_counter:
        frame = counter.start_counting(frame, track_dict)  # 카운팅 결과를 프레임에 반영

    return frame  # 어노테이션과 카운팅 정보가 추가된 프레임을 반환합니다.


def inference(video_path, yaml_path=None, model_name=None, total_counter=None):
    """
    비디오 파일에서 객체 탐지 및 추적을 수행하고 결과를 비디오로 저장하는 함수입니다.

    Args:
        video_path (str): 분석할 비디오 파일의 경로입니다.
        yaml_path (str, optional): 모델 구성 및 클래스 정보가 포함된 YAML 파일의 경로입니다.
        model_name (str, optional): 사용할 모델의 이름입니다.
        total_counter (list, optional): 객체 카운팅을 위해 사용할 카운터 객체들의 리스트입니다.

    이 함수는 YAML 파일에서 모델 구성을 로드하고, 비디오에서 프레임을 읽어 해당 프레임에 대해 객체 탐지를 수행합니다.
    탐지된 객체는 추적되며, 추적 정보는 객체 카운터를 통해 관리됩니다.
    처리된 결과는 새로운 비디오 파일로 저장되며, 전체 추론 시간과 프레임별 추론 시간은 계산되어 출력됩니다.
    """

    insp_option = None  # 검사 옵션 초기화 (현재 사용되지 않음)

    infer_times = []  # 각 프레임 처리에 걸린 시간 저장
    full_infer_times = []  # 전체 프로세스 시간 저장

    # YAML 파일 로드
    with open(yaml_path, 'r', encoding='utf-8') as f:
        yaml_info = yaml.load(f, Loader=yaml.FullLoader)  # YAML 정보 로딩

    base_dir = os.path.dirname(yaml_path)  # YAML 파일의 디렉토리 경로
    dataset_name = os.path.basename(base_dir)  # 데이터셋 이름 추출

    print(video_path)  # 비디오 경로 출력
    caption = cv2.VideoCapture(video_path)  # 비디오 파일 열기

    output_directory = f'./results/{dataset_name}/{model_name}'  # 결과 파일 저장 디렉토리
    create_directory_if_not_exists(output_directory)  # 디렉토리가 없으면 생성

    if caption.isOpened():  # 비디오가 정상적으로 열렸는지 확인
        fps = caption.get(cv2.CAP_PROP_FPS)  # 프레임 속도
        frame_count = caption.get(cv2.CAP_PROP_FRAME_COUNT)  # 총 프레임 수
        width = caption.get(cv2.CAP_PROP_FRAME_WIDTH)  # 프레임 너비
        height = caption.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 프레임 높이
        print(f'frame per second: {fps}')
        print(f'resolution: {width} x {height}')

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 비디오 코덱 설정
    output = cv2.VideoWriter(os.path.join(output_directory, f'{model_name}_{dataset_name}.avi'), fourcc, fps, (int(width), int(height)))

    hef_path = os.path.abspath(f'./results/{dataset_name}/{model_name}/{model_name}.hef')  # HEF 파일 경로
    hailo_instance = HailoInferences(hef_path, DL_TYPE)  # Hailo 인스턴스 생성

    tracker = BoTSORT(
        model_weights=None,  # 모델 가중치
        device='cpu',  # 연산 장치
        fp16=True,  # FP16 사용 여부
        with_reid=False,  # ReID 사용 여부
        match_thresh=0.6,  # 매칭 임계값
    )

    while caption.isOpened():  # 비디오가 열린 상태인 동안 반복
        success, frame = caption.read()  # 프레임 읽기

        if not success:
            print("End of video or error.")  # 비디오 끝이나 에러 발생 시 메시지 출력
            break
    
        pred_results, frame, infer_time, full_infer_time = hailo_instance.object_detection(frame, yaml_info, insp_option)  # 객체 탐지

        if pred_results is None or pred_results.shape[0] == 0:
            print("검출된 객체가 없습니다.")  # 객체 미검출 시 메시지 출력
            continue

        print(f"pred_results: {pred_results}")  # 탐지 결과 출력

        frame = object_counting(tracker, total_counter, frame, pred_results)  # 객체 카운팅

        cv2.imshow("YOLOv8 Tracking", frame)  # 추적된 프레임 표시
        output.write(frame)  # 처리된 프레임을 비디오 파일에 기록

        infer_times.append(infer_time)  # 추론 시간 저장
        full_infer_times.append(full_infer_time)  # 전체 추론 시간 저장

        if cv2.waitKey(2) & 0xFF == ord("q"):  # 'q' 키가 눌리면 종료
            break

    caption.release()  # 비디오 해제
    output.release()  # 출력 비디오 해제
    cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기

    print('\nAverage Inference Time:', sum(infer_times) / len(infer_times))  # 평균 추론 시간 출력
    print('Average entire Inference Time:', sum(full_infer_times) / len(full_infer_times))  # 평균 전체


def main():
    yaml_path, dataset_name = get_yaml_file()
    print(yaml_path)

    if yaml_path == '':
        exit()

    gui = HailoGUI()
    video_path, model_name = gui.get_user_input()

    # yaml 파일을 로드
    with open(yaml_path, 'r', encoding='utf-8') as f:
        yaml_info = yaml.load(f, Loader=yaml.FullLoader)

    # Define region points
    region_points = [(50, 540), (325, 540), (325, 525), (50, 525)]
    region_points_2 = [(375, 540), (650, 540), (650, 525), (375, 525)]
    # region_points_3 = [(700, 540), (975, 540), (975, 525), (700, 525)]

    # Init Object Counter
    total_counter = []

    total_counter.append(counter_ob(region_points, class_names= yaml_info['names'], ordered=1, descript="Line 1"))
    total_counter.append(counter_ob(region_points_2, class_names= yaml_info['names'], ordered=2, descript="Line 2"))
    # total_counter.append(counter_ob(region_points_3, class_names= yaml_info['names'], ordered=3, descript="Line 3"))

    print("START!, object counting.")
    
    inference(video_path=video_path, yaml_path=yaml_path, model_name=model_name, total_counter=total_counter)  # model_name을 인자로 넘겨주려면 해당 함수에도 반영해야 합니다.


if __name__ == "__main__":
    print("Hello World")
    main()
