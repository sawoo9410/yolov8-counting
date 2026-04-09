from collections import defaultdict

import cv2
from shapely.geometry import LineString, Point, Polygon

from src.utils.annotator import Annotator, colors


class ObjectCounter:
    """단일 라인(또는 폴리곤 영역) 기반으로 객체를 카운팅하는 클래스."""

    def __init__(self):
        # 마우스 이벤트
        self.is_drawing = False
        self.selected_point = None

        # 카운팅 라인/영역 설정
        self.reg_pts = [(20, 400), (1260, 400)]
        self.line_dist_thresh = 15
        self.counting_region = None
        self.region_color = (255, 0, 255)
        self.region_thickness = 5

        # 이미지 및 시각화 설정
        self.im0 = None
        self.tf = None
        self.view_img = False
        self.view_in_counts = True
        self.view_out_counts = True
        self.names = None
        self.annotator = None
        self.window_name = "Ultralytics YOLOv8 Object Counter"

        # 카운팅 상태
        self.in_counts = 0
        self.out_counts = 0
        self.counting_dict = {}
        self.count_txt_thickness = 0
        self.count_txt_color = (0, 0, 0)
        self.count_color = (255, 255, 255)

        # 트랙 정보
        self.track_history = defaultdict(list)
        self.track_thickness = 2
        self.draw_tracks = False
        self.track_color = (0, 255, 0)
        self.ordered = None
        self.descript = None

    def set_args(
        self,
        classes_names,
        reg_pts,
        descript,
        count_reg_color=(255, 0, 255),
        line_thickness=2,
        track_thickness=2,
        view_img=False,
        view_in_counts=True,
        view_out_counts=True,
        draw_tracks=False,
        count_txt_thickness=2,
        count_txt_color=(0, 0, 0),
        count_color=(255, 255, 255),
        track_color=(0, 255, 0),
        region_thickness=5,
        line_dist_thresh=15,
        ordered=1,
    ):
        """
        카운터의 동작 파라미터를 설정합니다.

        Args:
            classes_names (dict): 클래스 이름 딕셔너리
            reg_pts (list): 카운팅 영역 정의 점 (2점=라인, 3점 이상=폴리곤)
            descript (str): 영역 설명 텍스트
            count_reg_color (tuple): 카운팅 영역 색상
            line_thickness (int): 바운딩 박스 선 두께
            track_thickness (int): 트랙 선 두께
            view_img (bool): 실시간 영상 표시 여부
            view_in_counts (bool): In 카운트 표시 여부
            view_out_counts (bool): Out 카운트 표시 여부
            draw_tracks (bool): 트랙 궤적 표시 여부
            count_txt_thickness (int): 카운트 텍스트 두께
            count_txt_color (tuple): 카운트 텍스트 색상
            count_color (tuple): 카운트 배경 색상
            track_color (tuple): 트랙 색상
            region_thickness (int): 카운팅 영역 선 두께
            line_dist_thresh (int): 라인 거리 임계값
            ordered (int): 텍스트 배치 순서
        """
        self.tf = line_thickness
        self.view_img = view_img
        self.view_in_counts = view_in_counts
        self.view_out_counts = view_out_counts
        self.track_thickness = track_thickness
        self.draw_tracks = draw_tracks

        # 점 개수에 따라 라인 또는 폴리곤 카운터로 설정
        if len(reg_pts) == 2:
            print("Line Counter Initiated.")
            self.reg_pts = reg_pts
            self.counting_region = LineString(self.reg_pts)
        elif len(reg_pts) >= 3:
            print("Region Counter Initiated.")
            self.reg_pts = reg_pts
            self.counting_region = Polygon(self.reg_pts)
        else:
            print("Invalid Region points provided, using Line Counter.")
            self.counting_region = LineString(self.reg_pts)

        self.names = classes_names
        self.track_color = track_color
        self.count_txt_thickness = count_txt_thickness
        self.count_txt_color = count_txt_color
        self.count_color = count_color
        self.region_color = count_reg_color
        self.region_thickness = region_thickness
        self.line_dist_thresh = line_dist_thresh
        self.ordered = ordered
        self.descript = descript

    def mouse_event_for_region(self, event, x, y, flags, params):
        """마우스 이벤트로 카운팅 영역을 이동합니다."""
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, point in enumerate(self.reg_pts):
                if (
                    isinstance(point, (tuple, list))
                    and len(point) >= 2
                    and (abs(x - point[0]) < 10 and abs(y - point[1]) < 10)
                ):
                    self.selected_point = i
                    self.is_drawing = True
                    break
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing and self.selected_point is not None:
                self.reg_pts[self.selected_point] = (x, y)
                self.counting_region = Polygon(self.reg_pts)
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            self.selected_point = None

    def extract_and_process_tracks(self, tracks):
        """추적 결과를 처리하여 바운딩 박스, 트랙, 카운팅을 수행합니다."""
        self.annotator = Annotator(self.im0, self.tf, self.names)

        if tracks['track_id'] is not None:
            boxes = tracks["boxes"]
            clss = tracks["class"]
            track_ids = tracks["track_id"]

            for box, track_id, cls in zip(boxes, track_ids, clss):
                # 바운딩 박스 및 라벨 그리기
                self.annotator.box_label(box, label=f"{track_id}:{self.names[cls]}", color=colors(int(cls), True))

                # 트랙 히스토리 업데이트 (최대 30프레임)
                track_line = self.track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                    track_line.pop(0)

                # 트랙 궤적 그리기
                if self.draw_tracks:
                    self.annotator.draw_centroid_and_tracks(
                        track_line, color=self.track_color, track_thickness=self.track_thickness
                    )

                prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None
                centroid = Point((box[:2] + box[2:]) / 2)

                # 폴리곤 영역 기반 카운팅
                if len(self.reg_pts) >= 3:
                    is_inside = self.counting_region.contains(centroid)
                    current_position = "in" if is_inside else "out"

                    if prev_position is not None:
                        if self.counting_dict[track_id] != current_position and is_inside:
                            self.in_counts += 1
                            self.counting_dict[track_id] = "in"
                        elif self.counting_dict[track_id] != current_position and not is_inside:
                            self.out_counts += 1
                            self.counting_dict[track_id] = "out"
                        else:
                            self.counting_dict[track_id] = current_position
                    else:
                        self.counting_dict[track_id] = current_position

                # 라인 기반 카운팅
                elif len(self.reg_pts) == 2:
                    if prev_position is not None:
                        is_inside = (box[0] - prev_position[0]) * (
                            self.counting_region.centroid.x - prev_position[0]
                        ) > 0
                        current_position = "in" if is_inside else "out"

                        if self.counting_dict[track_id] != current_position and is_inside:
                            self.in_counts += 1
                            self.counting_dict[track_id] = "in"
                        elif self.counting_dict[track_id] != current_position and not is_inside:
                            self.out_counts += 1
                            self.counting_dict[track_id] = "out"
                        else:
                            self.counting_dict[track_id] = current_position
                    else:
                        self.counting_dict[track_id] = None

        # 카운트 레이블 표시
        incount_label = f"In Count : {self.in_counts}"
        outcount_label = f"OutCount : {self.out_counts}"

        counts_label = None
        if not self.view_in_counts and not self.view_out_counts:
            counts_label = None
        elif not self.view_in_counts:
            counts_label = f"{self.descript} {outcount_label}"
        elif not self.view_out_counts:
            counts_label = f"{self.descript} {incount_label}"
        else:
            counts_label = f"{self.descript} {incount_label} {outcount_label}"

        if counts_label is not None:
            self.annotator.count_labels(
                counts=counts_label,
                count_txt_size=self.count_txt_thickness,
                txt_color=self.count_txt_color,
                color=self.count_color,
                ordered=self.ordered,
            )

    def start_counting(self, im0, tracks):
        """
        객체 카운팅의 메인 함수. 프레임에 카운팅 결과를 시각화합니다.

        Args:
            im0 (ndarray): 현재 프레임 이미지
            tracks (dict): 추적 결과 {'boxes', 'class', 'track_id'}

        Returns:
            ndarray: 카운팅 결과가 그려진 프레임
        """
        self.im0 = im0
        self.extract_and_process_tracks(tracks)
        self.annotator.draw_region(
            reg_pts=self.reg_pts, color=self.region_color,
            thickness=self.region_thickness, descript=self.descript,
        )
        return self.im0


class CrossObjectCounter:
    """두 라인 사이를 통과하는 객체를 카운팅하는 클래스."""

    def __init__(self):
        self.is_drawing = False
        self.selected_point = None

        # 두 개의 카운팅 라인
        self.reg_pts_1 = [(20, 400), (1260, 400)]
        self.reg_pts_2 = [(20, 600), (1260, 600)]
        self.line_dist_thresh = 15
        self.counting_region_1 = LineString(self.reg_pts_1)
        self.counting_region_2 = LineString(self.reg_pts_2)
        self.region_color = (255, 0, 255)
        self.region_thickness = 5

        # 이미지 및 시각화 설정
        self.im0 = None
        self.tf = None
        self.view_img = False
        self.view_in_counts = True
        self.view_out_counts = True
        self.names = None
        self.annotator = None
        self.window_name = "Ultralytics YOLOv8 Object Counter"

        # 카운팅 상태
        self.cross_counts = 0
        self.counting_dict = defaultdict(lambda: None)
        self.count_txt_thickness = 0
        self.count_txt_color = (0, 0, 0)
        self.count_color = (255, 255, 255)

        # 트랙 정보
        self.track_history = defaultdict(list)
        self.track_thickness = 2
        self.draw_tracks = False
        self.track_color = (0, 255, 0)
        self.ordered = None
        self.descript = None

    def set_args(
        self,
        classes_names,
        reg_pts_1,
        reg_pts_2,
        descript,
        count_reg_color=(255, 0, 255),
        line_thickness=2,
        track_thickness=2,
        view_img=False,
        view_in_counts=True,
        view_out_counts=True,
        draw_tracks=False,
        count_txt_thickness=2,
        count_txt_color=(0, 0, 0),
        count_color=(255, 255, 255),
        track_color=(0, 255, 0),
        region_thickness=5,
        line_dist_thresh=15,
        ordered=1,
    ):
        """
        Cross 카운터의 동작 파라미터를 설정합니다.

        Args:
            classes_names (dict): 클래스 이름 딕셔너리
            reg_pts_1 (list): 첫 번째 카운팅 라인 정의 점 (2점)
            reg_pts_2 (list): 두 번째 카운팅 라인 정의 점 (2점)
            descript (str): 영역 설명 텍스트 (예: "Line1->Line2")
            count_reg_color (tuple): 카운팅 영역 색상
            line_thickness (int): 바운딩 박스 선 두께
            track_thickness (int): 트랙 선 두께
            view_img (bool): 실시간 영상 표시 여부
            view_in_counts (bool): In 카운트 표시 여부
            view_out_counts (bool): Out 카운트 표시 여부
            draw_tracks (bool): 트랙 궤적 표시 여부
            count_txt_thickness (int): 카운트 텍스트 두께
            count_txt_color (tuple): 카운트 텍스트 색상
            count_color (tuple): 카운트 배경 색상
            track_color (tuple): 트랙 색상
            region_thickness (int): 카운팅 영역 선 두께
            line_dist_thresh (int): 라인 거리 임계값 (객체가 라인에 근접했다고 판단하는 거리)
            ordered (int): 텍스트 배치 순서
        """
        self.tf = line_thickness
        self.view_img = view_img
        self.view_in_counts = view_in_counts
        self.view_out_counts = view_out_counts
        self.track_thickness = track_thickness
        self.draw_tracks = draw_tracks

        if len(reg_pts_1) == 2:
            self.reg_pts_1 = reg_pts_1
            self.counting_region_1 = LineString(self.reg_pts_1)
        else:
            raise ValueError("reg_pts_1은 반드시 2개의 점으로 구성되어야 합니다.")

        if len(reg_pts_2) == 2:
            self.reg_pts_2 = reg_pts_2
            self.counting_region_2 = LineString(self.reg_pts_2)
        else:
            raise ValueError("reg_pts_2는 반드시 2개의 점으로 구성되어야 합니다.")

        self.names = classes_names
        self.track_color = track_color
        self.count_txt_thickness = count_txt_thickness
        self.count_txt_color = count_txt_color
        self.count_color = count_color
        self.region_color = count_reg_color
        self.region_thickness = region_thickness
        self.line_dist_thresh = line_dist_thresh
        self.ordered = ordered
        self.descript = descript

    def extract_and_process_tracks(self, tracks):
        """
        추적 결과를 처리하여 두 라인 사이를 통과하는 객체를 카운팅합니다.

        카운팅 로직:
        - 객체가 첫 번째 라인 근처 → 두 번째 라인 근처로 이동하면 +1
        - 객체가 두 번째 라인 근처 → 첫 번째 라인 근처로 이동하면 -1
        """
        self.annotator = Annotator(self.im0, self.tf, self.names)

        if tracks['track_id'] is not None:
            boxes = tracks["boxes"]
            clss = tracks["class"]
            track_ids = tracks["track_id"]

            for box, track_id, cls in zip(boxes, track_ids, clss):
                # 바운딩 박스 및 라벨 그리기
                self.annotator.box_label(box, label=f"{track_id}:{self.names[cls]}", color=colors(int(cls), True))

                # 트랙 히스토리 업데이트 (최대 30프레임)
                track_line = self.track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                    track_line.pop(0)

                # 트랙 궤적 그리기
                if self.draw_tracks:
                    self.annotator.draw_centroid_and_tracks(
                        track_line, color=self.track_color, track_thickness=self.track_thickness
                    )

                prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None
                centroid = Point((box[:2] + box[2:]) / 2)

                if prev_position is not None:
                    # 각 라인과의 거리로 근접 여부 판단
                    near_line_1 = self.counting_region_1.distance(centroid) < self.line_dist_thresh
                    near_line_2 = self.counting_region_2.distance(centroid) < self.line_dist_thresh

                    if self.counting_dict[track_id] is None:
                        self.counting_dict[track_id] = (near_line_1, near_line_2)

                    prev_near_1, prev_near_2 = self.counting_dict[track_id]

                    # 라인1 → 라인2 이동 감지: 카운트 증가
                    if prev_near_1 and not near_line_1 and near_line_2:
                        self.cross_counts += 1
                    # 라인2 → 라인1 이동 감지: 카운트 감소
                    elif prev_near_2 and not near_line_2 and near_line_1:
                        self.cross_counts -= 1

                    self.counting_dict[track_id] = (near_line_1, near_line_2)

        # 카운트 레이블 표시
        counts_label = f"{self.descript} Cross Count: {self.cross_counts}"
        self.annotator.count_labels(
            counts=counts_label,
            count_txt_size=self.count_txt_thickness,
            txt_color=self.count_txt_color,
            color=self.count_color,
            ordered=self.ordered,
        )

    def start_counting(self, im0, tracks):
        """
        객체 카운팅의 메인 함수. 프레임에 두 라인과 카운팅 결과를 시각화합니다.

        Args:
            im0 (ndarray): 현재 프레임 이미지
            tracks (dict): 추적 결과 {'boxes', 'class', 'track_id'}

        Returns:
            ndarray: 카운팅 결과가 그려진 프레임
        """
        self.im0 = im0
        self.extract_and_process_tracks(tracks)

        # 두 라인을 각각 그리기 (descript에서 "Line1->Line2" 형태를 분리)
        first_line_name, second_line_name = self.descript.split('->')
        self.annotator.draw_region(
            reg_pts=self.reg_pts_1, color=self.region_color,
            thickness=self.region_thickness, descript=first_line_name,
        )
        self.annotator.draw_region(
            reg_pts=self.reg_pts_2, color=self.region_color,
            thickness=self.region_thickness, descript=second_line_name,
        )
        return self.im0
