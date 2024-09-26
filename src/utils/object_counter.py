# # Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict

import cv2

# from ultralytics.utils.checks import check_imshow, check_requirements
# from ultralytics.utils.plotting import Annotator, colors

from src.utils.annotator import Annotator, colors
# check_requirements("shapely>=2.0.0")

from shapely.geometry import LineString, Point, Polygon


class ObjectCounter:
    """A class to manage the counting of objects in a real-time video stream based on their tracks."""

    def __init__(self):
        """Initializes the Counter with default values for various tracking and counting parameters."""

        # Mouse events
        self.is_drawing = False
        self.selected_point = None

        # Region & Line Information
        self.reg_pts = [(20, 400), (1260, 400)]
        self.line_dist_thresh = 15
        self.counting_region = None
        self.region_color = (255, 0, 255)
        self.region_thickness = 5

        # Image and annotation Information
        self.im0 = None
        self.tf = None
        self.view_img = False
        self.view_in_counts = True
        self.view_out_counts = True

        self.names = None  # Classes names
        self.annotator = None  # Annotator
        self.window_name = "Ultralytics YOLOv8 Object Counter"

        # Object counting Information
        self.in_counts = 0
        self.out_counts = 0
        self.counting_dict = {}
        self.count_txt_thickness = 0
        self.count_txt_color = (0, 0, 0)
        self.count_color = (255, 255, 255)

        # Tracks info
        self.track_history = defaultdict(list)
        self.track_thickness = 2
        self.draw_tracks = False
        self.track_color = (0, 255, 0)
        self.ordered = None
        self.descript = None
        # Check if environment support imshow
        # self.env_check = check_imshow(warn=True)

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
        Configures the Counter's image, bounding box line thickness, and counting region points.

        Args:
            line_thickness (int): Line thickness for bounding boxes.
            view_img (bool): Flag to control whether to display the video stream.
            view_in_counts (bool): Flag to control whether to display the incounts on video stream.
            view_out_counts (bool): Flag to control whether to display the outcounts on video stream.
            reg_pts (list): Initial list of points defining the counting region.
            classes_names (dict): Classes names
            track_thickness (int): Track thickness
            draw_tracks (Bool): draw tracks
            count_txt_thickness (int): Text thickness for object counting display
            count_txt_color (RGB color): count text color value
            count_color (RGB color): count text background color value
            count_reg_color (RGB color): Color of object counting region
            track_color (RGB color): color for tracks
            region_thickness (int): Object counting Region thickness
            line_dist_thresh (int): Euclidean Distance threshold for line counter
        """
        self.tf = line_thickness
        self.view_img = view_img
        self.view_in_counts = view_in_counts
        self.view_out_counts = view_out_counts
        self.track_thickness = track_thickness
        self.draw_tracks = draw_tracks

        # Region and line selection
        if len(reg_pts) == 2:
            print("Line Counter Initiated.")
            self.reg_pts = reg_pts
            self.counting_region = LineString(self.reg_pts)
        elif len(reg_pts) >= 3:
            print("Region Counter Initiated.")
            self.reg_pts = reg_pts
            self.counting_region = Polygon(self.reg_pts)
        else:
            print("Invalid Region points provided, region_points must be 2 for lines or >= 3 for polygons.")
            print("Using Line Counter Now")
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
        self.descript=descript

    def mouse_event_for_region(self, event, x, y, flags, params):
        """
        This function is designed to move region with mouse events in a real-time video stream.

        Args:
            event (int): The type of mouse event (e.g., cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, etc.).
            x (int): The x-coordinate of the mouse pointer.
            y (int): The y-coordinate of the mouse pointer.
            flags (int): Any flags associated with the event (e.g., cv2.EVENT_FLAG_CTRLKEY,
                cv2.EVENT_FLAG_SHIFTKEY, etc.).
            params (dict): Additional parameters you may want to pass to the function.
        """
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
        """Extracts and processes tracks for object counting in a video stream."""

        # Annotator Init and region drawing
        self.annotator = Annotator(self.im0, self.tf, self.names)

        if tracks['track_id'] is not None:
            # boxes = tracks[0].boxes.xyxy.cpu()
            # clss = tracks[0].boxes.cls.cpu().tolist()
            # track_ids = tracks[0].boxes.id.int().cpu().tolist()

            #         x1, y1, x2, y2 = track.to_tlbr()
            #         class_name = track.others
            #         track_id = f"{track.track_id} {class_name}" # 여기에 실제 라벨을 넣으면 되겠네.

            boxes = tracks["boxes"]
            clss = tracks["class"]
            track_ids =  tracks["track_id"]

            for box, track_id, cls in zip(boxes, track_ids, clss):

                # Draw bounding box
                self.annotator.box_label(box, label=f"{track_id}:{self.names[cls]}", color=colors(int(cls), True))

                # Draw Tracks
                track_line = self.track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                    track_line.pop(0)

                # Draw track trails
                if self.draw_tracks:
                    self.annotator.draw_centroid_and_tracks(
                        track_line, color=self.track_color, track_thickness=self.track_thickness
                    )

                prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None
                centroid = Point((box[:2] + box[2:]) / 2)

                # Count objects
                if len(self.reg_pts) >= 3:  # any polygon
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

        incount_label = f"In Count : {self.in_counts}"
        outcount_label = f"OutCount : {self.out_counts}"

        # Display counts based on user choice
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

    def display_frames(self):
        """Display frame."""
        # if self.env_check:
        self.annotator.draw_region(reg_pts=self.reg_pts, color=self.region_color, thickness=self.region_thickness, descript=self.descript)
        cv2.namedWindow(self.window_name)
        if len(self.reg_pts) == 4:  # only add mouse event If user drawn region
            cv2.setMouseCallback(self.window_name, self.mouse_event_for_region, {"region_points": self.reg_pts})
        cv2.imshow(self.window_name, self.im0)
        # Break Window
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return

    def start_counting(self, im0, tracks):
        """
        Main function to start the object counting process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0  # store image
        self.extract_and_process_tracks(tracks)  # draw region even if no objects

        self.annotator.draw_region(reg_pts=self.reg_pts, color=self.region_color, thickness=self.region_thickness, descript=self.descript)

        # if self.view_img:
        #     self.display_frames()
        return self.im0


class CrossObjectCounter:
    """두 선을 교차하는 객체를 실시간 비디오 스트림에서 카운팅하는 클래스를 관리합니다."""

    def __init__(self):
        """다양한 추적 및 카운팅 매개변수에 대한 기본값으로 카운터를 초기화합니다."""
        self.is_drawing = False  # 영역을 그리기 위한 플래그
        self.selected_point = None  # 선택된 포인트

        # 카운팅 영역 및 선 정보
        self.reg_pts_1 = [(20, 400), (1260, 400)]  # 첫 번째 선의 좌표
        self.reg_pts_2 = [(20, 600), (1260, 600)]  # 두 번째 선의 좌표
        self.line_dist_thresh = 15  # 선과의 거리 임계값
        self.counting_region_1 = LineString(self.reg_pts_1)  # 첫 번째 선의 LineString 객체
        self.counting_region_2 = LineString(self.reg_pts_2)  # 두 번째 선의 LineString 객체
        self.region_color = (255, 0, 255)  # 영역의 색상 (RGB)
        self.region_thickness = 5  # 영역의 두께

        # 이미지 및 주석 정보
        self.im0 = None  # 현재 프레임
        self.tf = None  # 텍스트 두께
        self.view_img = False  # 이미지를 표시할지 여부
        self.view_in_counts = True  # 들어오는 객체 수를 표시할지 여부
        self.view_out_counts = True  # 나가는 객체 수를 표시할지 여부

        self.names = None  # 클래스 이름
        self.annotator = None  # 주석을 달기 위한 객체
        self.window_name = "Ultralytics YOLOv8 Object Counter"  # 윈도우 이름

        # 객체 카운팅 정보
        self.cross_counts = 0  # 교차 카운트 초기화
        self.counting_dict = defaultdict(lambda: None)  # 객체의 상태를 저장할 딕셔너리
        self.count_txt_thickness = 0  # 텍스트 두께
        self.count_txt_color = (0, 0, 0)  # 텍스트 색상 (RGB)
        self.count_color = (255, 255, 255)  # 배경 색상 (RGB)

        # 추적 정보
        self.track_history = defaultdict(list)  # 객체의 추적 이력
        self.track_thickness = 2  # 추적 선의 두께
        self.draw_tracks = False  # 추적 선을 그릴지 여부
        self.track_color = (0, 255, 0)  # 추적 선의 색상 (RGB)
        self.ordered = None  # 순서 정보
        self.descript = None  # 설명 텍스트

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
        카운터의 이미지, 바운딩 박스 선 두께 및 카운팅 영역 점을 구성합니다.

        Args:
            line_thickness (int): 바운딩 박스 선의 두께.
            view_img (bool): 비디오 스트림을 표시할지 여부.
            view_in_counts (bool): 비디오 스트림에 들어오는 객체 수를 표시할지 여부.
            view_out_counts (bool): 비디오 스트림에 나가는 객체 수를 표시할지 여부.
            reg_pts_1 (list): 첫 번째 카운팅 영역을 정의하는 점의 초기 목록.
            reg_pts_2 (list): 두 번째 카운팅 영역을 정의하는 점의 초기 목록.
            classes_names (dict): 클래스 이름
            track_thickness (int): 추적 선의 두께
            draw_tracks (bool): 추적 선을 그릴지 여부
            count_txt_thickness (int): 객체 카운팅 표시를 위한 텍스트 두께
            count_txt_color (RGB color): 카운트 텍스트 색상 값
            count_color (RGB color): 카운트 텍스트 배경 색상 값
            count_reg_color (RGB color): 객체 카운팅 영역의 색상
            track_color (RGB color): 추적 선의 색상
            region_thickness (int): 객체 카운팅 영역의 두께
            line_dist_thresh (int): 선 카운터를 위한 유클리드 거리 임계값
            ordered (int): 텍스트 배치 순서
            descript (str): 영역 또는 기능에 대한 설명
        """
        self.tf = line_thickness  # 바운딩 박스 선의 두께 설정
        self.view_img = view_img  # 비디오 스트림 표시 여부 설정
        self.view_in_counts = view_in_counts  # 들어오는 객체 수 표시 여부 설정
        self.view_out_counts = view_out_counts  # 나가는 객체 수 표시 여부 설정
        self.track_thickness = track_thickness  # 추적 선의 두께 설정
        self.draw_tracks = draw_tracks  # 추적 선 그릴지 여부 설정

        if len(reg_pts_1) == 2:
            self.reg_pts_1 = reg_pts_1
            self.counting_region_1 = LineString(self.reg_pts_1)
        else:
            raise ValueError("reg_pts_1은 반드시 2개의 점으로 구성된 선이어야 합니다.")

        if len(reg_pts_2) == 2:
            self.reg_pts_2 = reg_pts_2
            self.counting_region_2 = LineString(self.reg_pts_2)
        else:
            raise ValueError("reg_pts_2는 반드시 2개의 점으로 구성된 선이어야 합니다.")

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
        """비디오 스트림에서 객체 카운팅을 위해 트랙을 추출하고 처리합니다."""
        # Annotator 객체를 초기화합니다.
        self.annotator = Annotator(self.im0, self.tf, self.names)

        # 트랙 정보가 존재하는지 확인합니다.
        if tracks['track_id'] is not None:
            # 바운딩 박스, 클래스, 트랙 ID를 추출합니다.
            boxes = tracks["boxes"]
            clss = tracks["class"]
            track_ids = tracks["track_id"]

            # 각 트랙에 대해 처리합니다.
            for box, track_id, cls in zip(boxes, track_ids, clss):
                # 바운딩 박스를 그리고 레이블을 추가합니다.
                self.annotator.box_label(box, label=f"{track_id}:{self.names[cls]}", color=colors(int(cls), True))

                # 트랙 히스토리에 현재 중심 좌표를 추가합니다.
                track_line = self.track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                    track_line.pop(0)  # 트랙 히스토리가 30을 초과하면 첫 번째 요소를 제거합니다.

                # 트랙을 그리는 옵션이 활성화된 경우 트랙을 그립니다.
                if self.draw_tracks:
                    self.annotator.draw_centroid_and_tracks(
                        track_line, color=self.track_color, track_thickness=self.track_thickness
                    )

                # 이전 위치를 가져옵니다. 없으면 None을 반환합니다.
                prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None
                # 현재 바운딩 박스의 중심 좌표를 계산합니다.
                centroid = Point((box[:2] + box[2:]) / 2)

                # 이전 위치가 존재하는 경우
                if prev_position is not None:
                    # 첫 번째 카운팅 선과의 거리가 임계값 이하인지 확인합니다.
                    current_position_1 = self.counting_region_1.distance(centroid) < self.line_dist_thresh
                    # 두 번째 카운팅 선과의 거리가 임계값 이하인지 확인합니다.
                    current_position_2 = self.counting_region_2.distance(centroid) < self.line_dist_thresh

                    # 트랙 ID에 대한 현재 상태가 저장되지 않은 경우 초기화합니다.
                    if self.counting_dict[track_id] is None:
                        self.counting_dict[track_id] = (current_position_1, current_position_2)

                    # 이전 위치를 가져옵니다.
                    prev_pos_1, prev_pos_2 = self.counting_dict[track_id]

                    # 객체가 첫 번째 선에서 두 번째 선으로 이동했는지 확인합니다.
                    if prev_pos_1 and not current_position_1 and current_position_2:
                        self.cross_counts += 1  # 카운트 증가
                    # 객체가 두 번째 선에서 첫 번째 선으로 이동했는지 확인합니다.
                    elif prev_pos_2 and not current_position_2 and current_position_1:
                        self.cross_counts -= 1  # 카운트 감소

                    # 현재 상태를 업데이트합니다.
                    self.counting_dict[track_id] = (current_position_1, current_position_2)

        # 카운트 레이블을 만듭니다.
        crosscount_label = f"Cross Count: {self.cross_counts}"
        counts_label = f"{self.descript} {crosscount_label}"

        # 카운트 레이블이 존재하는 경우 화면에 표시합니다.
        if counts_label is not None:
            self.annotator.count_labels(
                counts=counts_label,
                count_txt_size=self.count_txt_thickness,
                txt_color=self.count_txt_color,
                color=self.count_color,
                ordered=self.ordered,
            )

        # 디버깅을 위해 현재 카운트와 레이블을 출력합니다.
        print(counts_label)  # 실시간 카운트 확인을 위한 출력

    def display_frames(self):
        """프레임을 표시합니다."""
        self.annotator.draw_region(reg_pts=self.reg_pts_1, color=self.region_color, thickness=self.region_thickness, descript=self.descript)
        self.annotator.draw_region(reg_pts=self.reg_pts_2, color=self.region_color, thickness=self.region_thickness, descript=self.descript)
        cv2.namedWindow(self.window_name)
        cv2.imshow(self.window_name, self.im0)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return

    def start_counting(self, im0, tracks):
        """
        객체 카운팅 프로세스를 시작하는 메인 함수.

        Args:
            im0 (ndarray): 비디오 스트림에서 현재 프레임.
            tracks (list): 객체 추적 프로세스로부터 얻은 트랙 목록.
        """
        self.im0 = im0
        self.extract_and_process_tracks(tracks)
        first_line_name, second_line_name = self.descript.split('->')
        self.annotator.draw_region(reg_pts=self.reg_pts_1, color=self.region_color, thickness=self.region_thickness, descript=first_line_name)
        self.annotator.draw_region(reg_pts=self.reg_pts_2, color=self.region_color, thickness=self.region_thickness, descript=second_line_name)
        return self.im0
    
# if __name__ == "__main__":
#     ObjectCounter()