import math
import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont


def is_ascii(s) -> bool:
    """문자열이 ASCII 문자로만 구성되어 있는지 확인합니다."""
    s = str(s)
    return all(ord(c) < 128 for c in s)


class Colors:
    """Ultralytics 기본 색상 팔레트. hex 코드를 RGB로 변환합니다."""

    def __init__(self):
        hexs = (
            "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231",
            "48F90A", "92CC17", "3DDB86", "1A9334", "00D4BB",
            "2C99A8", "00C2FF", "344593", "6473FF", "0018EC",
            "8438FF", "520085", "CB38FF", "FF95C8", "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array(
            [
                [255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0],
                [255, 153, 255], [153, 204, 255], [255, 102, 255], [255, 51, 255],
                [102, 178, 255], [51, 153, 255], [255, 153, 153], [255, 102, 102],
                [255, 51, 51], [153, 255, 153], [102, 255, 102], [51, 255, 51],
                [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255],
            ],
            dtype=np.uint8,
        )

    def __call__(self, i, bgr=False):
        """인덱스에 해당하는 RGB(또는 BGR) 색상을 반환합니다."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """hex 색상 코드를 RGB 튜플로 변환합니다."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()


class Annotator:
    """
    이미지에 바운딩 박스, 트랙, 카운트 등을 시각화하는 클래스.

    PIL과 OpenCV 두 가지 백엔드를 지원합니다.
    """

    def __init__(self, im, line_width=None, font_size=None, font="Arial.ttf", pil=False, example="abc"):
        input_is_pil = isinstance(im, Image.Image)
        self.pil = pil or input_is_pil
        self.lw = line_width or max(round(sum(im.size if input_is_pil else im.shape) / 2 * 0.003), 2)

        if self.pil:
            self.im = im if input_is_pil else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            size = font_size or max(round(sum(self.im.size) / 2 * 0.035), 12)
            try:
                self.font = ImageFont.truetype(font, size)
            except IOError:
                self.font = ImageFont.load_default()
            self.font.getsize = lambda x: self.font.getbbox(x)[2:4]
        else:
            assert im.data.contiguous, "Image not contiguous. Apply np.ascontiguousarray(im) to input images."
            self.im = im if im.flags.writeable else im.copy()
            self.tf = max(self.lw - 1, 1)  # 폰트 두께
            self.sf = self.lw / 3  # 폰트 스케일

        # Pose 시각화용 스켈레톤 구조
        self.skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7],
        ]
        self.limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        self.kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

    def box_label(self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255), rotated=False):
        """바운딩 박스와 라벨을 이미지에 그립니다."""
        if isinstance(box, torch.Tensor):
            box = box.tolist()

        if self.pil or not is_ascii(label):
            if rotated:
                p1 = box[0]
                self.draw.polygon([tuple(b) for b in box], width=self.lw, outline=color)
            else:
                p1 = (box[0], box[1])
                self.draw.rectangle(box, width=self.lw, outline=color)
            if label:
                w, h = self.font.getsize(label)
                outside = p1[1] - h >= 0
                self.draw.rectangle(
                    (p1[0], p1[1] - h if outside else p1[1], p1[0] + w + 1, p1[1] + 1 if outside else p1[1] + h + 1),
                    fill=color,
                )
                self.draw.text((p1[0], p1[1] - h if outside else p1[1]), label, fill=txt_color, font=self.font)
        else:
            if rotated:
                p1 = [int(b) for b in box[0]]
                cv2.polylines(self.im, [np.asarray(box, dtype=int)], True, color, self.lw)
            else:
                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                w, h = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)
                cv2.putText(
                    self.im, label,
                    (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0, self.sf, txt_color, thickness=self.tf, lineType=cv2.LINE_AA,
                )

    def draw_region(self, reg_pts=None, color=(0, 255, 0), thickness=5, descript=''):
        """
        카운팅 영역 라인을 그리고 설명 텍스트를 표시합니다.

        Args:
            reg_pts (list): 영역 정의 점 (2점=라인, 4점 이상=폴리곤)
            color (tuple): 영역 색상
            thickness (int): 선 두께
            descript (str): 영역 설명 텍스트
        """
        if reg_pts:
            cv2.polylines(self.im, [np.array(reg_pts, dtype=np.int32)], isClosed=True, color=color, thickness=thickness)

        if descript and reg_pts:
            if len(reg_pts) >= 4:
                text_pos = (reg_pts[0][0] + 10, reg_pts[2][1] - 10)
            else:
                text_pos = (reg_pts[0][0] + 10, reg_pts[0][1] - 10)
            cv2.putText(self.im, descript, text_pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.75, color=color, thickness=2)

    def draw_centroid_and_tracks(self, track, color=(255, 0, 255), track_thickness=2):
        """
        트랙 궤적과 중심점을 그립니다.

        Args:
            track (list): 트랙 궤적 좌표 리스트
            color (tuple): 궤적 색상
            track_thickness (int): 궤적 선 두께
        """
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(self.im, [points], isClosed=False, color=color, thickness=track_thickness)
        cv2.circle(self.im, (int(track[-1][0]), int(track[-1][1])), track_thickness * 2, color, -1)

    def count_labels(self, counts=0, count_txt_size=2, color=(255, 255, 255), txt_color=(0, 0, 0), margin=10, padding=20, ordered=1):
        """
        카운트 정보를 우측 상단에 표시합니다.
        ordered 값에 따라 여러 카운터의 텍스트가 겹치지 않게 배치됩니다.

        Args:
            counts: 표시할 카운트 텍스트
            count_txt_size (int): 텍스트 크기
            color (tuple): 배경색
            txt_color (tuple): 텍스트 색상
            margin (int): 이미지 가장자리 여백
            padding (int): 텍스트 간 간격
            ordered (int): 텍스트 배치 순서 (1부터 시작)
        """
        self.tf = count_txt_size
        tl = self.tf or round(0.002 * (self.im.shape[0] + self.im.shape[1]) / 2) + 1
        tf = max(tl - 1, 1)

        t_size = cv2.getTextSize(str(counts), 0, fontScale=tl / 2, thickness=tf)[0]
        text_width, text_height = t_size

        # 우측 상단 기준 좌표 계산
        text_x = self.im.shape[1] - text_width - margin
        base_y = margin + text_height
        text_y = base_y + (text_height + padding) * (ordered - 1)

        # 텍스트 배경
        cv2.rectangle(
            self.im,
            (text_x - 5, text_y - text_height - 5),
            (text_x + text_width + 5, text_y + 5),
            color, -1,
        )
        # 텍스트
        cv2.putText(
            self.im, str(counts),
            (text_x, text_y),
            0, tl / 2, txt_color, tf, lineType=cv2.LINE_AA,
        )

    def fromarray(self, im):
        """numpy 배열로부터 PIL 이미지를 업데이트합니다."""
        self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.im)

    def result(self):
        """어노테이션된 이미지를 numpy 배열로 반환합니다."""
        return np.asarray(self.im)
