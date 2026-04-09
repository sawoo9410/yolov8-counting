# YOLOv8 Conveyor Belt Object Counting

YOLOv8 기반의 컨베이어 벨트 객체 카운팅 시스템입니다.  
컨베이어 벨트 위를 지나가는 객체를 **탐지 + 추적 + 카운팅**하여 실시간으로 수량을 집계합니다.

[![Watch the video](https://img.youtube.com/vi/YzCje8Fl-xg/0.jpg)](https://www.youtube.com/watch?v=YzCje8Fl-xg)

---

## 주요 기능

| 모드 | 설명 |
|------|------|
| **Train** | 커스텀 데이터셋으로 YOLOv8 파인튜닝 후 `.pt` / `.onnx` 모델 저장 |
| **Validation** | 학습된 모델의 추론 결과 시각화 및 mAP50-95 성능 평가 |
| **Video** | BoTSORT 트래커 기반 객체 추적 + 다중 카운팅 라인 지원 |

### 카운팅 방식

- **Single Line** (4좌표) : 단일 라인을 기준으로 In / Out 카운팅
- **Cross Line** (8좌표) : 두 라인 사이를 통과하는 객체를 교차 카운팅

---

## 프로젝트 구조

```
yolov8-counting/
├── main.py                        # CLI 엔트리포인트
├── src/
│   ├── download.py                # YOLOv8 사전학습 가중치 다운로드
│   ├── train.py                   # 모델 학습
│   ├── validation.py              # 모델 검증
│   ├── video_inference.py         # 영상 객체 카운팅
│   └── utils/
│       ├── annotator.py           # 바운딩박스 / 트랙 / 카운트 시각화
│       └── object_counter.py      # 라인 기반 객체 카운터
├── models/                        # YOLOv8 사전학습 가중치 (.pt)
├── datasets/                      # 학습 데이터셋
│   └── <dataset_name>/
│       ├── <dataset_name>.yaml    # 데이터셋 설정
│       ├── classes.txt            # 클래스 목록
│       ├── images/train/          # 학습 이미지
│       ├── labels/train/          # YOLO 포맷 라벨
│       └── *.mp4                  # 카운팅용 영상
├── results/                       # 학습/추론 결과 저장
│   └── <dataset_name>/<model_name>/
│       ├── best.pt                # 파인튜닝된 모델
│       ├── best.onnx              # ONNX 변환 모델
│       ├── metrics.txt            # 성능 평가 결과
│       ├── output_images/         # 검증 결과 이미지
│       └── video_counting/        # 카운팅 결과 영상
├── requirements.txt               # Python 패키지 의존성
└── env_install.bat                # Windows 환경 설치 스크립트
```

---

## 설치

### 자동 설치 (Windows)

```bash
env_install.bat
```

### 수동 설치

```bash
conda create -n yolov8_tracking python=3.9
conda activate yolov8_tracking

pip install -r requirements.txt
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

python ./src/download.py
```

---

## 사용법

### 1. 모델 학습

```bash
python main.py --mode train \
    --dataset_name conveyor_belt \
    --model yolov8n \
    --epochs 30 \
    --batch_size 16 \
    --img_size 640
```

### 2. 모델 검증

```bash
python main.py --mode validation \
    --dataset_name conveyor_belt \
    --model yolov8n
```

### 3. 영상 객체 카운팅

```bash
python main.py --mode video \
    --dataset_name conveyor_belt \
    --model yolov8n \
    --video_filename example_conveyor_20s.mp4 \
    --line_points_list 0 550 325 550 0 600 325 600 \
    --line_points_list 375 550 600 550 375 600 600 600 \
    --line_points_list 650 550 900 550 650 600 900 600
```

`--line_points_list`는 여러 번 지정하여 다중 카운팅 라인을 설정할 수 있습니다.

- **4개 좌표** (`x1 y1 x2 y2`) : Single Line 모드
- **8개 좌표** (`x1 y1 x2 y2 x3 y3 x4 y4`) : Cross Line 모드 (두 선 사이 통과 감지)

---

## CLI 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--mode` | (필수) | 실행 모드: `train`, `validation`, `video` |
| `--dataset_name` | (필수) | 데이터셋 이름 (`datasets/` 하위 폴더명) |
| `--model` | `yolov8n` | 모델 선택: `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l` |
| `--epochs` | `30` | 학습 에폭 수 |
| `--batch_size` | `16` | 학습 배치 크기 |
| `--img_size` | `640` | 입력 이미지 크기 |
| `--video_filename` | - | 카운팅할 영상 파일명 (video 모드 필수) |
| `--line_points_list` | - | 카운팅 라인 좌표 (여러 개 지정 가능) |

---

## 데이터셋 구성

커스텀 데이터셋을 추가하려면 `datasets/` 아래에 다음 구조로 폴더를 만드세요.

```
datasets/<dataset_name>/
├── <dataset_name>.yaml    # YOLO 데이터셋 설정 파일
├── classes.txt            # 클래스 목록 (줄바꿈 구분)
├── images/train/          # 학습 이미지 (.jpg)
├── labels/train/          # YOLO 포맷 라벨 (.txt)
└── *.mp4                  # 카운팅용 영상 (video 모드)
```

YAML 파일 예시:
```yaml
train: images/train
val: images/train
nc: 1
names:
  - bacchus
```

---

## 기술 스택

- **객체 탐지**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **객체 추적**: [BoTSORT (BoxMOT)](https://github.com/mikel-brostrom/boxmot)
- **카운팅 로직**: Shapely 기반 라인 교차 판정
- **환경**: Python 3.9 / PyTorch 2.2 / CUDA 11.8
