import argparse
from src.train import pretrain_yolov8
from src.validation import validation_yolov8
from src.video_inference import video_object_counting


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Training, Validation, and Video Object Counting Script")

    # 공통 인자
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'validation', 'video'],
                        help="실행 모드: 'train' (학습), 'validation' (검증), 'video' (영상 카운팅)")
    parser.add_argument('--dataset_name', type=str, required=True,
                        help="데이터셋 이름 (datasets/ 하위 폴더명)")
    parser.add_argument('--model', type=str, default='yolov8n',
                        choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l'],
                        help="YOLOv8 모델 선택 (default: yolov8n)")
    parser.add_argument('--epochs', type=int, default=30,
                        help="학습 에폭 수 (default: 30)")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="학습 배치 크기 (default: 16)")
    parser.add_argument('--img_size', type=int, default=640,
                        help="입력 이미지 크기 (default: 640)")

    # Video 모드 전용 인자
    parser.add_argument('--video_filename', type=str,
                        help="카운팅할 영상 파일명 (datasets/<dataset_name>/ 내 위치)")
    parser.add_argument('--line_points_list', type=int, nargs='+', action='append',
                        help="카운팅 라인 좌표. 4개(single line) 또는 8개(cross line). 여러 번 지정 가능.")

    args = parser.parse_args()

    if args.mode == 'train':
        pretrain_yolov8(
            DATASET_NAME=args.dataset_name,
            MODEL_NAME=args.model,
            EPOCHS=args.epochs,
            BATCH_SIZE=args.batch_size,
            IMG_SIZE=args.img_size,
        )
    elif args.mode == 'validation':
        validation_yolov8(
            DATASET_NAME=args.dataset_name,
            MODEL_NAME=args.model,
        )
    elif args.mode == 'video':
        if not args.video_filename:
            print("Error: --video_filename is required for video mode.")
            exit(1)
        video_object_counting(
            video_filename=args.video_filename,
            model_name=args.model,
            dataset_name=args.dataset_name,
            line_points_list=args.line_points_list,
        )


if __name__ == '__main__':
    # 실행 예시:
    # python main.py --mode video --dataset_name conveyor_belt --model yolov8n \
    #     --video_filename example_conveyor_20s.mp4 \
    #     --line_points_list 0 550 325 550 0 600 325 600 \
    #     --line_points_list 375 550 600 550 375 600 600 600 \
    #     --line_points_list 650 550 900 550 650 600 900 600
    main()
