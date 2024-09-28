# main.py

import argparse
from src.train import pretrain_yolov8
from src.validation import validation_yolov8
from src.video_inference import video_object_counting

def main():
    # Argument parser를 통해 사용자로부터 입력을 받습니다.
    parser = argparse.ArgumentParser(description="YOLOv8 Training, Validation, and Video Object Counting Script")

    # 공통 인자 추가
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'validation', 'video'],
                        help="Mode to run: 'train' for training, 'validation' for model inference, 'video' for object counting on video.")
    parser.add_argument('--dataset_name', type=str, required=True,  
                        help="Dataset name to use for training/validation (required)")
    parser.add_argument('--model', type=str, default='yolov8n', choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l'], 
                        help="Choose the model to train or validate: yolov8n, yolov8s, yolov8m, or yolov8l (default: yolov8n)")
    parser.add_argument('--epochs', type=int, default=30, help="Number of epochs to train (default: 30)")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training (default: 16)")
    parser.add_argument('--img_size', type=int, default=640, help="Image size for training/validation (default: 640)")
    parser.add_argument('--test_results_dir', type=str, default='./results', help="Directory to save validation results (default: ./results)")

    # Video object counting 전용 인자 추가
    parser.add_argument('--video_filename', type=str, help="Name of the input video file for object counting (located in ./videos/ directory)")
    parser.add_argument('--line_points_list', type=int, nargs='+', action='append', 
                        help="Multiple sets of coordinates for counting lines. Use 4 integers for normal mode or 8 for cross mode.")

    # 인자 파싱
    args = parser.parse_args()

    # 모드에 따른 동작
    if args.mode == 'train':
        pretrain_yolov8(DATASET_NAME=args.dataset_name, MODEL_NAME=args.model, EPOCHS=args.epochs, 
                        BATCH_SIZE=args.batch_size, IMG_SIZE=args.img_size)
    elif args.mode == 'validation':
        validation_yolov8(DATASET_NAME=args.dataset_name, MODEL_NAME=args.model)
    elif args.mode == 'video':
        # video_filename이 제공되지 않은 경우 에러 출력
        if not args.video_filename:
            print("Error: --video_filename is required for video mode.")
            exit(1)
        
        # video_object_counting 함수 호출
        video_object_counting(
            video_filename=args.video_filename,
            model_name=args.model,
            dataset_name=args.dataset_name,
            line_points_list=args.line_points_list
        )

if __name__ == '__main__':
    # python main.py --mode video --dataset_name conveyor_belt --model yolov8n --video_filename example_conveyor_20s.mp4 --line_points_list 0 550 325 550 0 600 325 600 --line_points_list 375 550 600 550 375 600 600 600 --line_points_list 650 550 900 550 650 600 900 600
    main()


