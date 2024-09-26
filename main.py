# # main.py

# import argparse
# from src.train import pretrain_yolov8
# from src.validation import validation_yolov8
# from src.video_inference import video_object_counting

# def main():
#     # Argument parser를 통해 사용자로부터 입력을 받습니다.
#     parser = argparse.ArgumentParser(description="YOLOv8 Training, Validation, and Video Object Counting Script")

#     # 공통 인자 추가
#     parser.add_argument('--mode', type=str, required=True, choices=['train', 'validation', 'video'],
#                         help="Mode to run: 'train' for training, 'validation' for model inference, 'video' for object counting on video.")
#     parser.add_argument('--dataset_name', type=str, required=True,  
#                         help="Dataset name to use for training/validation (required)")
#     parser.add_argument('--model', type=str, default='yolov8n', choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l'], 
#                         help="Choose the model to train or validate: yolov8n, yolov8s, yolov8m, or yolov8l (default: yolov8n)")
#     parser.add_argument('--epochs', type=int, default=30, help="Number of epochs to train (default: 30)")
#     parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training (default: 16)")
#     parser.add_argument('--img_size', type=int, default=640, help="Image size for training/validation (default: 640)")
#     parser.add_argument('--test_results_dir', type=str, default='./results', help="Directory to save validation results (default: ./results)")

#     # Video object counting 전용 인자 추가
#     parser.add_argument('--video_filename', type=str, help="Name of the input video file for object counting (located in ./videos/ directory)")
#     parser.add_argument('--line_points', type=int, nargs='+', metavar='x1 y1 x2 y2 [x3 y3 x4 y4]',
#                         help="Coordinates for the counting lines. Provide four integers for a single line (x1 y1 x2 y2) or eight integers for two lines (x1 y1 x2 y2 x3 y3 x4 y4).")
#     parser.add_argument('--cross_mode', action='store_true', help="Enable cross counting mode with two lines")

#     # 인자 파싱
#     args = parser.parse_args()

#     # 모드에 따른 동작
#     if args.mode == 'train':
#         pretrain_yolov8(DATASET_NAME=args.dataset_name, MODEL_NAME=args.model, EPOCHS=args.epochs, 
#                         BATCH_SIZE=args.batch_size, IMG_SIZE=args.img_size)
#     elif args.mode == 'validation':
#         validation_yolov8(DATASET_NAME=args.dataset_name, MODEL_NAME=args.model)
#     elif args.mode == 'video':
#         # video_filename이 제공되지 않은 경우 에러 출력
#         if not args.video_filename:
#             print("Error: --video_filename is required for video mode.")
#             exit(1)
#         # line_points가 제공되지 않았거나 올바른 개수가 아닌 경우 에러 출력
#         if not args.line_points:
#             print("Error: --line_points is required for video mode.")
#             exit(1)
        
#         if args.cross_mode:
#             if len(args.line_points) != 8:
#                 print("Error: --cross_mode is enabled. Provide eight integers representing two lines (x1 y1 x2 y2 x3 y3 x4 y4).")
#                 exit(1)
#             # 두 개의 선을 언팩
#             line_p1 = (args.line_points[0], args.line_points[1])
#             line_p2 = (args.line_points[2], args.line_points[3])
#             line_p3 = (args.line_points[4], args.line_points[5])
#             line_p4 = (args.line_points[6], args.line_points[7])
#             line_points = [line_p1, line_p2, line_p3, line_p4]
#             cross_mode = True
#         else:
#             if len(args.line_points) != 4:
#                 print("Error: --line_points requires four integers representing one line (x1 y1 x2 y2).")
#                 exit(1)
#             # 한 개의 선을 언팩
#             line_p1 = (args.line_points[0], args.line_points[1])
#             line_p2 = (args.line_points[2], args.line_points[3])
#             line_points = [line_p1, line_p2]
#             cross_mode = False

#         # video_object_counting 함수 호출
#         video_object_counting(
#             video_filename=args.video_filename,
#             model_name=args.model,
#             dataset_name=args.dataset_name,
#             line_points=line_points,
#             cross_mode=cross_mode
#         )

# if __name__ == '__main__':
#     # 실행 예시:
#     # Training: python main.py --mode train --dataset_name conveyor_belt --model yolov8n --epochs 100 --batch_size 16 --img_size 640
#     # Validation: python main.py --mode validation --dataset_name conveyor_belt --model yolov8n
#     # Video Object Counting: python main.py --mode video --dataset_name conveyor_belt --model yolov8n --video_filename example_conveyor_20s.mp4 --line_points 100 200 500 600
#     # Video Object Counting with Cross Mode: python main.py --mode video --dataset_name conveyor_belt --model yolov8n --video_filename example_conveyor_20s.mp4 --cross_mode --line_points 0 550 350 550 0 600 350 600

#     main()

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


