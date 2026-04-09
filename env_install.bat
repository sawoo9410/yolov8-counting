set root=%homepath%\miniforge3

call %root%\Scripts\activate.bat %root%

call cd %~dp0
call conda create -n yolov8_tracking python=3.9

call conda activate yolov8_tracking
call pip install -r requirements.txt
call pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
call python ./src/download.py

pause