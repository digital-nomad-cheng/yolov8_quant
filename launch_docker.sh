# docker run -it -v ./:/workspace --gpus all yolov8_quant:latest
# docker run -it --gpus all -v ./:/workspace yolov8_quant:latest
# docker run -it --gpus all --shm-size=8gb --cap-add=SYS_ADMIN -v ./:/workspace dlmc:work
docker run -it --gpus all --shm-size=8gb --cap-add=SYS_ADMIN -v ./:/workspace nvcr.io/nvidia/pytorch:23.05-py3
