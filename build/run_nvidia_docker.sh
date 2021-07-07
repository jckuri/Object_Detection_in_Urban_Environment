# docker run --runtime=nvidia -it -p "8265:8265" -p "8888:8888" -v "$PWD/..:/app/project/" --shm-size=21g project-dev 
# docker run -v <PATH TO LOCAL PROJECT FOLDER>:/app/project/ -ti project-dev bash
docker run --gpus all -p "8265:8265" -p "8888:8888" -v "$PWD/..:/app/project/" --shm-size=21g -ti project-dev bash
