sudo docker build -t mnist_deeplearning_classifier:0.0.0 .

xhost +local:root
docker run -it --rm \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    kimera_vio
# Disallow X server connection
xhost -local:root