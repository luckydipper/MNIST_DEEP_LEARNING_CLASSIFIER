xhost +local:root
docker run -it --rm \
    --net=host --ipc=host \
    --env="DISPLAY=docker.for.mac.host.internal:0" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="./:/root/MNIST_DEEP_LEARNING_CLASSIFIER" \
    mnist_deeplearning_classifier:0.0.1
# Disallow X server connection
xhost -local:root