#!/usr/bin/env bash

# Important: run this script in the host.

image_name='rocm/jax-maxtext-training-private:20250530-te2.1-fp8-tf2.18.1'

if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep --quiet "${image_name}"; then
    echo 'Image does not exist. Pulling it...'
    echo 'Use rocmshared token to login.'
    docker login --username rocmshared
    docker pull "${image_name}"
    docker logout
else
    echo 'Image already exists.'
fi

user_name=$(id --user --name)
container_name="${user_name}_axlearn_fa"

container_exists=$(
    docker ps --all --format '{{.Names}}' \
    | grep --word-regexp "${container_name}"
)

if [ -z "${container_exists}" ]; then
    echo 'Container does not exist. Creating it...'
    docker run \
        -it \
        --detach \
        --name "${container_name}" \
        --network host \
        --ipc host \
        --device /dev/kfd \
        --device /dev/dri \
        --security-opt seccomp=unconfined \
        --cap-add SYS_PTRACE \
        --group-add video \
        --shm-size=16G \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --mount "type=bind,source=${HOME},target=/hhome" \
        --mount "type=bind,source=${HOME}/.ssh,target=/root/.ssh,readonly" \
        --workdir /workspace \
        "${image_name}"
else
    echo 'Container already exists.'
fi

container_status() {
    container_name="${1}"
    docker inspect --format '{{.State.Status}}' "${container_name}"
}

wait_for_running() {
    container_name="${1}"
    echo 'Waiting for container to be running...'
    while [ "$(container_status "${container_name}")" != running ]; do
        sleep 1
    done
}

status=$(container_status "${container_name}")

case "${status}" in
    exited)
        echo 'Container is exited. Restarting it...'
        docker restart "${container_name}"
        wait_for_running "${container_name}"
        ;;
    paused)
        echo 'Container is paused. Unpausing it...'
        docker unpause "${container_name}"
        wait_for_running "${container_name}"
        ;;
    running)
        echo 'Container is running.'
        ;;
    *)
        echo "Unexpected container state: [${status}]"
        echo 'Exiting...'
        exit 1
        ;;
esac

docker exec -it "${container_name}" bash
