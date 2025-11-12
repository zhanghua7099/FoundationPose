#!/usr/bin/env bash
set -euo pipefail

CONTAINER="foundationpose"
IMAGE="foundationpose:latest"
DIR="$(cd .. && pwd)"  # 稳妥获取 $(pwd)/../

# 可选参数：--recreate 强制重建容器
if [[ "${1-}" == "--recreate" ]]; then
  if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    docker rm -f "$CONTAINER"
  fi
fi

# X11 授权（比 xhost + 更安全，只允许本机 root）
if command -v xhost >/dev/null 2>&1; then
  xhost +local:root >/dev/null 2>&1 || true
fi

# 若容器不存在：创建并进入
if ! docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo ">> Creating container: $CONTAINER"
  exec docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 \
    -it --network host --name "$CONTAINER" \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -v "$DIR":"$DIR" \
    -v /home:/home \
    -v /mnt:/mnt \
    -v /tmp:/tmp \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --ipc host \
    -e DISPLAY="$DISPLAY" -e GIT_INDEX_FILE \
    "$IMAGE" bash -lc "cd '$DIR'; exec bash"
fi

# 已存在：若未运行则先启动
if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" != "true" ]]; then
  echo ">> Starting container: $CONTAINER"
  docker start "$CONTAINER" >/dev/null
fi

# 直接进入容器，并切到工作目录
exec docker exec -it \
  -e DISPLAY="$DISPLAY" \
  -w "$DIR" \
  "$CONTAINER" bash -l
