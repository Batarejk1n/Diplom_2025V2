FROM ubuntu:22.04

# Установка зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    unzip \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Установка LibTorch
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip \
    && unzip libtorch*.zip -d /usr/local \
    && rm libtorch*.zip

# Настройка переменных окружения
ENV Torch_DIR=/usr/local/libtorch
ENV LD_LIBRARY_PATH=/home/orange/nerualLink/libtorch-shared-with-deps-2.6.0+cu126/libtorch/lib:$LD_LIBRARY_PATH

# Копирование исходников
WORKDIR /app
COPY . /app

# Сборка проекта
RUN mkdir build && cd build \
    && cmake -DCMAKE_PREFIX_PATH=/usr/local/libtorch .. \
    && cmake --build . --config Release

# Запуск
CMD ["./build/face_recognition"]