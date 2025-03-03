cmake_minimum_required(VERSION 3.12)
project(face_recognition)

# Настройки компилятора
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Путь к LibTorch
set(LIBTORCH_PATH "/home/orange/nerualLink/libtorch-shared-with-deps-2.6.0+cu126/libtorch")

# Поиск LibTorch
find_package(Torch REQUIRED PATHS ${LIBTORCH_PATH})

# Поиск OpenCV
find_package(OpenCV REQUIRED)

# Добавление исполняемого файла
add_executable(face_recognition
    src/main.cpp
    src/data_loader.cpp
    src/model.cpp
)

# Подключение заголовочных файлов
target_include_directories(face_recognition
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# Линковка библиотек
target_link_libraries(face_recognition
    PRIVATE
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)