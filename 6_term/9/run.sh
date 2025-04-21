#!/bin/bash
# Скрипт для компиляции и запуска всех программ по порядку

# Создание папки obj, если она не существует
if [ ! -d obj ]; then
    mkdir -p obj
fi

# 1. CUDA API
nvcc cuda_api.cu -o obj/cuda_api -arch=sm_86
./obj/cuda_api

# 2. CUDA Driver API (C/C++)
nvcc -ptx matrix_mul.cu -arch=sm_86 -o obj/matrix_mul.ptx
nvcc driver_api.cpp -o obj/driver_api -lcuda -lcudart
./obj/driver_api

# 3. CUDA Driver API (Python)


# 4. Numba


# 5. PyCuda

