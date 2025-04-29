#!/bin/bash
# Скрипт для компиляции и запуска всех программ по порядку

# [V] Создание папки obj, если она не существует
if [ ! -d obj ]; then
    mkdir -p obj
fi

# [V] Также собираем PTX файл для работы Driver API и PyCuda
nvcc -ptx -O3 -arch=sm_86 matrix_mul.cu -o obj/matrix_mul.ptx

# [V] 1. CUDA API
nvcc cuda_api.cu -o obj/cuda_api -arch=sm_86
./obj/cuda_api

# [V] 2. CUDA Driver API (C/C++)
nvcc driver_api.cpp -o obj/driver_api -lcuda -lcudart -arch=sm_86
./obj/driver_api

# [V] 3. CUDA Driver API (Python)
python3 driver_api.py

# [X] 4. Numba
#python3 numba_impl.py

# [X] 5. PyCuda
#python3 pycuda_impl.py
