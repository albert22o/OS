# nvcc -ptx matrix_mul.cu -arch=sm_86
# python3 driver_api.py
# ожидаемое время выполнения: ~1200 μs

import numpy as np
from cuda_driver import *
from ctypes import *
import sys

def check_cuda_error(res, msg):
    if res != CUDA_SUCCESS:
        error_str = c_char_p()
        cuGetErrorString(res, byref(error_str))
        print(f"Ошибка {msg}: {error_str.value.decode()}")
        sys.exit(1)

N = 512
BLOCK_SIZE = 16
size = N * N * np.float32().itemsize

# Инициализация CUDA
check_cuda_error(cuInit(0), "cuInit")

cuDevice = c_int(0)
check_cuda_error(cuDeviceGet(byref(cuDevice), 0), "cuDeviceGet")

cuContext = c_void_p()
check_cuda_error(cuCtxCreate(byref(cuContext), 0, cuDevice), "cuCtxCreate")

# Матрицы
h_A = np.ones((N, N), dtype=np.float32)
h_B = np.ones((N, N), dtype=np.float32)
h_C = np.zeros((N, N), dtype=np.float32)

# Выделение памяти на GPU
d_A = c_void_p(0)
d_B = c_void_p(0)
d_C = c_void_p(0)
check_cuda_error(cuMemAlloc(byref(d_A), size), "cuMemAlloc(d_A)")
check_cuda_error(cuMemAlloc(byref(d_B), size), "cuMemAlloc(d_B)")
check_cuda_error(cuMemAlloc(byref(d_C), size), "cuMemAlloc(d_C)")

# Копирование данных
check_cuda_error(cuMemcpyHtoD(d_A, h_A.ctypes.data, size), "HtoD(d_A)")
check_cuda_error(cuMemcpyHtoD(d_B, h_B.ctypes.data, size), "HtoD(d_B)")

# Загрузка PTX
cuModule = c_void_p()
check_cuda_error(cuModuleLoad(byref(cuModule), b"obj/matrix_mul.ptx"), "cuModuleLoad")

# Получение ядра
cuFunction = c_void_p()
check_cuda_error(cuModuleGetFunction(byref(cuFunction), cuModule, b"matrixMul"), "cuModuleGetFunction")

# Аргументы ядра
N_ctypes = c_int(N)
args = [
    cast(d_A, c_void_p),
    cast(d_B, c_void_p),
    cast(d_C, c_void_p),
    cast(byref(N_ctypes), c_void_p)
]
arg_ptrs = (c_void_p * len(args))(*args)

# Конфигурация запуска
gridX = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
gridY = gridX

# События для замера времени
start = c_void_p()
stop = c_void_p()
check_cuda_error(cuEventCreate(byref(start), 0), "cuEventCreate(start)")
check_cuda_error(cuEventCreate(byref(stop), 0), "cuEventCreate(stop)")

check_cuda_error(cuEventRecord(start, 0), "cuEventRecord(start)")
check_cuda_error(cuLaunchKernel(cuFunction, gridX, gridY, 1, BLOCK_SIZE, BLOCK_SIZE, 1, 0, 0, arg_ptrs, 0), "cuLaunchKernel")
cuCtxSynchronize()  # Синхронизация!
check_cuda_error(cuEventRecord(stop, 0), "cuEventRecord(stop)")
check_cuda_error(cuEventSynchronize(stop), "cuEventSynchronize")

# Время выполнения
time_ms = c_float()
check_cuda_error(cuEventElapsedTime(byref(time_ms), start, stop), "cuEventElapsedTime")
print(f"Время: {time_ms.value:.3f} мс")

# Проверка результатов
check_cuda_error(cuMemcpyDtoH(h_C.ctypes.data, d_C, size), "DtoH")
print("Проверка C[0][0]:", h_C[0, 0])  # Должно быть 512.0

# Очистка
cuMemFree(d_A)
cuMemFree(d_B)
cuMemFree(d_C)
cuCtxDestroy(cuContext)
