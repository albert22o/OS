# nvcc -ptx matrix_mul.cu -arch=sm_86
# python3 driver_api.py
# ожидаемое время выполнения: ~1200 μs

import numpy as np
from ctypes import c_void_p, c_float, c_int, c_char_p, POINTER, byref, cast, create_string_buffer
import sys
import cuda_driver as cuda 

# Константы
CUDA_SUCCESS = cuda.CUDA_SUCCESS
block_size = 16  # Размер блока (16x16 потоков)

def check_cuda_error(err_code):
    """Проверка кода ошибки CUDA и генерация исключения при необходимости."""
    if err_code != CUDA_SUCCESS:
        err_str = c_char_p()
        cuda.cuGetErrorString(err_code, byref(err_str))
        raise RuntimeError(f"CUDA ошибка: {err_str.value.decode()}")

def main():
    # Инициализация CUDA
    check_cuda_error(cuda.cuInit(0))

    # Получение количества устройств
    device_count = c_int()
    check_cuda_error(cuda.cuDeviceGetCount(byref(device_count)))
    if device_count.value == 0:
        raise RuntimeError("Нет доступных CUDA-устройств")
    print(f"Найдено устройств: {device_count.value}")

    # Получение дескриптора устройства
    device = c_int()
    check_cuda_error(cuda.cuDeviceGet(byref(device), 0))

    # Создание контекста
    context = c_void_p()
    check_cuda_error(cuda.cuCtxCreate(byref(context), 0, device))

    # Загрузка PTX-модуля
    module = c_void_p()
    with open("obj/matrix_mul.ptx", "rb") as f:
        ptx_data = f.read()
    check_cuda_error(cuda.cuModuleLoadData(byref(module), ptx_data))

    # Получение функции ядра
    kernel_func = c_void_p()
    kernel_name = b"matmul"
    check_cuda_error(cuda.cuModuleGetFunction(byref(kernel_func), module, kernel_name))

    # Параметры матриц
    M, N, K = 512, 512, 512  # Размеры матриц (MxN) * (NxK) = (MxK)
    size_A = M * N * np.dtype(np.float32).itemsize
    size_B = N * K * np.dtype(np.float32).itemsize
    size_C = M * K * np.dtype(np.float32).itemsize

    # Выделение памяти на устройстве
    d_A = c_void_p()
    d_B = c_void_p()
    d_C = c_void_p()
    check_cuda_error(cuda.cuMemAlloc(byref(d_A), size_A))
    check_cuda_error(cuda.cuMemAlloc(byref(d_B), size_B))
    check_cuda_error(cuda.cuMemAlloc(byref(d_C), size_C))

    # Инициализация данных на хосте
    A = np.random.randn(M, N).astype(np.float32)
    B = np.random.randn(N, K).astype(np.float32)
    C_host = np.zeros((M, K), dtype=np.float32)

    # Копирование данных на устройство
    check_cuda_error(cuda.cuMemcpyHtoD(d_A, A.ctypes.data, size_A))
    check_cuda_error(cuda.cuMemcpyHtoD(d_B, B.ctypes.data, size_B))

    # Настройка параметров запуска ядра
    grid_x = (K + block_size - 1) // block_size
    grid_y = (M + block_size - 1) // block_size
    args = [d_A, d_B, d_C, M, N, K]
    args = [cast(byref(arg), c_void_p) for arg in args]

    # Создание событий для замера времени
    start_event = c_void_p()
    end_event = c_void_p()
    check_cuda_error(cuda.cuEventCreate(byref(start_event), 0))
    check_cuda_error(cuda.cuEventCreate(byref(end_event), 0))

    # Запуск ядра и замер времени
    check_cuda_error(cuda.cuEventRecord(start_event, 0))
    check_cuda_error(cuda.cuLaunchKernel(
        kernel_func,
        grid_x, grid_y, 1,  # grid dimensions
        block_size, block_size, 1,  # block dimensions
        0, 0,  # shared memory and stream
        (c_void_p * len(args))(*args), 0
    ))
    check_cuda_error(cuda.cuEventRecord(end_event, 0))
    check_cuda_error(cuda.cuEventSynchronize(end_event))

    # Расчет времени выполнения
    time_ms = c_float()
    check_cuda_error(cuda.cuEventElapsedTime(byref(time_ms), start_event, end_event))
    print(f"Время выполнения: {time_ms.value} мс")

    # Копирование результата на хост
    check_cuda_error(cuda.cuMemcpyDtoH(C_host.ctypes.data, d_C, size_C))

    # Проверка результата с помощью numpy
    C_np = np.dot(A, B)
    if np.allclose(C_host, C_np, atol=1e-3):
        print("Результат верный")
    else:
        print("Результат неверный")

    # Освобождение ресурсов
    check_cuda_error(cuda.cuMemFree(d_A))
    check_cuda_error(cuda.cuMemFree(d_B))
    check_cuda_error(cuda.cuMemFree(d_C))
    check_cuda_error(cuda.cuEventDestroy(start_event))
    check_cuda_error(cuda.cuEventDestroy(end_event))
    check_cuda_error(cuda.cuModuleUnload(module))
    check_cuda_error(cuda.cuCtxDestroy(context))

if __name__ == "__main__":
    main()
