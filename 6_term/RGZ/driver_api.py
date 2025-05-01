import numpy as np
from ctypes import c_void_p, c_float, c_int, c_char_p, POINTER, byref, cast
import cuda_driver as cuda  # твоя обёртка над CUDA Driver API

# Константы
CUDA_SUCCESS = cuda.CUDA_SUCCESS
BLOCK_SIZE = 16
N = 512

def check_cuda_error(err_code):
    if err_code != CUDA_SUCCESS:
        err_str = c_char_p()
        cuda.cuGetErrorString(err_code, byref(err_str))
        raise RuntimeError(f"CUDA ошибка: {err_str.value.decode()}")

def main():
    # Инициализация CUDA
    check_cuda_error(cuda.cuInit(0))

    # Получение устройства
    device = c_int()
    check_cuda_error(cuda.cuDeviceGet(byref(device), 0))

    # Создание контекста
    context = c_void_p()
    check_cuda_error(cuda.cuCtxCreate(byref(context), 0, device))

    # Выделение памяти на хосте
    A = np.ones((N, N), dtype=np.float32)
    B = np.ones((N, N), dtype=np.float32)
    C_host = np.zeros((N, N), dtype=np.float32)

    # Выделение памяти на устройстве
    d_A = c_void_p()
    d_B = c_void_p()
    d_C = c_void_p()
    data_size = N * N * np.dtype(np.float32).itemsize

    check_cuda_error(cuda.cuMemAlloc(byref(d_A), data_size))
    check_cuda_error(cuda.cuMemAlloc(byref(d_B), data_size))
    check_cuda_error(cuda.cuMemAlloc(byref(d_C), data_size))

    # Копирование данных на устройство
    check_cuda_error(cuda.cuMemcpyHtoD(d_A, A.ctypes.data, data_size))
    check_cuda_error(cuda.cuMemcpyHtoD(d_B, B.ctypes.data, data_size))

    # Загрузка PTX
    module = c_void_p()
    with open("obj/matrix_mul.ptx", "rb") as f:
        ptx_data = f.read()
    check_cuda_error(cuda.cuModuleLoadData(byref(module), ptx_data))

    # Получение функции ядра
    kernel_func = c_void_p()
    check_cuda_error(cuda.cuModuleGetFunction(byref(kernel_func), module, b"matrixMul"))

    # Расчет размеров сетки
    grid_x = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_y = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Аргументы ядра
    N_cint = c_int(N)
    args = [d_A, d_B, d_C, N_cint]

    # Создаём массив указателей на аргументы
    kernel_params = (c_void_p * len(args))(
        cast(byref(args[0]), c_void_p),
        cast(byref(args[1]), c_void_p),
        cast(byref(args[2]), c_void_p),
        cast(byref(args[3]), c_void_p)
    )

    # Создание событий для измерения времени
    start_event = c_void_p()
    end_event = c_void_p()
    check_cuda_error(cuda.cuEventCreate(byref(start_event), 0))
    check_cuda_error(cuda.cuEventCreate(byref(end_event), 0))

    # Запуск ядра
    check_cuda_error(cuda.cuEventRecord(start_event, 0))
    check_cuda_error(cuda.cuLaunchKernel(
        kernel_func,
        grid_x, grid_y, 1,
        BLOCK_SIZE, BLOCK_SIZE, 1,
        0, 0,
        kernel_params, 0
    ))
    check_cuda_error(cuda.cuEventRecord(end_event, 0))
    check_cuda_error(cuda.cuEventSynchronize(end_event))

    # Замер времени
    time_ms = c_float()
    check_cuda_error(cuda.cuEventElapsedTime(byref(time_ms), start_event, end_event))
    print(f"CUDA Driver API (Python) \tTime: {time_ms.value:.3f} ms")

    # Копирование результата обратно на хост
    check_cuda_error(cuda.cuMemcpyDtoH(C_host.ctypes.data, d_C, data_size))

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
