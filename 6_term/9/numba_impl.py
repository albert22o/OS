# pip install numba numpy
# python3 numba_impl.py
# ожидаемое время выполнения: ~800 μs

import numpy as np
from numba import cuda
import time

@cuda.jit
def matrixMul(A, B, C, N):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y  # Глобальная Y-координата
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # Глобальная X-координата
    
    if row < N and col < N:
        tmp = 0.0
        for k in range(N):  # Вычисление элемента C[row][col]
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

N = 512
BLOCK_SIZE = 16

# Инициализация данных
A = np.ones((N, N), np.float32)
B = np.ones((N, N), np.float32)
C = np.empty((N, N), np.float32)

# Копирование данных на устройство
d_A = cuda.to_device(A)
d_B = cuda.to_device(B)
d_C = cuda.device_array_like(C)

# Конфигурация запуска
grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE, (N + BLOCK_SIZE - 1) // BLOCK_SIZE)
block = (BLOCK_SIZE, BLOCK_SIZE)

# Тайминг
start = time.time()
matrixMul[grid, block](d_A, d_B, d_C, N)  # Запуск ядра
cuda.synchronize()  # Ожидание завершения
elapsed = (time.time() - start) * 1e6  # В микросекундах

print(f"Numba Time: {elapsed:.3f} μs")