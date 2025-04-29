import numpy as np
from numba import cuda, float32
import time
import os

# Настройки среды
os.environ['NUMBA_ENABLE_CUDASIM'] = '0'
os.environ['NUMBA_CUDA_DEBUGINFO'] = '0'

# Размер плитки
BLOCK_SIZE = 16

@cuda.jit
def matrixMul_optimized(A, B, C, N):
    # Shared memory
    sA = cuda.shared.array((BLOCK_SIZE, BLOCK_SIZE), dtype=float32)
    sB = cuda.shared.array((BLOCK_SIZE, BLOCK_SIZE), dtype=float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    row = by * BLOCK_SIZE + ty
    col = bx * BLOCK_SIZE + tx

    tmp = 0.0

    for m in range(N // BLOCK_SIZE):
        # Загружаем плитки A и B в shared memory
        sA[ty, tx] = A[row, m * BLOCK_SIZE + tx]
        sB[ty, tx] = B[m * BLOCK_SIZE + ty, col]
        
        cuda.syncthreads()

        # Перемножение текущих плиток
        for k in range(BLOCK_SIZE):
            tmp += sA[ty, k] * sB[k, tx]

        cuda.syncthreads()

    C[row, col] = tmp


def main():
    N = 512
    assert N % BLOCK_SIZE == 0, "Размер должен быть кратен BLOCK_SIZE"

    # Хостовые матрицы
    A = np.ones((N, N), dtype=np.float32)
    B = np.ones((N, N), dtype=np.float32)
    C = np.zeros((N, N), dtype=np.float32)

    # Копирование на устройство
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.device_array_like(C)

    # Настройка сетки
    threads_per_block = (BLOCK_SIZE, BLOCK_SIZE)
    blocks_per_grid = (N // BLOCK_SIZE, N // BLOCK_SIZE)

    # Прогрев
    matrixMul_optimized[blocks_per_grid, threads_per_block](d_A, d_B, d_C, N)
    cuda.synchronize()

    # Точное измерение времени
    start = cuda.event()
    end = cuda.event()
    start.record()

    matrixMul_optimized[blocks_per_grid, threads_per_block](d_A, d_B, d_C, N)

    end.record()
    end.synchronize()
    elapsed_ms = cuda.event_elapsed_time(start, end)

    # Копирование результата
    d_C.copy_to_host(C)
    expected = N
    correct = np.allclose(C, expected, atol=1e-3)

    print(f"Numba \t\t\t\tTime: {elapsed_ms:.3f} ms")


if __name__ == "__main__":
    main()
