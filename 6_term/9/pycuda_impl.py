# pip install pycuda numpy
# nvcc -ptx matrix_mul.cu -arch=sm_86
# python3 pycuda_impl.py
# ожидаемое время выполнения: ~600 μs

import pycuda.autoinit  
import pycuda.driver as drv
import numpy as np
from pycuda import gpuarray
import time

N = 512
BLOCK_SIZE = 16

# Загрузка PTX-модуля (предварительно сгенерируйте файл)
mod = drv.module_from_file("obj/matrix_mul.ptx")
matrixMul = mod.get_function("matrixMul")

# Инициализация данных
A = np.ones((N, N), np.float32)
B = np.ones((N, N), np.float32)
C = np.empty((N, N), np.float32)

# Копирование данных на устройство
d_A = gpuarray.to_gpu(A)
d_B = gpuarray.to_gpu(B)
d_C = gpuarray.empty_like(d_A)

# Конфигурация запуска
grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE, (N + BLOCK_SIZE - 1) // BLOCK_SIZE)

# Тайминг через события CUDA
start = drv.Event()
end = drv.Event()
start.record()

matrixMul(d_A, d_B, d_C, np.int32(N),
          block=(BLOCK_SIZE, BLOCK_SIZE, 1),
          grid=(grid[0], grid[1]))

end.record()
end.synchronize()

elapsed = start.time_till(end)  # Переводим в микросекунды
print(f"PyCUDA Time: {elapsed:.3f} ms")