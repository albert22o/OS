# pip install pycuda numpy
# nvcc -ptx matrix_mul.cu -arch=sm_86 -o obj/matrix_mul.ptx
# python3 pycuda_impl.py
# ожидаемое время выполнения: ~0.6 мс (600 мкс)

import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda import gpuarray

N = 512
BLOCK_SIZE = 16

# Загрузка PTX-модуля
mod = drv.module_from_file("obj/matrix_mul.ptx")
matrixMul = mod.get_function("matrixMul")

# Данные на хосте
A = np.ones((N, N), dtype=np.float32)
B = np.ones((N, N), dtype=np.float32)
C = np.empty((N, N), dtype=np.float32)

# Передача на GPU
d_A = gpuarray.to_gpu(A)
d_B = gpuarray.to_gpu(B)
d_C = gpuarray.empty_like(d_A)

# Размеры сетки
grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,
        (N + BLOCK_SIZE - 1) // BLOCK_SIZE)

# Таймер CUDA
start = drv.Event()
end = drv.Event()
start.record()

# ВАЖНО: передавать .gpudata, а не объекты gpuarray
matrixMul(d_A.gpudata, d_B.gpudata, d_C.gpudata, np.int32(N),
          block=(BLOCK_SIZE, BLOCK_SIZE, 1),
          grid=(grid[0], grid[1]))

end.record()
end.synchronize()

# Время в миллисекундах
elapsed = start.time_till(end)
print(f"PyCUDA \t\t\t\tTime: {elapsed:.3f} ms")
