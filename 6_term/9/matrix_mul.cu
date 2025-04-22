// matrix_mul.cu
#define BLOCK_SIZE 32

extern "C" __global__ void matrixMul(float *A, float *B, float *C, int N)
{
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float sum = 0.0f;

    for (int m = 0; m < N / BLOCK_SIZE; ++m)
    {
        // Загрузка тайлов в shared memory
        sA[ty][tx] = A[row * N + (m * BLOCK_SIZE + tx)];
        sB[ty][tx] = B[(m * BLOCK_SIZE + ty) * N + col];
        __syncthreads();

        // Вычисление суммы для подматриц
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            sum += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();
    }

    // Сохранение результата
    if (row < N && col < N)
        C[row * N + col] = sum;
}