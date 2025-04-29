// matrix_mul.cu
#define BLOCK_SIZE 16

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

    for (int m = 0; m < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m)
    {
        int tiled_col = m * BLOCK_SIZE + tx;
        int tiled_row = m * BLOCK_SIZE + ty;

        sA[ty][tx] = (row < N && tiled_col < N) ? A[row * N + tiled_col] : 0.0f;
        sB[ty][tx] = (tiled_row < N && col < N) ? B[tiled_row * N + col] : 0.0f;

        __syncthreads();

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