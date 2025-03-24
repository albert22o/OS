#include <cuda.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

__global__ void gTransposition(int *a, int *b, int N, int K)
{
    unsigned int k = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int n = threadIdx.y + blockIdx.y * blockDim.y;
    b[n + k * N] = a[k + n * K];
}

int main()
{
    const int num = 1 << 12;
    int N = 4 * num, K = 8 * num, threads_per_block = 128;
    float elapsedTime;
    int *GPU_pre_matrix, *local_pre_matrix, *GPU_after_matrix, *local_after_matrix;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc((void **)&GPU_pre_matrix, N * K * sizeof(int));
    cudaMalloc((void **)&GPU_after_matrix, N * K * sizeof(int));
    local_pre_matrix = (int *)calloc(N * K, sizeof(int));
    local_after_matrix = (int *)calloc(N * K, sizeof(int));

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            local_pre_matrix[j + i * K] = j + i * K + 1;
        }
    }


    cudaMemcpy(GPU_pre_matrix, local_pre_matrix, K * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(start, nullptr);

    gTransposition<<<dim3((K + threads_per_block - 1) / threads_per_block, (N + threads_per_block - 1) / threads_per_block),
                     dim3(threads_per_block, threads_per_block)>>>(GPU_pre_matrix, GPU_after_matrix, N, K);

    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    cudaMemcpy(local_after_matrix, GPU_after_matrix, K * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventElapsedTime(&elapsedTime, start, stop);

    cout << "CUDA Event time:\n\t"
         << elapsedTime
         << endl;

    cudaFree(GPU_pre_matrix);
    cudaFree(GPU_after_matrix);
    free(local_pre_matrix);
    free(local_after_matrix);

    return 0;
}
