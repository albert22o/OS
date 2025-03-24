#include <cuda.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

// Добавим проверку ошибок CUDA для отладки
#define CUDA_CHECK(call)                                                    \
    do                                                                      \
    {                                                                       \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess)                                             \
        {                                                                   \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                 << cudaGetErrorString(err) << endl;                        \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

__global__ void gTransposition(int *a, int *b, int N, int K)
{
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (k < K && n < N)
    { // Проверка границ
        b[n + k * N] = a[k + n * K];
    }
}

int main()
{
    const int num = 1 << 12;
    int N = 4 * num; // 16384
    int K = 8 * num; // 32768

    // Оптимальный размер блока (16x16=256 потоков)
    const int block_size = 16;
    dim3 threads_per_block(block_size, block_size);

    float elapsedTime;
    int *GPU_pre_matrix, *local_pre_matrix, *GPU_after_matrix, *local_after_matrix;
    cudaEvent_t start, stop;

    // Инициализация событий CUDA с проверкой ошибок
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Выделение памяти с проверкой ошибок
    CUDA_CHECK(cudaMalloc((void **)&GPU_pre_matrix, N * K * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&GPU_after_matrix, N * K * sizeof(int)));

    local_pre_matrix = (int *)malloc(N * K * sizeof(int));
    local_after_matrix = (int *)malloc(N * K * sizeof(int));

    if (!local_pre_matrix || !local_after_matrix)
    {
        cerr << "Host memory allocation failed" << endl;
        exit(EXIT_FAILURE);
    }

    // Инициализация исходной матрицы
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            local_pre_matrix[j + i * K] = j + i * K + 1;
        }
    }

    // Копирование на устройство
    CUDA_CHECK(cudaMemcpy(GPU_pre_matrix, local_pre_matrix, N * K * sizeof(int),
                          cudaMemcpyHostToDevice));

    // Вычисление размеров grid с округлением вверх
    dim3 blocks_per_grid((K + threads_per_block.x - 1) / threads_per_block.x,
                         (N + threads_per_block.y - 1) / threads_per_block.y);

    // Замер времени
    CUDA_CHECK(cudaEventRecord(start, nullptr));

    // Запуск ядра
    gTransposition<<<blocks_per_grid, threads_per_block>>>(GPU_pre_matrix, GPU_after_matrix, N, K);
    CUDA_CHECK(cudaGetLastError()); // Проверка ошибок ядра

    CUDA_CHECK(cudaEventRecord(stop, nullptr));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Копирование результатов обратно (исправлен тип на sizeof(int))
    CUDA_CHECK(cudaMemcpy(local_after_matrix, GPU_after_matrix, N * K * sizeof(int),
                          cudaMemcpyDeviceToHost));

    // Вывод времени выполнения
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    cout << "CUDA Event time: " << elapsedTime << " ms" << endl;

    // Проверка корректности транспонирования (выборочная)
    bool correct = true;
    for (int i = 0; i < min(10, N) && correct; ++i)
    {
        for (int j = 0; j < min(10, K) && correct; ++j)
        {
            if (local_after_matrix[i + j * N] != local_pre_matrix[j + i * K])
            {
                cout << "Error at position [" << i << "][" << j << "]" << endl;
                correct = false;
            }
        }
    }
    if (correct)
    {
        cout << "Transposition verified successfully (sample check)" << endl;
    }

    // Освобождение памяти
    CUDA_CHECK(cudaFree(GPU_pre_matrix));
    CUDA_CHECK(cudaFree(GPU_after_matrix));
    free(local_pre_matrix);
    free(local_after_matrix);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
