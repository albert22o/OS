/*
nvcc cuda_api.cu -o cuda_api -arch=sm_86
./cuda_api
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16 // Оптимальный размер блока для матричных операций

// Ядро для умножения матриц
__global__ void matrixMul(float *A, float *B, float *C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Глобальная координата Y
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Глобальная координата X

    if (row < N && col < N)
    {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
        { // Вычисление элемента C[row][col]
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main()
{
    int N = 512;                         // Размер матрицы N x N
    size_t size = N * N * sizeof(float); // Общий объем данных

    // Выделение памяти на хосте
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Инициализация матриц
    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    // Выделение памяти на устройстве
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Копирование данных на устройство
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Конфигурация запуска ядра
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);                                              // Блок 16x16 потоков
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE); // Сетка блоков

    // Замер времени выполнения
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matrixMul<<<grid, block>>>(d_A, d_B, d_C, N); // Запуск ядра

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("CUDA API Time: %.3f ms\n", milliseconds);

    // Освобождение памяти
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}