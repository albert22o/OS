#include <stdio.h>
#include <cuda_runtime.h>

#define N (1 << 20)

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    // Выделение памяти на хосте
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    // Инициализация векторов
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // Выделение памяти на устройстве
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Копирование данных на устройство
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Список размеров блоков для тестирования
    int block_sizes[] = {1, 16, 32, 64, 128, 256, 512, 1024};
    int num_sizes = sizeof(block_sizes) / sizeof(int);

    // События для замера времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < num_sizes; ++i) {
        int block_size = block_sizes[i];
        int grid_size = (N + block_size - 1) / block_size;

        cudaEventRecord(start);
        vectorAdd<<<grid_size, block_size>>>(d_a, d_b, d_c, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Block size: %4d, Time: %f ms\n", block_size, milliseconds);

        // Проверка ошибок
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error: %s\n", cudaGetErrorString(err));
        }
    }

    // Копирование результата обратно
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Проверка корректности
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != 3.0f) {
            correct = false;
            break;
        }
    }
    printf("Result: %s\n", correct ? "Correct" : "Incorrect");

    // Освобождение ресурсов
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}