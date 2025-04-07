#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define N 512 // Размер сетки
#define PI 3.14159265358979323846

// Текстура для доступа к данным
texture<float, 2, cudaReadModeElementType> texData;

// Константная память для параметров сферы
__constant__ float sphereParams[3]; // радиус сферы и шаги сетки

// Функция, которую мы интегрируем (например, sin(theta) * cos(phi))
__device__ float func(float theta, float phi)
{
    return sinf(theta) * cosf(phi);
}

// Ядро CUDA для вычисления интеграла с использованием текстурной памяти
__global__ void computeIntegralWithTexture(float *result)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < N && idy < N)
    {
        // Углы на сфере
        float theta = (float)idx * 2 * PI / N;
        float phi = (float)idy * PI / N;

        // Получаем значение функции в точке (theta, phi)
        float value = func(theta, phi);

        // Площадь элемента на сфере
        float dA = sinf(theta) * (2 * PI / N) * (PI / N);

        // Суммируем интеграл
        atomicAdd(result, value * dA);
    }
}

// Основная функция
int main()
{
    float *d_result, h_result = 0.0f;
    cudaMalloc(&d_result, sizeof(float));
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    // Инициализируем константную память для параметров сферы
    float sphereParamsHost[3] = {1.0f, 2 * PI / N, PI / N};
    cudaMemcpyToSymbol(sphereParams, sphereParamsHost, sizeof(float) * 3);

    // Настроим параметры сетки и блока
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Измерение времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Запуск ядра для вычисления интеграла с использованием текстурной памяти
    computeIntegralWithTexture<<<gridSize, blockSize>>>(d_result);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time with texture memory: " << milliseconds << " ms" << std::endl;

    // Копирование результата на хост
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Integral result with texture memory: " << h_result << std::endl;

    // Освобождение памяти
    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
