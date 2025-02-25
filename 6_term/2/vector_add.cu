#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <chrono>
using namespace std;
typedef std::chrono::milliseconds ms;
typedef std::chrono::nanoseconds ns;

__global__ void vectorAdd(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        // Искусственно вызываем ошибку для некоторых нитей
        if (i % 100 == 0)
        {
            c[i] = a[i + 1000] + b[i]; // Обращение к элементу за пределами массива
        }
        else
        {
            c[i] = a[i] + b[i];
        }
    }
}

int main()
{
    for (long n = 10; n <= 100000000; n *= 10)
    {
        cout << endl
             << "n = " << n << endl;
        float elapsedTime;
        cudaEvent_t start, stop;
        chrono::time_point<chrono::system_clock> start_chrono, end_chrono;

        float *d_a, *d_b, *d_c;
        cudaMalloc((void **)&d_a, n * sizeof(float));
        cudaMalloc((void **)&d_b, n * sizeof(float));
        cudaMalloc((void **)&d_c, n * sizeof(float));

        float *h_a = new float[n];
        float *h_b = new float[n];
        for (int i = 0; i < n; ++i)
        {
            h_a[i] = i;
            h_b[i] = i * 2;
        }

        cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

        int blockSize = 1024; // Количество нитей в блоке
        int numBlocks = n;    // Количество блоков

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        start_chrono = chrono::system_clock::now();
        vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

        // Проверка ошибок выполнения ядра
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
        }

        cudaDeviceSynchronize(); // Синхронизация для проверки ошибок
        cudaEventRecord(stop, 0);
        end_chrono = chrono::system_clock::now();

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);

        float *h_c = new float[n];
        cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

        cout << "CUDA Event time: " << elapsedTime * 1000 << "ns" << endl
             << "Chrono time: " << chrono::duration_cast<ms>(end_chrono - start_chrono).count() << "ms"
             << endl
             << chrono::duration_cast<ns>(end_chrono - start_chrono).count() << "ns" << endl;

        delete[] h_a;
        delete[] h_b;
        delete[] h_c;
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }
    return 0;
}
