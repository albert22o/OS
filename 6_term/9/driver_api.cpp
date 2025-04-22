/*
nvcc -ptx matrix_mul.cu -arch=sm_86
nvcc driver_api.cpp -o driver_api -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda
./driver_api
ожидаемое время выполнения: ~550 μs
*/

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <cmath>

#define BLOCK_SIZE 16
const int N = 512;

void checkCUDAError(CUresult res, const char *msg)
{
    if (res != CUDA_SUCCESS)
    {
        const char *errorStr;
        cuGetErrorString(res, &errorStr);
        printf("Ошибка: %s (%s)\n", msg, errorStr);
        exit(1);
    }
}

int main()
{
    CUresult res;

    // Инициализация CUDA
    res = cuInit(0);
    checkCUDAError(res, "cuInit");

    CUdevice device;
    res = cuDeviceGet(&device, 0);
    checkCUDAError(res, "cuDeviceGet");

    CUcontext context;
    res = cuCtxCreate(&context, 0, device);
    checkCUDAError(res, "cuCtxCreate");

    // Выделение памяти на хосте
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];
    float *h_C_ref = new float[N * N]; // Ожидаемый результат

    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
        h_C_ref[i] = N; // Сумма N единиц
    }

    // Выделение памяти на устройстве
    CUdeviceptr d_A, d_B, d_C;
    res = cuMemAlloc(&d_A, N * N * sizeof(float));
    checkCUDAError(res, "cuMemAlloc d_A");
    res = cuMemAlloc(&d_B, N * N * sizeof(float));
    checkCUDAError(res, "cuMemAlloc d_B");
    res = cuMemAlloc(&d_C, N * N * sizeof(float));
    checkCUDAError(res, "cuMemAlloc d_C");

    // Копирование данных на устройство
    res = cuMemcpyHtoD(d_A, h_A, N * N * sizeof(float));
    checkCUDAError(res, "cuMemcpyHtoD d_A");
    res = cuMemcpyHtoD(d_B, h_B, N * N * sizeof(float));
    checkCUDAError(res, "cuMemcpyHtoD d_B");

    // Загрузка PTX
    CUmodule module;
    res = cuModuleLoad(&module, "obj/matrix_mul.ptx");
    checkCUDAError(res, "cuModuleLoad");

    // Получение функции
    CUfunction kernel;
    res = cuModuleGetFunction(&kernel, module, "matrixMul");
    checkCUDAError(res, "cuModuleGetFunction");

    // Настройка запуска
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // Аргументы ядра (важно: передаем значения CUdeviceptr, а не их адреса!)
    void *args[] = {(void *)&d_A, (void *)&d_B, (void *)&d_C, (void *)&N};

    // Тайминг
    CUevent start, stop;
    cuEventCreate(&start, 0);
    cuEventCreate(&stop, 0);
    cuEventRecord(start, 0);

    // Запуск ядра
    res = cuLaunchKernel(kernel,
                         grid.x, grid.y, 1,
                         block.x, block.y, 1,
                         0, 0, args, 0);
    checkCUDAError(res, "cuLaunchKernel");

    // Синхронизация
    cuCtxSynchronize();
    cuEventRecord(stop, 0);
    cuEventSynchronize(stop);

    // Замер времени
    float ms;
    cuEventElapsedTime(&ms, start, stop);
    printf("CUDA Driver API (C/C++) Time: %.3f мс\n", ms);

    // Копирование результатов
    res = cuMemcpyDtoH(h_C, d_C, N * N * sizeof(float));
    checkCUDAError(res, "cuMemcpyDtoH");

    // Освобождение памяти
    cuMemFree(d_A);
    cuMemFree(d_B);
    cuMemFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;

    return 0;
}