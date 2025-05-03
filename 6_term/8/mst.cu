#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <cuda.h>  // Добавляем заголовок для CUDA Driver API

// Функция для загрузки PTX-файла в строку
char* loadPTXFile(const char* filePath, size_t* size) {
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        fprintf(stderr, "Не удалось открыть файл %s\n", filePath);
        return nullptr;
    }
    
    *size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    char* buffer = new char[*size + 1];
    file.read(buffer, *size);
    buffer[*size] = '\0';
    
    file.close();
    return buffer;
}

int main() {
    int N = 1024;
    float *a, *b;
    float *d_a, *d_b;

    // Выделяем память на CPU
    a = (float*)malloc(N * sizeof(float));
    b = (float*)malloc(N * sizeof(float));

    // Инициализируем данные
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Выделяем память на GPU
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));

    // Копируем данные на GPU
    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

    // Загружаем PTX-модуль
    size_t ptxSize;
    char* ptxSource = loadPTXFile("kernel.ptx", &ptxSize);
    if (!ptxSource) {
        return 1;
    }

    // Инициализируем CUDA Driver API
    CUresult result;
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;

    result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "Ошибка инициализации CUDA Driver API\n");
        delete[] ptxSource;
        return 1;
    }

    result = cuDeviceGet(&device, 0);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "Ошибка получения устройства CUDA\n");
        delete[] ptxSource;
        return 1;
    }

    result = cuCtxCreate(&context, 0, device);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "Ошибка создания контекста CUDA\n");
        delete[] ptxSource;
        return 1;
    }

    // Загружаем PTX-код
    result = cuModuleLoadDataEx(&module, ptxSource, 0, 0, 0);
    if (result != CUDA_SUCCESS) {
      const char* errorStr;
      cuGetErrorString(result, &errorStr);
      printf("Ошибка загрузки PTX: %s\n", errorStr);
      delete[] ptxSource;
      return 1;
  }

    // Получаем указатель на функцию ядра
    result = cuModuleGetFunction(&kernel, module, "add");
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "Ошибка получения функции ядра\n");
        delete[] ptxSource;
        cuModuleUnload(module);
        cuCtxDestroy(context);
        return 1;
    }

    // Запускаем ядро
    int blockSize = 128;
    int gridSize = (N + blockSize - 1) / blockSize;

    void* args[] = { &d_a, &d_b, &N };
    result = cuLaunchKernel(kernel,
                           gridSize, 1, 1,    // grid dim
                           blockSize, 1, 1,   // block dim
                           0, 0,              // shared mem and stream
                           args, 0);

    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "Ошибка запуска ядра\n");
        delete[] ptxSource;
        cuModuleUnload(module);
        cuCtxDestroy(context);
        return 1;
    }

    // Проверяем ошибки
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Ошибка ядра: %s\n", cudaGetErrorString(err));
        delete[] ptxSource;
        cuModuleUnload(module);
        cuCtxDestroy(context);
        return 1;
    }

    // Копируем результат обратно на CPU
    cudaMemcpy(a, d_a, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Выводим первые 5 элементов для проверки
    for (int i = 0; i < 5; i++) {
        printf("a[%d] = %f\n", i, a[i]);
    }

    // Освобождаем память
    delete[] ptxSource;
    cuModuleUnload(module);
    cuCtxDestroy(context);
    cudaFree(d_a);
    cudaFree(d_b);
    free(a);
    free(b);

    return 0;
}