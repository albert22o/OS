#include <iostream>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

// === Константная память ===
__constant__ float d_center[3]; // x0, y0, z0
__constant__ float d_radius;    // R

// === Текстурная память ===
texture<float, 3, cudaReadModeElementType> tex3DRef;

// === Ядро CUDA ===
__global__ void integrateOnSphereKernel(float *partialSums, int thetaSteps, int phiSteps)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPoints = thetaSteps * phiSteps;
    if (tid >= totalPoints)
        return;

    int i = tid / phiSteps;
    int j = tid % phiSteps;

    float theta = M_PI * (i + 0.5f) / thetaSteps;
    float phi = 2.0f * M_PI * (j + 0.5f) / phiSteps;

    float x = d_center[0] + d_radius * sinf(theta) * cosf(phi);
    float y = d_center[1] + d_radius * sinf(theta) * sinf(phi);
    float z = d_center[2] + d_radius * cosf(theta);

    float val = tex3D(tex3DRef, x, y, z);

    float dtheta = M_PI / thetaSteps;
    float dphi = 2.0f * M_PI / phiSteps;
    float dS = d_radius * d_radius * sinf(theta) * dtheta * dphi;

    partialSums[tid] = val * dS;
}

// === Функция интегрирования с замером времени ===
float integrateOnSphere(float *h_volume, int nx, int ny, int nz,
                        float x0, float y0, float z0, float R,
                        int thetaSteps, int phiSteps)
{
    // Подготовка 3D текстурного массива
    cudaExtent volumeSize = make_cudaExtent(nx, ny, nz);
    cudaArray *d_array;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaMalloc3DArray(&d_array, &desc, volumeSize);

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr(h_volume, nx * sizeof(float), nx, ny);
    copyParams.dstArray = d_array;
    copyParams.extent = volumeSize;
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);

    // Привязка текстуры
    tex3DRef.normalized = true;                 // используем реальные координаты
    tex3DRef.filterMode = cudaFilterModeLinear; // отключаем линейную интерполяцию
    tex3DRef.addressMode[0] = cudaAddressModeClamp;
    tex3DRef.addressMode[1] = cudaAddressModeClamp;
    tex3DRef.addressMode[2] = cudaAddressModeClamp;
    cudaBindTextureToArray(tex3DRef, d_array, desc);

    // Копирование параметров в константную память
    float h_center[3] = {x0, y0, z0};
    cudaMemcpyToSymbol(d_center, h_center, sizeof(float) * 3);
    cudaMemcpyToSymbol(d_radius, &R, sizeof(float));

    // Настройка ядра
    int totalPoints = thetaSteps * phiSteps;
    float *d_partialSums;
    cudaMalloc(&d_partialSums, sizeof(float) * totalPoints);

    int blockSize = 256;
    int gridSize = (totalPoints + blockSize - 1) / blockSize;

    // === Замер времени выполнения ядра ===
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    integrateOnSphereKernel<<<gridSize, blockSize>>>(d_partialSums, thetaSteps, phiSteps);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU kernel execution time: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Получение результата
    std::vector<float> h_partialSums(totalPoints);
    cudaMemcpy(h_partialSums.data(), d_partialSums, sizeof(float) * totalPoints, cudaMemcpyDeviceToHost);

    float result = 0.0f;
    for (float val : h_partialSums)
        result += val;

    // Очистка
    cudaUnbindTexture(tex3DRef);
    cudaFreeArray(d_array);
    cudaFree(d_partialSums);

    return result;
}

// === Создание объёма: f(x,y,z) = 1 ===
void generateConstantVolume(std::vector<float> &volume, int nx, int ny, int nz, float value)
{
    volume.resize(nx * ny * nz, value);
}

// === main ===
int main()
{
    const int nx = 64, ny = 64, nz = 64;

    std::vector<float> volume;
    generateConstantVolume(volume, nx, ny, nz, 1.0f); // f(x, y, z) = 1

    float x0 = nx / 2.0f;
    float y0 = ny / 2.0f;
    float z0 = nz / 2.0f;
    float R = 10.0f;

    int thetaSteps = 180;
    int phiSteps = 360;

    float integral = integrateOnSphere(volume.data(), nx, ny, nz, x0, y0, z0, R, thetaSteps, phiSteps);

    std::cout << "--- TEXTURE OPTIMIZED ---" << std::endl;
    std::cout << "Computed integral: " << integral << std::endl;
    std::cout << "Expected (4πR²): " << 4.0 * M_PI * R * R << std::endl;

    return 0;
}
