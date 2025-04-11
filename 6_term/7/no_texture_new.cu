#include <iostream>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

// =========== Настройки интерполяции ===========
enum InterpMode
{
    NEAREST = 0,
    LINEAR = 1
};

// =========== Ядро интегрирования ===========
__global__ void integrateOnSphereKernel(
    const float *volume, int nx, int ny, int nz,
    float *partialSums,
    float x0, float y0, float z0, float R,
    int thetaSteps, int phiSteps,
    InterpMode interpMode)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPoints = thetaSteps * phiSteps;
    if (tid >= totalPoints)
        return;

    int i = tid / phiSteps;
    int j = tid % phiSteps;

    float theta = M_PI * (i + 0.5f) / thetaSteps;
    float phi = 2.0f * M_PI * (j + 0.5f) / phiSteps;

    float x = x0 + R * sinf(theta) * cosf(phi);
    float y = y0 + R * sinf(theta) * sinf(phi);
    float z = z0 + R * cosf(theta);

    float val = 0.0f;

    if (interpMode == NEAREST)
    {
        int xi = __float2int_rn(x);
        int yi = __float2int_rn(y);
        int zi = __float2int_rn(z);

        if (xi >= 0 && xi < nx &&
            yi >= 0 && yi < ny &&
            zi >= 0 && zi < nz)
        {
            int idx = zi * nx * ny + yi * nx + xi;
            val = volume[idx];
        }
    }
    else if (interpMode == LINEAR)
    {
        int x0i = floorf(x), x1i = x0i + 1;
        int y0i = floorf(y), y1i = y0i + 1;
        int z0i = floorf(z), z1i = z0i + 1;

        float xd = x - x0i;
        float yd = y - y0i;
        float zd = z - z0i;

        auto at = [&](int xi, int yi, int zi) -> float
        {
            if (xi < 0 || xi >= nx ||
                yi < 0 || yi >= ny ||
                zi < 0 || zi >= nz)
                return 0.0f;
            return volume[zi * nx * ny + yi * nx + xi];
        };

        float c000 = at(x0i, y0i, z0i);
        float c001 = at(x0i, y0i, z1i);
        float c010 = at(x0i, y1i, z0i);
        float c011 = at(x0i, y1i, z1i);
        float c100 = at(x1i, y0i, z0i);
        float c101 = at(x1i, y0i, z1i);
        float c110 = at(x1i, y1i, z0i);
        float c111 = at(x1i, y1i, z1i);

        float c00 = c000 * (1 - xd) + c100 * xd;
        float c01 = c001 * (1 - xd) + c101 * xd;
        float c10 = c010 * (1 - xd) + c110 * xd;
        float c11 = c011 * (1 - xd) + c111 * xd;

        float c0 = c00 * (1 - yd) + c10 * yd;
        float c1 = c01 * (1 - yd) + c11 * yd;

        val = c0 * (1 - zd) + c1 * zd;
    }

    float dtheta = M_PI / thetaSteps;
    float dphi = 2.0f * M_PI / phiSteps;
    float dS = R * R * sinf(theta) * dtheta * dphi;

    partialSums[tid] = val * dS;
}

// =========== Хост-функция ===========
float integrateOnSphere(
    const std::vector<float> &h_volume, int nx, int ny, int nz,
    float x0, float y0, float z0, float R,
    int thetaSteps, int phiSteps,
    InterpMode interpMode)
{
    int N = nx * ny * nz;
    float *d_volume;
    cudaMalloc(&d_volume, sizeof(float) * N);
    cudaMemcpy(d_volume, h_volume.data(), sizeof(float) * N, cudaMemcpyHostToDevice);

    int totalPoints = thetaSteps * phiSteps;
    float *d_partialSums;
    cudaMalloc(&d_partialSums, sizeof(float) * totalPoints);

    int blockSize = 256;
    int gridSize = (totalPoints + blockSize - 1) / blockSize;

    integrateOnSphereKernel<<<gridSize, blockSize>>>(
        d_volume, nx, ny, nz,
        d_partialSums,
        x0, y0, z0, R,
        thetaSteps, phiSteps,
        interpMode);
    cudaDeviceSynchronize();

    std::vector<float> h_partialSums(totalPoints);
    cudaMemcpy(h_partialSums.data(), d_partialSums, sizeof(float) * totalPoints, cudaMemcpyDeviceToHost);

    float result = 0.0f;
    for (float val : h_partialSums)
        result += val;

    cudaFree(d_volume);
    cudaFree(d_partialSums);
    return result;
}

// =========== Создание объема ===========
void generateConstantVolume(std::vector<float> &volume, int nx, int ny, int nz, float value)
{
    volume.resize(nx * ny * nz, value);
}

// =========== main ===========
int main()
{
    const int nx = 64, ny = 64, nz = 64;
    std::vector<float> volume;
    generateConstantVolume(volume, nx, ny, nz, 1.0f); // f(x,y,z) = 1

    float x0 = nx / 2.0f;
    float y0 = ny / 2.0f;
    float z0 = nz / 2.0f;
    float R = 10.0f;

    int thetaSteps = 180;
    int phiSteps = 360;

    std::cout << "--- NEAREST ---\n";
    float integral_nearest = integrateOnSphere(volume, nx, ny, nz, x0, y0, z0, R, thetaSteps, phiSteps, NEAREST);
    std::cout << "Computed integral: " << integral_nearest << std::endl;

    std::cout << "--- LINEAR ---\n";
    float integral_linear = integrateOnSphere(volume, nx, ny, nz, x0, y0, z0, R, thetaSteps, phiSteps, LINEAR);
    std::cout << "Computed integral: " << integral_linear << std::endl;

    std::cout << "Expected (4πR²): " << 4.0 * M_PI * R * R << std::endl;
    return 0;
}
