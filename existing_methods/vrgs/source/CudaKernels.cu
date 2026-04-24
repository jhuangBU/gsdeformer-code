#include "CudaKernels.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void fill_kernel(int width, int height, float value, cudaSurfaceObject_t surface) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x < width && y < height) {
		float4 rgba;
		rgba.x = value * x / width * y / height;
		rgba.y = value;
		rgba.z = value * (width - x) / width * (height - y) / height;
		rgba.w = value;
		surf2Dwrite(rgba, surface, (int)sizeof(float4)*x, y, cudaBoundaryModeClamp);
	}
}

__global__ void splat_to_texture_single_kernel(int width, int height, float* InData, cudaSurfaceObject_t surface, float NormFac) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x < width && y < height) {

		//Flip y
		uint32_t y_flip = height - 1 - y;

		float4 rgba;
		rgba.x = InData[(y_flip * width + x)] / NormFac;
		rgba.y = InData[(y_flip * width + x)] / NormFac;
		rgba.z = InData[(y_flip * width + x)] / NormFac;
		rgba.w = 1;
		
		surf2Dwrite(rgba, surface, (int)sizeof(float4) * x, y, cudaBoundaryModeClamp);
	}
}

__global__ void splat_to_texture_kernel(int width, int height, float* InData, cudaSurfaceObject_t surface) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x < width && y < height) {

		//Flip y
		uint32_t y_flip = height - 1 - y;

		float4 rgba;
		rgba.x = InData[0 * width * height + (y_flip * width + x)];
		rgba.y = InData[1 * width * height + (y_flip * width + x)];
		rgba.z = InData[2 * width * height + (y_flip * width + x)];
		rgba.w = 1;
		
		surf2Dwrite(rgba, surface, (int)sizeof(float4) * x, y, cudaBoundaryModeClamp);
	}
}

template <typename T> T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

void cuda_fill(int width, int height, float value, cudaSurfaceObject_t surface) {
	const dim3 threads = { 16, 16, 1 };
	const dim3 blocks = { div_round_up<uint32_t>((uint32_t)width, threads.x), div_round_up<uint32_t>((uint32_t)height, threads.y), 1 };
	fill_kernel<<<blocks, threads>>> (width, height, value, surface);
}

void cuda_splat_to_texture_single(int width, int height, float* data, cudaSurfaceObject_t surface, float NormFac) {
	const dim3 threads = { 16, 16, 1 };
	const dim3 blocks = { div_round_up((uint32_t)width, threads.x), div_round_up((uint32_t)height, threads.y), 1 };
	splat_to_texture_single_kernel<<<blocks, threads>>>(width, height, data, surface, NormFac);
}

void cuda_splat_to_texture(int width, int height, float* rgb, cudaSurfaceObject_t surface) {
	const dim3 threads = { 16, 16, 1 };
	const dim3 blocks = { div_round_up((uint32_t)width, threads.x), div_round_up((uint32_t)height, threads.y), 1 };
	splat_to_texture_kernel<<<blocks, threads>>>(width, height, rgb, surface);
}

__global__ void convert_to_rgba(float *deviceData, unsigned char *hostData, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        int baseIndex = idx + width * height; // Index for the start of each channel

        float r = deviceData[0 * width * height + (y * width + x)]; // Red channel
        float g = deviceData[1 * width * height + (y * width + x)]; // Green channel
        float b = deviceData[2 * width * height + (y * width + x)]; // Blue channel

        hostData[4 * idx] = static_cast<unsigned char>(min(r, 1.0f) * 255.0f); // R
        hostData[4 * idx + 1] = static_cast<unsigned char>(min(g, 1.f) * 255.0f); // G
        hostData[4 * idx + 2] = static_cast<unsigned char>(min(b, 1.f) * 255.0f); // B
        hostData[4 * idx + 3] = 255; // Alpha
    }
}

void save_image(float *deviceData, int width, int height, const char* filename) {
    unsigned char *hostData = new unsigned char[width * height * 4];
	unsigned char *hostData_d = nullptr;
	reset_and_zero(hostData_d, sizeof(unsigned char), width * height * 4);
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    convert_to_rgba<<<gridSize, blockSize>>>(deviceData, hostData_d, width, height);
    cudaDeviceSynchronize();
	cudaMemcpy(hostData, hostData_d, sizeof(unsigned char) * width * height * 4, cudaMemcpyDeviceToHost);
    stbi_write_png(filename, width, height, 4, hostData, width * 4);
    delete[] hostData;
	cudaFree(hostData_d);
	hostData_d = nullptr;
}