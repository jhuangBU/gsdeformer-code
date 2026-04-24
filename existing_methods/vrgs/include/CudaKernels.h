#pragma once

#include <string>
#include <cuda_runtime.h>
#include <exception>
#include <iomanip>
#include <stdexcept>

#define DEBUG 1

inline void cuda_error_throw() {
#if DEBUG || _DEBUG
	if (cudaPeekAtLastError() != cudaSuccess) {
		throw std::runtime_error(cudaGetErrorString(cudaGetLastError()));
	}
	cudaDeviceSynchronize();
	if (cudaPeekAtLastError() != cudaSuccess) {
		throw std::runtime_error(cudaGetErrorString(cudaGetLastError()));
	}
#endif
}

inline bool cuda_error(std::string& _message) {
#if DEBUG || _DEBUG
	if (cudaPeekAtLastError() != cudaSuccess) {
		_message.assign(cudaGetErrorString(cudaGetLastError()));
		return true;
	}
	cudaDeviceSynchronize();
	if (cudaPeekAtLastError() != cudaSuccess) {
		_message.assign(cudaGetErrorString(cudaGetLastError()));
		return true;
	}
#endif
	return false;
}

#define CUDA_SAFE_CALL_ALWAYS(A) A; cuda_error_throw();
#define CUDA_SAFE_CALL(A) A; cuda_error_throw();

template <class T, class DT>
inline void reset_and_copy(T *&device_ptr, size_t base_size, int element_shape, const DT *host_ptr) {
	if (device_ptr != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)device_ptr)); }
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&device_ptr, base_size * element_shape));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(device_ptr, host_ptr, base_size * element_shape, cudaMemcpyHostToDevice));
}

template <class T>
inline void reset_and_zero(T *&device_ptr, size_t base_size, int element_shape) {
	if (device_ptr != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)device_ptr)); }
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&device_ptr, base_size * element_shape));
	CUDA_SAFE_CALL_ALWAYS(cudaMemset(device_ptr, 0, base_size * element_shape));
}

void cuda_fill(int width, int height, float value, cudaSurfaceObject_t surface);
void cuda_splat_to_texture_single(int width, int height, float* data, cudaSurfaceObject_t surface, float NormFac = 1.0f);
void cuda_splat_to_texture(int width, int height, float* rgb, cudaSurfaceObject_t surface);

#ifdef __NVCC__
template<class T>
__global__ void clean_bit(int num, T *value, T bit) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	if (x < num) {
		value[x] &= ~bit;
	}
}
#endif

void save_image(float *deviceData, int width, int height, const char* filename);