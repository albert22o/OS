from ctypes import *
import sys

if 'linux' in sys.platform:
    cuda = CDLL('libcuda.so')
elif 'win' in sys.platform:
    cuda = CDLL('nvcuda.dll')

CUDA_SUCCESS = 0
CUDA_ERRORS = {
    0: "CUDA_SUCCESS",
    1: "CUDA_ERROR_INVALID_VALUE",
    100: "CUDA_ERROR_INVALID_HANDLE", 
    200: "CUDA_ERROR_INVALID_IMAGE",
    201: "CUDA_ERROR_INVALID_CONTEXT",
    400: "CUDA_ERROR_INVALID_HANDLE"
}

cuInit = cuda.cuInit
cuInit.argtypes = [c_uint]
cuInit.restype = int

cuDeviceGetCount = cuda.cuDeviceGetCount
cuDeviceGetCount.argtypes = [POINTER(c_int)]
cuDeviceGetCount.restype = int

cuDeviceGet = cuda.cuDeviceGet
cuDeviceGet.argtypes = [POINTER(c_int), c_int]
cuDeviceGet.restype = int

cuCtxCreate = cuda.cuCtxCreate
cuCtxCreate.argtypes = [c_void_p, c_uint, c_int]
cuCtxCreate.restype = int

cuModuleLoad = cuda.cuModuleLoad
cuModuleLoad.argtypes = [c_void_p, c_char_p]
cuModuleLoad.restype  = int

cuCtxSynchronize = cuda.cuCtxSynchronize
cuCtxSynchronize.argtypes = []
cuCtxSynchronize.restype = int

cuModuleGetFunction = cuda.cuModuleGetFunction
cuModuleGetFunction.argtypes = [c_void_p, c_void_p, c_char_p ]
cuModuleGetFunction.restype = int

cuMemAlloc = cuda.cuMemAlloc
cuMemAlloc.argtypes = [c_void_p, c_size_t]
cuMemAlloc.restype = int

cuMemcpyHtoD = cuda.cuMemcpyHtoD 
cuMemcpyHtoD.argtypes = [c_void_p, c_void_p, c_size_t]
cuMemAlloc.restype = int

cuMemcpyDtoH = cuda.cuMemcpyDtoH 
cuMemcpyDtoH.argtypes = [c_void_p, c_void_p, c_size_t]
cuMemcpyDtoH.restype = int

cuMemFree = cuda.cuMemFree
cuMemFree.argtypes = [c_void_p] 
cuMemFree.restype = int

cuLaunchKernel = cuda.cuLaunchKernel
cuLaunchKernel.argtypes = [c_void_p, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_void_p, c_void_p, c_void_p]
cuLaunchKernel.restype = int

cuCtxDestroy = cuda.cuCtxDestroy
cuCtxDestroy.argtypes = [c_void_p]
cuCtxDestroy.restype = int

# Добавленные функции
cuEventCreate = cuda.cuEventCreate
cuEventCreate.argtypes = [POINTER(c_void_p), c_uint]
cuEventCreate.restype = int

cuEventRecord = cuda.cuEventRecord
cuEventRecord.argtypes = [c_void_p, c_void_p]
cuEventRecord.restype = int

cuEventSynchronize = cuda.cuEventSynchronize
cuEventSynchronize.argtypes = [c_void_p]
cuEventSynchronize.restype = int

cuEventElapsedTime = cuda.cuEventElapsedTime
cuEventElapsedTime.argtypes = [POINTER(c_float), c_void_p, c_void_p]
cuEventElapsedTime.restype = int

cuGetErrorString = cuda.cuGetErrorString
cuGetErrorString.argtypes = [c_int, POINTER(c_char_p)]
cuGetErrorString.restype = int
