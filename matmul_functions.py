import numpy as np
from numba import njit, cuda
import timeit


def matmul_transpose_trivial(X):
    result = np.zeros((len(X), len(X)), dtype=np.float64)
    for i in range(0, len(X)):
        for k in range(0, len(X[0])):
            for j in range(0, len(X)):
                result[i][j] += X[i][k] * X[j][k]
    return result

@njit
def matmul_transpose_numba(X):
    result = np.zeros((X.shape[0], X.shape[0]), dtype=np.float64)
    for i in range(0, X.shape[0]):
        for j in range(0, i+1):
            partial_sum = 0
            for k in range(0, X.shape[1]):
                partial_sum += X[i][k] * X[j][k]
            result[i][j] = partial_sum
            result[j][i] = partial_sum

    return result


def matmul_transpose_gpu(X):
    resultArray = np.zeros((X.shape[0], X.shape[0]), dtype=np.float64)
    X_gpu = cuda.to_device(X)
    resultArr_gpu = cuda.to_device(resultArray)
    matmul_kernel[1, 1024](X_gpu, resultArr_gpu)
    resultArray = resultArr_gpu.copy_to_host(resultArray)
    return resultArray

@cuda.jit
def matmul_kernel(A, C):
    i = cuda.threadIdx.x

    for threadNum in range(i, C.shape[0] * C.shape[0], 1024):
        row = threadNum // C.shape[1]
        col = threadNum % C.shape[1]
        arr = 0
        if row > col:
            continue

        for z in range(0, A.shape[1]):
            arr += A[row, z] * A[col, z]
        cuda.atomic.add(C[row], col, arr)
        cuda.atomic.add(C[col], row, arr)
    cuda.syncthreads()

#this is the comparison function - keep it as it is, don't change X or Y.
def matmul_comparison():
    X = np.random.randn(784, 128)
    Xt = X.copy().transpose()
    def timer(f, functionParameters):
        return min(timeit.Timer(lambda: f(X) if functionParameters == 1 else f(X,Xt)).repeat(3, 100))

    #print('Python:', timer(matmul_transpose_trivial, 1)) we will not consider this since it takes infinite time :)
    print('Numpy:', timer(np.matmul, 2))
    print('Numba:', timer(matmul_transpose_numba, 1))
    print('CUDA:', timer(matmul_transpose_gpu, 1))


if __name__ == '__main__':
    matmul_comparison()
