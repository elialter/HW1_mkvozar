import numpy as np
from numba import cuda, njit, prange, float32
import timeit


def dist_cpu(A, B, p):
    """
     Returns
     -------
     np.array
         p-dist between A and B
     """
    sum = 0
    for i in range(0,1000):
        for j in range(0, 1000):
            sum += pow(abs(A[i][j] - B[i][j]), p)
    return pow(sum, 1/(float(p)))



@njit(parallel=True)
def dist_numba(A, B, p):
    """
     Returns
     -------
     np.array
         p-dist between A and B
     """
    sum = 0.0
    for i in prange(1000):
        for j in prange(1000):
            tmp = A[i][j] - B[i][j]
            if tmp > 0:
                sum += tmp**p
            else:
                sum += (tmp * -1) ** p
    return sum**(1 / p)

def dist_gpu(A, B, p):
    C = np.zeros(1, np.float64)

    gpuA = cuda.to_device(A)
    gpuB = cuda.to_device(B)
    gpuC = cuda.to_device(C)

    dist_kernel[1000, 1000](gpuA, gpuB, p, gpuC)
    C = gpuC.copy_to_host(C)
    return C[0] ** (1.0 / p)


@cuda.jit
def dist_kernel(A, B, p, C):
    i = cuda.threadIdx.x
    j = cuda.blockIdx.x
    s_arr = cuda.shared.array(1, float32)

    if i == 0:
        s_arr[0] = 0.0
    cuda.syncthreads()

    if i < 1000 and j < 1000:
        thread_sum = abs(A[i][j] - B[i][j]) ** p
        cuda.atomic.add(s_arr, 0, thread_sum)
    cuda.syncthreads()

    if i == 0:
        cuda.atomic.add(C, 0, s_arr[0])

   
#this is the comparison function - keep it as it is.
def dist_comparison():
    A = np.random.randint(0,256,(1000, 1000))
    B = np.random.randint(0,256,(1000, 1000))
    p = [1, 2]

    def timer(f, q):
        return min(timeit.Timer(lambda: f(A, B, q)).repeat(3, 20))


    for power in p:
        print('p=' + str(power))
        print('     [*] CPU:', timer(dist_cpu,power))
        print('     [*] Numba:', timer(dist_numba,power))
        print('     [*] CUDA:', timer(dist_gpu, power))

if __name__ == '__main__':
    dist_comparison()
