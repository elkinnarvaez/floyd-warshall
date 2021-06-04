from sys import stdin
import random
import datetime
import time
from timeit import default_timer
import numpy as np
import pycuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as drv
from pycuda import gpuarray

n = None

def print_matrix(dis, n):
    for i in range(n):
        for j in range(n):
            print(dis[i][j], end = " ")
        print()
    print()

ker = SourceModule("""
    #include <stdio.h>
    #define MIN(x, y) (((x) < (y)) ? (x) : (y))
    __global__ void calculate_kernel(int *dis_gpu, int n, int k){
        int i = threadIdx.x, j;
        for(j = 0; j < n; j++){
            dis_gpu[i * n + j] = MIN(dis_gpu[i * n + j], dis_gpu[i * n + k] + dis_gpu[i * n + j]);
        }
    }
    """)

# ker = SourceModule("""
#     __global__ void calculate_kernel(int *dis, int k, int n){
#         int i = threadIdx.x, j;
#         for(j = 0; j < n; j++){
#             dis[i] = %(MAX_N)i;
#         }
#     }
#     """%{'MAX_N': MAX_N})

calculate_kernel = ker.get_function("calculate_kernel")

def floyd_warshall_sequential(dis, n):
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dis[i][j] = min(dis[i][j], dis[i][k] + dis[k][j])

def floyd_warshall_parallel(dis, n):
    dis_gpu = gpuarray.to_gpu(dis)
    for k in range(n):
        calculate_kernel(dis_gpu, np.int32(n), np.int32(k), block = (n, 1, 1), grid = (1, 1, 1))
    dis = np.array(dis_gpu.get())
    

def generate_randon_graph(n):
    dis = np.array([[0 for _ in range(n)] for _ in range(n)], dtype='int')
    for i in range(n):
        for j in range(n):
            if(i == j):
                dis[i][j] = np.int64(0)
            else:
                dis[i][j] = np.int64(random.randint(1, 11))
    return dis

def multiple_examples_running_time():
    N = 500
    for n in range(50, N + 1, 50):
        sum_time = 0
        num_iter = 5
        dis = generate_randon_graph(n)
        running_times = list()
        for _ in range(num_iter):
            start = default_timer()
            floyd_warshall_sequential(dis, n)
            end = default_timer()
            elapsed = float(end - start)
            running_times.append(elapsed)
            sum_time += elapsed
        print(f'{n} {sum_time/num_iter} {num_iter}', end="")
        for t in running_times:
            print(f" {t}", end = "")
        print()

def individual_example_running_time():
    n = 500
    sum_time = 0
    num_iter = 5
    dis = generate_randon_graph(n)
    running_times = list()
    for _ in range(num_iter):
        start = time.time()
        floyd_warshall_sequential(dis, n)
        end = time.time()
        elapsed = end - start
        running_times.append(elapsed)
        sum_time += elapsed
    print(f'{n} {sum_time/num_iter} {num_iter}', end="")
    for t in running_times:
        print(f" {t}", end = "")
    print()

def correctness_test():
    n = 5
    dis = generate_randon_graph(n)
    dis_copy = np.array(dis)

    print_matrix(dis, n)
    floyd_warshall_sequential(dis, n)
    print_matrix(dis, n)
    dis = np.array(dis_copy)
    print_matrix(dis, n)
    floyd_warshall_parallel(dis, n)
    print_matrix(dis, n)
            
def main():
    correctness_test()
    return 0

if __name__ == '__main__':
    main()