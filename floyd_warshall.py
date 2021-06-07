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
import math

def print_matrix(dis, n):
    for i in range(n):
        for j in range(n):
            print(dis[i][j], end = " ")
        print()
    print()

ker = SourceModule("""
    #include <stdio.h>
    #define MIN(x, y) (((x) < (y)) ? (x) : (y))
    __global__ void calculate_kernel(int *dis, int n, int k){
        int i = threadIdx.x;
        for(int j = 0; j < n; j++){
            dis[i * n + j] = MIN(dis[i * n + j], dis[i * n + k] + dis[k * n + j]);
        }
    }
    """)

calculate_kernel = ker.get_function("calculate_kernel")

def floyd_warshall_sequential(dis, n):
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dis[i][j] = min(dis[i][j], dis[i][k] + dis[k][j])

def floyd_warshall_parallel(dis, n):
    dis_temp = np.array([0 for _ in range(n*n)], dtype=np.int32)
    for i in range(n):
        for j in range(n):
            dis_temp[i*n + j] = dis[i][j]
    for k in range(n):
        calculate_kernel(drv.InOut(dis_temp), np.int32(n), np.int32(k), block = (n if n < 1024 else 1024, 1, 1), grid = (int(math.ceil(n/1024)), 1, 1))
    k = 0
    for i in range(n):
        for j in range(n):
            dis[i][j] = dis_temp[i*n + j]
    
    

def generate_random_graph(n):
    dis = np.array([[0 for _ in range(n)] for _ in range(n)], dtype=np.int32)
    for i in range(n):
        for j in range(n):
            if(i == j):
                dis[i][j] = np.int32(0)
            else:
                dis[i][j] = np.int32(random.randint(1, 11))
    return dis

def generate_static_graph():
    dis = [[0, 3, 100, 7],[8, 0, 2, 100], [5, 100, 0, 1], [2, 100, 100, 0]]
    return dis

def multiple_examples_running_time():
    N = 1024
    for n in range(50, N + 1, 50):
        sum_time = 0
        num_iter = 5
        dis = generate_random_graph(n)
        running_times = list()
        for _ in range(num_iter):
            start = default_timer()
            floyd_warshall_parallel(dis, n)
            end = default_timer()
            elapsed = float(end - start)
            running_times.append(elapsed)
            sum_time += elapsed
        print(f'{n} {sum_time/num_iter} {np.std(running_times)}')

def individual_example_running_time():
    n = 500
    sum_time = 0
    num_iter = 1
    dis = generate_random_graph(n)
    running_times = list()
    for _ in range(num_iter):
        start = time.time()
        floyd_warshall_sequential(dis, n)
        end = time.time()
        elapsed = end - start
        running_times.append(elapsed)
        sum_time += elapsed
    print(f'{n} {sum_time/num_iter} {np.std(running_times)}')

def correctness_test():
    n = 4
    dis = generate_random_graph(n)
    dis_copy = np.array(dis)
    print_matrix(dis, n)

    floyd_warshall_sequential(dis, n)
    print_matrix(dis, n)
    dis = np.array(dis_copy)
    floyd_warshall_parallel(dis, n)
    print_matrix(dis, n)

def main():
    correctness_test()
    return 0

if __name__ == '__main__':
    main()