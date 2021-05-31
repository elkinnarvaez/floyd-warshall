from sys import stdin
import random
import datetime
import time
from timeit import default_timer

dis = [[None for _ in range(1001)] for _ in range(1001)]

def floyd_warshall_sequential(n):
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dis[i][j] = min(dis[i][j], dis[i][k] + dis[k][j])

def generate_randon_graph(n):
    for i in range(n):
        for j in range(n):
            if(i == j):
                dis[i][j] = 0
            else:
                dis[i][j] = random.randint(1, 11)
    return dis

def multiple_examples_running_time():
    N = 500
    for n in range(50, N + 1, 50):
        sum_time = 0
        num_iter = 5
        generate_randon_graph(n)
        running_times = list()
        for _ in range(num_iter):
            start = default_timer()
            floyd_warshall_sequential(n)
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
    generate_randon_graph(n)
    running_times = list()
    for _ in range(num_iter):
        start = time.time()
        floyd_warshall_sequential(n)
        end = time.time()
        elapsed = end - start
        running_times.append(elapsed)
        sum_time += elapsed
    print(f'{n} {sum_time/num_iter} {num_iter}', end="")
    for t in running_times:
        print(f" {t}", end = "")
    print()
            
def main():
    individual_example_running_time()

if __name__ == '__main__':
    main()