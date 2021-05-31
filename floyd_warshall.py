from sys import stdin


def floyd_warshall_sequential(dis, n):
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dis[i][j] = min(dis[i][j], dis[i][k] + dis[k][j])

def generate_randon_graph(n):
    dis = [[None for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if(i == j):
                dis[i][j] = 0
            else:
                pass
    return dis
def multiple_examples_running_time():
    for n in range(3, 5000, 500):
        sum_time = 0
        num_iter = 5
        dis = generate_randon_graph(n)
        for _ in range(num_iter):
            start, stop = None, None
            floyd_warshall_sequential(dis)
            
def main():
    return 0

if __name__ == '__main__':
    main()