from sys import stdin
import matplotlib
import matplotlib.pyplot as plt

def main():
    f = open("../data/running_time_values_sequential.txt", "r")
    lines1 = f.readlines()
    f.close()
    f = open("../data/running_time_values_parallel_8_threads.txt", "r")
    lines2 = f.readlines()
    f.close()
    x_axis1 = list(); x_axis2 = list()
    y_axis1 = list(); y_axis2 = list()
    for line in lines1:
        if len(line) > 0:
            l = line.strip().split()
            n = int(l[0])
            t = float(l[1])
            x_axis1.append(n)
            y_axis1.append(t)
    for line in lines2:
        if len(line) > 0:
            l = line.strip().split()
            n = int(l[0])
            t = float(l[1])
            x_axis2.append(n)
            y_axis2.append(t)
    speed_up = [y_axis1[i]/y_axis2[i] for i in range(len(y_axis1))]
    fig, ax = plt.subplots()
    ax.plot(x_axis1, y_axis1, label='Sequential')
    ax.plot(x_axis2, y_axis2, label='Parallel')
    # ax.set_title(f'Running time as number of nodes increase. Number of threads = 8\n')
    ax.set_xlabel(f'N')
    ax.set_ylabel(f'Time (seconds)')
    ax.legend()
    ax.grid(True)

    fig2, ax2 = plt.subplots()
    ax2.plot(x_axis1, speed_up, c='r')
    # ax2.set_title(f'Speed up. Number of threads = 8\n')
    ax2.set_xlabel(f'N')
    ax2.set_ylabel(f'Parallel time / Sequential time')
    ax2.grid(True)

    plt.show()
    return 0

if __name__ == '__main__':
    main()