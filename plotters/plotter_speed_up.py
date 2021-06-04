from sys import stdin
import matplotlib
import matplotlib.pyplot as plt

def main():
    f1 = open("data/running_time_sequential.txt", "r")
    f2 = open("data/running_time_parallel.txt", "r")
    lines1 = f1.readlines()
    lines2 = f2.readlines()
    f1.close()
    f2.close()
    x_axis1, x_axis2 = list(), list()
    y_axis1, y_axis2 = list(), list()
    for line in lines1:
        if len(line) > 0:
            l = line.strip().split()
            p = int(l[0])
            t = float(l[1])
            x_axis1.append(p)
            y_axis1.append(t)
    for line in lines2:
        if len(line) > 0:
            l = line.strip().split()
            p = int(l[0])
            t = float(l[1])
            x_axis2.append(p)
            y_axis2.append(t)
    efficiency = [(y_axis1[i]/y_axis2[i]) for i in range(len(y_axis1))]
    fig1, ax1 = plt.subplots()
    ax1.plot(x_axis1, efficiency, c='b')
    ax1.scatter(x_axis1, efficiency, c='b')
    ax1.set_title(f'Speed up')
    ax1.set_xlabel(f'N')
    ax1.set_ylabel(f'Speed up')
    ax1.grid(True)
    ax1.legend()

    plt.show()

    return 0

if __name__ == '__main__':
    main()