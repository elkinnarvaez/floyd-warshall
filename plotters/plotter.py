from sys import stdin
import matplotlib
import matplotlib.pyplot as plt

def main():
    f = open("../data/running_time_values_sequential.txt", "r")
    lines1 = f.readlines()
    f.close()
    f = open("../data/running_time_values_parallel_2_threads.txt", "r")
    lines2 = f.readlines()
    f.close()
    f = open("../data/running_time_values_parallel_4_threads.txt", "r")
    lines3 = f.readlines()
    f.close()
    f = open("../data/running_time_values_parallel_6_threads.txt", "r")
    lines4 = f.readlines()
    f.close()
    f = open("../data/running_time_values_parallel_8_threads.txt", "r")
    lines5 = f.readlines()
    f.close()
    x_axis1 = list(); x_axis2 = list(); x_axis3 = list(); x_axis4 = list(); x_axis5 = list()
    y_axis1 = list(); y_axis2 = list(); y_axis3 = list(); y_axis4 = list(); y_axis5 = list()
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
    for line in lines3:
        if len(line) > 0:
            l = line.strip().split()
            n = int(l[0])
            t = float(l[1])
            x_axis3.append(n)
            y_axis3.append(t)
    for line in lines4:
        if len(line) > 0:
            l = line.strip().split()
            n = int(l[0])
            t = float(l[1])
            x_axis4.append(n)
            y_axis4.append(t)
    for line in lines5:
        if len(line) > 0:
            l = line.strip().split()
            n = int(l[0])
            t = float(l[1])
            x_axis5.append(n)
            y_axis5.append(t)
    speed_up2 = [y_axis1[i]/y_axis2[i] for i in range(len(y_axis1))]
    speed_up3 = [y_axis1[i]/y_axis3[i] for i in range(len(y_axis1))]
    speed_up4 = [y_axis1[i]/y_axis4[i] for i in range(len(y_axis1))]
    speed_up5 = [y_axis1[i]/y_axis5[i] for i in range(len(y_axis1))]
    fig, ax = plt.subplots()
    ax.plot(x_axis1, y_axis1, label='Sequential')
    ax.plot(x_axis2, y_axis2, label='Parallel (num threads = 2)')
    ax.plot(x_axis3, y_axis3, label='Parallel (num threads = 4)')
    ax.plot(x_axis4, y_axis4, label='Parallel (num threads = 6)')
    ax.plot(x_axis5, y_axis5, label='Parallel (num threads = 8)')
    # ax.set_title(f'Running time comparison\n')
    ax.set_xlabel(f'N')
    ax.set_ylabel(f'Time (seconds)')
    ax.legend()
    ax.grid(True)

    fig2, ax2 = plt.subplots()
    ax2.plot(x_axis1, speed_up2, label='Number of threads = 2')
    ax2.plot(x_axis1, speed_up3, label='Number of threads = 4')
    ax2.plot(x_axis1, speed_up4, label='Number of threads = 6')
    ax2.plot(x_axis1, speed_up5, label='Number of threads = 8')
    # ax2.set_title(f'Speed up comparison\n')
    ax2.set_xlabel(f'N')
    ax2.set_ylabel(f'Speed up')
    ax2.legend()
    ax2.grid(True)

    plt.show()
    return 0

if __name__ == '__main__':
    main()