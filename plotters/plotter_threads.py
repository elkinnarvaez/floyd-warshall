from sys import stdin
import matplotlib
import matplotlib.pyplot as plt

def main():
    f = open("../data/threads.txt", "r")
    lines = f.readlines()
    f.close()
    sequential_time = float(lines[0])
    x_axis = [2, 3, 4, 5, 6, 7, 8]
    y_axis = list()
    for i in range(1, len(lines)):
        y_axis.append(float(lines[i]))
    speed_up = [sequential_time/y_axis[i] for i in range(len(y_axis))]
    fig, ax = plt.subplots()
    ax.plot(x_axis, y_axis)
    # ax.set_title(f'Running time comparison\n')
    ax.set_xlabel(f'Number of threads')
    ax.set_ylabel(f'Time (seconds)')
    ax.grid(True)

    fig2, ax2 = plt.subplots()
    ax2.plot(x_axis, speed_up)
    # ax2.set_title(f'Speed up comparison\n')
    ax2.set_xlabel(f'Number of threads')
    ax2.set_ylabel(f'Speed up')
    ax2.grid(True)

    plt.show()
    return 0

if __name__ == '__main__':
    main()