from sys import stdin
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def main():
    f = open("data/ram_usage.txt", "r")
    lines = f.readlines()
    f.close()
    x_axis = list()
    y_axis_time = list()
    y_axis_std = list()
    for line in lines:
        if len(line) > 0:
            l = line.strip().split()
            n = int(l[0])
            t = float(l[1])
            x_axis.append(n)
            y_axis_time.append(t)
            
    fig, ax = plt.subplots()
    ax.plot(x_axis, y_axis_time, c='r')
    ax.scatter(x_axis, y_axis_time, c='r', s=25)
    ax.set_title(f'VRAM usage')
    ax.set_xlabel(f'N')
    ax.set_ylabel(f'Usage (%)')
    ax.grid(True)


    plt.show()
    return 0

if __name__ == '__main__':
    main()