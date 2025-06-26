import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import pandas as pd


fig, axs = plt.subplots(3, 2)


def consumer(i):
    try:
        data = pd.read_csv("output.csv")

        axs[0, 0].cla()
        axs[0, 0].plot(data.time, data.rr_interval)
        axs[0, 0].set_title('RR Interval')

        axs[0, 1].cla()
        axs[0, 1].plot(data.time, data.hr)
        axs[0, 1].set_title('Heart Rate')

        axs[1, 0].cla()
        axs[1, 0].plot(data.time, data.lf)
        axs[1, 0].set_title('LF')

        axs[1, 1].cla()
        axs[1, 1].plot(data.time, data.hf)
        axs[1, 1].set_title('HF')

        axs[2, 0].cla()
        axs[2, 0].plot(data.time, data.rmssd)
        axs[2, 0].set_title('RMSSD')

        axs[2, 1].cla()
        axs[2, 1].plot(data.time, data.sdnn)
        axs[2, 1].set_title('SDNN')


    except Exception as e:
        print(str(e))


def main():
    animation = FuncAnimation(fig, consumer, interval=1000)
    plt.show()


if __name__ == '__main__':
    main()
