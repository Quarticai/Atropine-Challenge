import pandas as pd
import matplotlib.pyplot as plt


def show(
    path: str,
    iter: int
):
    df = pd.read_csv(path)
    df = df[df.iteration == iter]
    print(df)

    # plots
    plt.close("all")
    plt.figure(0)
    plt.plot(df.E, label='Real Output')
    plt.plot(df.ESS, linestyle="--", label='Steady State Output')
    plt.xlabel('Time [min]')
    plt.ylabel('E-Factor [A.U.]')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # create figure (fig), and array of axes (ax)
    fig, axs = plt.subplots(nrows=2, ncols=2)
    axs[0, 0].plot(df.U1, label='Real Input')
    axs[0, 0].plot(df.USS1, linestyle="--", label='Steady State Input')
    axs[0, 0].set_ylabel(u'U1 [\u03bcL/min]')
    axs[0, 0].set_xlabel('time [min]')
    axs[0, 0].grid()

    axs[0, 1].plot(df.U2, label='Real Input')
    axs[0, 1].plot(df.USS2, linestyle="--", label='Steady State Input')
    axs[0, 1].set_ylabel(u'U2 [\u03bcL/min]')
    axs[0, 1].set_xlabel('time [min]')
    axs[0, 1].grid()

    axs[1, 0].plot(df.U3, label='Real Input')
    axs[1, 0].plot(df.USS3, linestyle="--", label='Steady State Input')
    axs[1, 0].set_ylabel(u'U3 [\u03bcL/min]')
    axs[1, 0].set_xlabel('time [min]')
    axs[1, 0].grid()

    axs[1, 1].plot(df.U4, label='Real Input')
    axs[1, 1].plot(df.USS4, linestyle="--", label='Steady State Input')
    axs[1, 1].set_ylabel(u'U4 [\u03bcL/min]')
    axs[1, 1].set_xlabel('time [min]')
    axs[1, 1].legend()
    plt.tight_layout()
    plt.grid()
    plt.show()


path = './atropine.csv'

show(path, 988)
