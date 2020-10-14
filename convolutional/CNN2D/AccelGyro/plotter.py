import matplotlib.pyplot as plt
import pandas as pd


def hist_plotter(hist_dict):
    hist_df = pd.DataFrame.from_dict(hist_dict)
    plt.plot(hist_df.index, hist_df.val_loss, color="blue")
    plt.plot(hist_df.index, hist_df.loss, color="orange")
    plt.twinx()
    try:
        plt.scatter(hist_df.index, hist_df.lr, color="green")
    except:
        pass
    plt.show()
