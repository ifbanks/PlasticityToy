import matplotlib.pyplot as plt
import csv
import h5py
import numpy as np
import pandas as pd

from bmtk.analyzer.spike_trains import to_dataframe

n_bins = 40




def plot_spikes():
    print("what node u want")
    select = input()
    df = to_dataframe(config_file='simulation_config.json') # best way to get spikes into data frame
    #df.set_option("display.max_rows", None, "display.max_columns", None)

    df0 = df.loc[(df['node_ids'] == int(select)) & (df.timestamps <= 400)] # this is the node ID you want
    # print(df0)
    # print(df0)
    x = df0['timestamps'].tolist()
    plt.hist(x=x) #hist for the whole sim will need to generate for different parts of sim
    plt.show()

def plot_9_spikes_400():
    # h = h5py.File('output/spikes.h5', 'r')
    # timestamps = h['spikes']['biophysical']['timestamps'][:]
    # node_ids = h['spikes']['biophysical']['node_ids'][:]
    # df = pd.DataFrame({'node_ids': node_ids, 'ts': timestamps})

    df = to_dataframe(config_file='simulation_config.json')

    fig, axs = plt.subplots(3, 3, sharey=True, tight_layout=True)
    fig.suptitle('Spike histogram for the first 400 ms')

    #node 0
    df0 = df.loc[(df['node_ids'] == 0) & (df.timestamps <= 400)]
    x = df0['timestamps'].tolist()
    axs[0, 0].hist(x=x, bins=n_bins)
    axs[0, 0].set_title('P1')
    axs[0, 0].set_xlim([0, 400])


    df0 = df.loc[(df['node_ids'] == 1) & (df.timestamps <= 400)]
    x = df0['timestamps'].tolist()
    axs[0, 1].hist(x=x, bins=n_bins)
    axs[0, 1].set_title('P2')
    axs[0, 1].set_xlim([0, 400])


    df0 = df.loc[(df['node_ids'] == 2) & (df.timestamps <= 400)]
    x = df0['timestamps'].tolist()
    axs[0, 2].hist(x=x, bins=n_bins)
    axs[0, 2].set_title('P3')
    axs[0, 2].set_xlim([0, 400])


    df0 = df.loc[(df['node_ids'] == 3) & (df.timestamps <= 400)]
    x = df0['timestamps'].tolist()
    axs[1, 0].hist(x=x, bins=n_bins)
    axs[1, 0].set_title('P4')
    axs[1, 0].set_xlim([0, 400])


    df0 = df.loc[(df['node_ids'] == 4) & (df.timestamps <= 400)]
    x = df0['timestamps'].tolist()
    axs[1, 1].hist(x=x, bins=n_bins)
    axs[1, 1].set_title('P5')
    axs[1, 1].set_xlim([0, 400])


    df0 = df.loc[(df['node_ids'] == 5) & (df.timestamps <= 400)]
    x = df0['timestamps'].tolist()
    axs[1, 2].hist(x=x, bins=n_bins)
    axs[1, 2].set_title('P6')
    axs[1, 2].set_xlim([0, 400])


    df0 = df.loc[(df['node_ids'] == 6) & (df.timestamps <= 400)]
    x = df0['timestamps'].tolist()
    axs[2, 0].hist(x=x, bins=n_bins)
    axs[2, 0].set_title('P7')
    axs[2, 0].set_xlim([0, 400])


    df0 = df.loc[(df['node_ids'] == 7) & (df.timestamps <= 400)]
    x = df0['timestamps'].tolist()
    axs[2, 1].hist(x=x, bins=n_bins)
    axs[2, 1].set_title('P8')
    axs[2, 1].set_xlim([0, 400])


    df0 = df.loc[(df['node_ids'] == 8) & (df.timestamps <= 400)]
    x = df0['timestamps'].tolist()
    axs[2, 2].hist(x=x, bins=n_bins)
    axs[2, 2].set_title('I1')
    axs[2, 2].set_xlim([0, 400])
    # axs[0,0].plot(dh.loc[(dh.node_ids==9) & (dh.ts <= 400)])
    for ax in axs.flat:
        ax.set(xlabel='time(ms)', ylabel='# of spikes')
    plt.show()

    #TESTING EXTINCTION
    fig, axs = plt.subplots(3, 3, sharey=True, tight_layout=True)
    fig.suptitle('Spike histogram after waiting (extinction)')

    # node 0
    df0 = df.loc[(df['node_ids'] == 0) & (df.timestamps <= 40200)]
    x = df0['timestamps'].tolist()
    axs[0, 0].hist(x=x)
    axs[0, 0].set_title('P1')
    axs[0, 0].set_xlim([1200, 10000])

    df0 = df.loc[(df['node_ids'] == 1) & (df.timestamps <= 40200)]
    x = df0['timestamps'].tolist()
    axs[0, 1].hist(x=x)
    axs[0, 1].set_title('P2')
    axs[0, 1].set_xlim([1200, 10000])

    df0 = df.loc[(df['node_ids'] == 2) & (df.timestamps <= 40200)]
    x = df0['timestamps'].tolist()
    axs[0, 2].hist(x=x)
    axs[0, 2].set_title('P3')
    axs[0, 2].set_xlim([1200, 10000])

    df0 = df.loc[(df['node_ids'] == 3) & (df.timestamps <= 40200)]
    x = df0['timestamps'].tolist()
    axs[1, 0].hist(x=x)
    axs[1, 0].set_title('P4')
    axs[1, 0].set_xlim([1200, 10000])

    df0 = df.loc[(df['node_ids'] == 4) & (df.timestamps <= 40200)]
    x = df0['timestamps'].tolist()
    axs[1, 1].hist(x=x)
    axs[1, 1].set_title('P5')
    axs[1, 1].set_xlim([1200, 10000])

    df0 = df.loc[(df['node_ids'] == 5) & (df.timestamps <= 40200)]
    x = df0['timestamps'].tolist()
    axs[1, 2].hist(x=x)
    axs[1, 2].set_title('P6')
    axs[1, 2].set_xlim([1200, 10000])

    df0 = df.loc[(df['node_ids'] == 6) & (df.timestamps <= 40200)]
    x = df0['timestamps'].tolist()
    axs[2, 0].hist(x=x)
    axs[2, 0].set_title('P7')
    axs[2, 0].set_xlim([1200, 10000])

    df0 = df.loc[(df['node_ids'] == 7) & (df.timestamps <= 40200)]
    x = df0['timestamps'].tolist()
    axs[2, 1].hist(x=x)
    axs[2, 1].set_title('P8')
    axs[2, 1].set_xlim([1200, 10000])

    df0 = df.loc[(df['node_ids'] == 8) & (df.timestamps <= 40200)]
    x = df0['timestamps'].tolist()
    axs[2, 2].hist(x=x)
    axs[2, 2].set_title('I1')
    axs[2, 2].set_xlim([1200, 10000])
    # axs[0,0].plot(dh.loc[(dh.node_ids==9) & (dh.ts <= 400)])
    for ax in axs.flat:
        ax.set(xlabel='time(ms)', ylabel='# of spikes')
    plt.show()

#$plot_spikes()
plot_9_spikes_400()