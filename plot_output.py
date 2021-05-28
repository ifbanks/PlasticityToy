import numpy as np
import matplotlib.pyplot as plt
import h5py
from bmtk.analyzer.cell_vars import _get_cell_report, plot_report
import matplotlib.pyplot as plt
import csv

#PLOT OF ALL INPUTS
#AFTER SOME TIME TONE AND SHOCK COMES ON
#RANDOM INPUTS
#NEXT PANEL COULD BE TONE INPUT AND HTEN SHOCK INPUT
#RASTER PLOT OF WHEN CELLS SPIKE (ISABEL)

#PLOT CSVS OF TONE AND SHOCK INPUT 
from bmtk.analyzer.compartment import plot_traces
import pandas as pd
from bmtk.utils.reports.spike_trains.plotting import plot_raster
from bmtk.utils.reports.spike_trains.plotting import plot_rates
from bmtk.utils.reports.spike_trains.plotting import plot_rates_boxplot
from scipy.signal import find_peaks
from bmtk.utils.reports.spike_trains.spike_train_buffer import STCSVBuffer
import pdb
from bmtk.analyzer.spike_trains import to_dataframe

# Load data
config_file = "simulation_config.json"
raster_file = './output/spikes.h5'

mem_pot_file = './output/v_report.h5'
cai_file = './output/cai_report.h5'

shock_file = 'shock_spikes.csv'
tone_file = 'tone_spikes.csv'

# load 
f = h5py.File(mem_pot_file,'r')
g = h5py.File(cai_file,'r')

plot_raster(raster_file, with_histogram=True, node_groups=[{'node_ids' : range(0,9), 'c':'b', 'label': 'all'}])
plot_rates(raster_file, node_groups=[{'node_ids' : range(0,9), 'c':'b', 'label': 'all'}])
plot_rates_boxplot(raster_file, node_groups=[{'node_ids' : range(0,1), 'c':'b', 'label': 'all'}])

df = to_dataframe(config_file='simulation_config.json')
df0 = df.loc[df['node_ids'] == 0]
x = df0['timestamps'].tolist()
plt.hist(x=x)
plt.show()



shock_array = []
with open(shock_file, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        shock_array.append(row)

print(shock_array)
shock_x = []
shock_y = []
for element in shock_array:
    word = element[0].split('\'')
    #print(word[0])
    shock_x.append(int(word[0]))
    shock_y.append(int(word[2]))

tone_array = []
with open(tone_file, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        tone_array.append(row)

print(tone_array)
tone_x = []
tone_y = []
for element in tone_array:
    word = element[0].split('\'')
    #print(word[0])
    tone_x.append(int(word[0]))
    tone_y.append(int(word[2]))

plt.plot(tone_x, tone_y, '.', label="tone input")
plt.plot(shock_x, shock_y, '.', label="shock input")
plt.xlabel("Time elapsed")
plt.ylabel("Node id")
plt.legend()
plt.show()



mem_potential = f['report']['biophysical']['data']
plt.plot(np.arange(0,mem_potential.shape[0]/10,.1),mem_potential[:,0])
plt.text(200,-80,'tone')
plt.text(700,-80,'tone+shock')
plt.text(1600,-80,'tone+shock')
plt.text(2600,-80,'tone')
plt.xlabel('time (ms)')
plt.ylabel('membrane potential (mV)')



_ = plot_traces(config_file='simulation_config.json', node_ids=[0], report_name='v_report')
plot_two = plot_traces(config_file='simulation_config.json', node_ids=[0], report_name='cai_report')
#caiplt = g['report']['biophysical']['data']
#plt.plot(caiplt[:,0])

plt.show()

h = h5py.File('output\\spikes.h5', 'r')
timestamps = h['spikes']['biophysical']['timestamps'][:]
node_ids = h['spikes']['biophysical']['node_ids'][:]



dh = pd.DataFrame({'node_ids':node_ids, 'ts':timestamps})
print(dh)
#print(dh.head())
plt.plot(dh.ts, dh.node_ids, '.')
plt.xlabel("time(ms)")
plt.ylabel("node id")
plt.plot(tone_x, tone_y, '.', label="tone input")
plt.plot(shock_x, shock_y, '.', label="shock input")
plt.legend()
plt.show()

print(dh.loc[(dh.node_ids==0) & (dh.ts <= 400)].head())
to_slice_df = dh.loc[(dh.node_ids==0), 'ts']
fig, axs = plt.subplots(3,3, sharey=True, tight_layout=True)
#POLISH
#4/23/21
#Add titles, figure out orange bar (might be beccause of size of y)
#plot outputs to see if its actually displaying data correctly
#4/26/21
#create list out of data and then plot THAT rather than using the dataframe search values
#set_xlim(0,400) set x values
axs[0,0].hist(dh.loc[(dh.node_ids==0)])

axs[0,1].hist(dh.loc[(dh.node_ids==1)])
axs[0,2].hist(dh.loc[(dh.node_ids==2)])
axs[1,0].hist(dh.loc[(dh.node_ids==3)])
axs[1,1].hist(dh.loc[(dh.node_ids==4)])
axs[1,2].hist(dh.loc[(dh.node_ids==5)])
axs[2,0].hist(dh.loc[(dh.node_ids==6)])
axs[2,1].hist(dh.loc[(dh.node_ids==7)])
axs[2,2].hist(dh.loc[(dh.node_ids==8)])
plt.show()

fig, axs = plt.subplots(3,3, sharey=True, tight_layout=True)
axs[0,0].hist(dh.loc[(dh.node_ids==0) & (dh.ts <= 400)])

axs[0,1].hist(dh.loc[(dh.node_ids==1) & (dh.ts <= 400)],color="blue")
axs[0,2].hist(dh.loc[(dh.node_ids==2) & (dh.ts <= 400)])
axs[1,0].hist(dh.loc[(dh.node_ids==3) & (dh.ts <= 400)])
axs[1,1].hist(dh.loc[(dh.node_ids==4) & (dh.ts <= 400)])
axs[1,2].hist(dh.loc[(dh.node_ids==5) & (dh.ts <= 400)])
axs[2,0].hist(dh.loc[(dh.node_ids==6) & (dh.ts <= 400)])
axs[2,1].hist(dh.loc[(dh.node_ids==7) & (dh.ts <= 400)])
axs[2,2].hist(dh.loc[(dh.node_ids==8) & (dh.ts <= 400)])
#axs[0,0].plot(dh.loc[(dh.node_ids==9) & (dh.ts <= 400)])
#print("DATA FRAME VALUES" + to_slice_df)
#to_slice_df = dh.loc[dh.ts <= 400]
print(to_slice_df.head())
plt.show()

# plt.hist(dh.loc[dh.node_ids==0, 'ts'])
# plt.ylabel('node id 0 spike #')
# plt.xlabel('time elapsed (ms)')
# plt.show()
#
# plt.hist(dh.loc[dh.node_ids==1, 'ts'])
# plt.ylabel('node id 1 spike #')
# plt.xlabel('time elapsed (ms)')
# plt.show()
#
# plt.hist(dh.loc[dh.node_ids==2, 'ts'])
# plt.ylabel('node id 2 spike #')
# plt.xlabel('time elapsed (ms)')
# plt.show()
#
# plt.hist(dh.loc[dh.node_ids==3, 'ts'])
# plt.ylabel('node id 3 spike #')
# plt.xlabel('time elapsed (ms)')
# plt.show()
#
# plt.hist(dh.loc[dh.node_ids==4, 'ts'])
# plt.ylabel('node id 4 spike #')
# plt.xlabel('time elapsed (ms)')
# plt.show()
#
# plt.hist(dh.loc[dh.node_ids==5, 'ts'])
# plt.ylabel('node id 5 spike #')
# plt.xlabel('time elapsed (ms)')
# plt.show()
#
# plt.hist(dh.loc[dh.node_ids==6, 'ts'])
# plt.ylabel('node id 6 spike #')
# plt.xlabel('time elapsed (ms)')
# plt.show()
#
# plt.hist(dh.loc[dh.node_ids==7, 'ts'])
# plt.ylabel('node id 7 spike #')
# plt.xlabel('time elapsed (ms)')
# plt.show()
#
# plt.hist(dh.loc[dh.node_ids==8, 'ts'])
# plt.ylabel('node id 8 spike #')
# plt.xlabel('time elapsed (ms)')
# plt.show()
#
# plt.hist(dh.loc[dh.node_ids==9, 'ts'])
# plt.ylabel('node id 9 spike #')
# plt.xlabel('time elapsed (ms)')
# plt.show()
#
# print(dh.loc[dh.node_ids==9, 'ts'])