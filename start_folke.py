# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 09:41:12 2023

@author: folke
"""

from pycowview.data import csv_read_FA
from pycowview.plot import plot_cow
from pycowview.plot import plot_barn
from pycowview.animation import animate_cows
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

#%%
#
data_filename = 'data/FA_20200922T000000UTC.csv'
barn_filename = 'data/barn.csv'
nrows = 0
df = csv_read_FA(data_filename, nrows)
tags = df['tag_id'].unique()
#%%
def remove_stationary_tags(df):
    y_min_max = df.groupby('tag_id')['y'].agg(['min', 'max'])

    y_min_max['diff'] = y_min_max['max'] - y_min_max['min']

    stationary_list = y_min_max.loc[y_min_max['diff'] < 4e3].index.to_list()

    df = df[~df['tag_id'].isin(stationary_list)]

    
    return df, stationary_list

df_new, stationary_list = remove_stationary_tags(df)
tags_cow = [tag for tag in tags if tag not in stationary_list]
#%%
def time_of_day(start_day, time_to_convert, time_zone):
    seconds_diff = (time_to_convert-start_day)/1_000
    time_of_day = datetime.timedelta(seconds=seconds_diff+time_zone*3_600)
    return time_of_day
start_day = df['time'].iloc[0]
time_zone = 2
#%%




def divide_groups(df, tags, x_divide):
    x_avg = df[1_000_000:].groupby('tag_id')['x'].mean()
    tags_g1 = np.array([tag for tag in tags if x_avg[tag] <= x_divide])
    tags_g2 = np.array([tag for tag in tags if x_avg[tag] > x_divide])
    df_g1 = df[df['tag_id'].isin(tags_g1)] 
    df_g2 = df[df['tag_id'].isin(tags_g2)]
    return df_g1, df_g2, tags_g1, tags_g2


df_g1, df_g2, tags_g1, tags_g2 = divide_groups(df_new, tags_cow, 1_670)

#%%
#def milk_times(df)
n_splits = 48
y_arr = np.zeros(n_splits)
splits = np.array_split(df_g1, n_splits)
for i, split in  enumerate(splits):
    y_arr[i] = split['y'].mean()
indices = np.argpartition(y_arr, 2)[:2]
df_milk_1 = pd.concat(splits[indices[1]-1:indices[1]+3])
#%%
milking_dct = {tag:False for tag in tags_g1}
entry_times = {tag:[] for tag in tags_g1}
entry_times_day = {tag:[] for tag in tags_g1}
exit_times = {tag:[] for tag in tags_g1}
pos_lst = []
#%%
# crossing line
for row in df_milk_1.iterrows():
    tag = row[1]['tag_id']
    if milking_dct[tag] == False:
        if row[1]['y'] <= 2000 and 1400 < row[1]['x'] < 2000:
            entry_times[tag].append(row[1]['time'])
            entry_times_day[tag].append(time_of_day(start_day, row[1]['time'], time_zone))
            milking_dct[tag] = True
    else:
        if row[1]['y'] >= 2500 and 1400 < row[1]['x'] < 2000:
            exit_times[tag].append(time_of_day(start_day, row[1]['time'], time_zone))
            milking_dct[tag] = False
         

#%%
# disappear from data
for cow in tags_g1:    
    single_cow = df_milk_1.loc[df_milk_1['tag_id'] == cow].copy()
    single_cow['time_diff'] = single_cow['time'].diff()
    max_row = single_cow.loc[single_cow['time_diff'] == single_cow['time_diff'].max()]
    exit_time = max_row['time'].values[0]
    entry_times[cow].append(time_of_day(start_day, exit_time-max_row['time_diff'].values[0], time_zone))
    exit_times[cow].append(time_of_day(start_day, exit_time, time_zone))
    pos_lst.append((max_row['x'].values[0], max_row['y'].values[0]))
    
#single_cow['time_of_day'] = single_cow['time'].apply(lambda x : time_of_day(start_day, x, time_zone))
#%%

plot_cow(df, 2428890, barn_filename)
#%%

x = [point[0] for point in pos_lst]
y = [point[1] for point in pos_lst]

# Create a scatter plot
fig, ax = plot_barn(barn_filename)
plt.scatter(x, y, color='b', label='Data Points')
#%%
animate_cows(df_milk_1, 2428721, 2427878, barn_filename, save_path='test2.gif')

#%%
with open('results.txt', 'w') as f:
    for tag in tags_g1:
        f.write(f'{tag} \t {entry_times_day[tag][0]} \t \n')


#%%
#fig, ax = plot_barn(barn_filename)
#plt.scatter([1400, 2000], [2000, 2000])