# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 09:41:12 2023

@author: folke
"""

from pycowview.data import csv_read_FA
from pycowview.plot import plot_cow
from pycowview.plot import plot_barn
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
def time_of_day(start_time, time_to_convert, time_zone):
    seconds_diff = (time_to_convert-start_time)/1_000
    time_of_day = datetime.timedelta(seconds=seconds_diff+time_zone*3_600)
    return time_of_day

time = time_of_day(1600732800023, 1600751577548, 2)
#%%




def divide_groups(df, tags, x_divide):
    x_avg = df[1_000_000:].groupby('tag_id')['x'].mean()
    tags_g1 = np.array([tag for tag in tags if x_avg[tag] >= x_divide])
    tags_g2 = np.array([tag for tag in tags if x_avg[tag] < x_divide])
    df_g1 = df[df['tag_id'].isin(tags_g1)] 
    df_g2 = df[df['tag_id'].isin(tags_g2)]
    return df_g1, df_g2, tags_g1, tags_g2


df_g1, df_g2, tags_g1, tags_g2 = divide_groups(df_new, tags_cow, 1_670)

#%%
#def milk_times(df)
n_splits = 12
y_arr = np.zeros(n_splits)
for i, interval in  enumerate(np.array_split(df_g1, n_splits)):
    y_arr[i] = interval['y'].mean()
indices = np.argpartition(y_arr, 2)[:2]

#%%
milking_dct = {tag:False for tag in tags_cow}
entry_times = {tag:[] for tag in tags_cow}
exit_times = {tag:[] for tag in tags_cow}
#%%
for row in df_new[:4000000].iterrows():
    tag = row[1]['tag_id']
    if milking_dct[tag] == False:
        if row[1]['y'] <= 2000 and 1400 < row[1]['x'] < 2000:
            entry_times[tag].append(row[1]['time'])
            milking_dct[tag] = True
    else:
        if row[1]['y'] >= 2000 and 1400 < row[1]['x'] < 2000 and entry_times[tag][-1]+20*60_000<row[1]['time']:
            exit_times[tag].append(row[1]['time'])
            milking_dct[tag] = False
         
            
    

#%%

plot_cow(df_new[2940252:], tag, barn_filename)

#%%
#fig, ax = plot_barn(barn_filename)
#plt.scatter([1400, 2000], [2000, 2000])