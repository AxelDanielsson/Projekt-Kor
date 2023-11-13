# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 16:24:30 2023

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
def time_of_day(start_day, time_to_convert, time_zone=2):
    seconds_diff = (time_to_convert-start_day)/1_000
    time_of_day = datetime.timedelta(seconds=seconds_diff+time_zone*3_600)
    return time_of_day
start_day = df['time'].iloc[0]

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
def milk_times(df, n_splits=48, n_before=2, n_after=4):
    y_arr = np.zeros(n_splits)
    splits = np.array_split(df, n_splits)
    for i, split in  enumerate(splits):
        y_arr[i] = split['y'].mean()
    indices = np.argpartition(y_arr, 2)[:2]
    morning = np.min(indices)
    evening = np.max(indices)
    df_milk_1 = pd.concat(splits[morning-n_before:morning+n_after])
    df_milk_2 = pd.concat(splits[evening-n_before:evening+n_after])
    return df_milk_1, df_milk_2
df_milk_1, df_milk_2 = milk_times(df_g2)

#%%
def entry_exit(df, tags, start_day, x_divide=1670, length=250):
    low_lim = x_divide-length
    high_lim = x_divide+length
    entry_times = {tag:None for tag in tags}
    exit_times = {tag:None for tag in tags}
    
    tag_count = df['tag_id'].value_counts()
    
    pos_mask = (df.y < high_lim) & (df.y > low_lim) & (df.x < high_lim) & (df.x > low_lim)
    df = df.loc[pos_mask]
    for cow in tags:    
        if tag_count[cow] > 1000:
            single_cow = df.loc[df['tag_id'] == cow].copy()
            entry_times[cow] = single_cow['time'].iloc[0]
            exit_times[cow] = single_cow['time'].iloc[-1]
    sorted_entry = dict(sorted(entry_times.items(), key=lambda x:x[1]))
    sorted_exit = dict(sorted(exit_times.items(), key=lambda x:x[1]))
    return sorted_entry, sorted_exit

entry_times, exit_times = entry_exit(df_milk_2, tags_g2, start_day)

#%%
def write_to_file(filename, tags, entry_times, exit_times, start_day):
    with open(filename, 'w') as f:
        f.write(f'Tag     \t Entry time       \t Exit time \n')
        for tag in entry_times.keys():
            entry_time = time_of_day(start_day, entry_times[tag])
            exit_time = time_of_day(start_day, exit_times[tag])
            f.write(f'{tag} \t {entry_time} \t {exit_time} \n')
    return

write_to_file('results_test_g2_2.txt', tags_g2, entry_times, exit_times, start_day)
#%%

plot_cow(df_milk_1, tags_g2[-1], barn_filename)


