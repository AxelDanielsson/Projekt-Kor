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
from matplotlib.animation import FuncAnimation
from datetime import datetime, timedelta

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
def entry_exit(df, tags, start_day, x_divide=1670, length=250, tag_lim=1000):
    low_lim = x_divide-length
    high_lim = x_divide+length
    tag_count = df['tag_id'].value_counts()
    active_tags = [tag for tag in tags if tag_count[tag] > tag_lim]
    entry_times = {tag:None for tag in active_tags}
    exit_times = {tag:None for tag in active_tags}
    
    pos_mask = (df.y < high_lim) & (df.x < high_lim) & (df.x > low_lim)
    df = df.loc[pos_mask]
    for cow in active_tags:    
        single_cow = df.loc[df['tag_id'] == cow].copy()
        entry_times[cow] = single_cow['time'].iloc[0]
        exit_times[cow] = single_cow['time'].iloc[-1]
    sorted_entry = dict(sorted(entry_times.items(), key=lambda x: x[1]))
    sorted_exit = dict(sorted(exit_times.items(), key=lambda x: x[1]))
    return sorted_entry, sorted_exit

entry_times, exit_times = entry_exit(df_milk_1, tags_g2, start_day)

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
def first_entry(entry_times):
    return next(iter(entry_times.items()))[1]

milk_start = first_entry(entry_times)
#%%
def before_milk_pos(df, milk_start, tags, minutes=1, minutes_before=30):
    conv = 0.5*(1000*60)
    milk_start = milk_start - (minutes_before*60*1000)
    time_mask = (
    (df['time'] > milk_start - (minutes * conv)) &
    (df['time'] < milk_start + (minutes * conv)) &
    (df['tag_id'].isin(tags))
    )
    df_time = df.loc[time_mask]
    avg_pos = df_time.groupby('tag_id')[['x', 'y']].mean().reset_index()
    return avg_pos
df_avg_pos = before_milk_pos(df_g2, milk_start, list(entry_times.keys()), minutes_before=20)
#%%
def plot_gantt(entry_times, exit_times, start_day):
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, tag in enumerate(reversed(entry_times.keys())):
        ax.barh(str(tag),
               left=(entry_times[tag]-start_day)/(1000*60*60)+2,
               width=(exit_times[tag]-entry_times[tag])/(1000*60*60),
               color='black',
               edgecolor='black')
    plt.title('Gantt Chart')
    plt.yticks([])
    plt.xticks([4,5,6,7], ['04:00', '05:00', '06:00', '07:00'])
    plt.grid(True)
    plt.show()
    return
plot_gantt(entry_times, exit_times, start_day)
#%%
def plot_pos(df_pos, tags, barn_filename):
    fig, ax = plot_barn(barn_filename)
    df_plot_red = df_pos.loc[df_pos['tag_id'].isin(tags)]
    df_plot_grey = df_pos.loc[~df_pos['tag_id'].isin(tags)]
    for row in df_plot_grey.iterrows():
        plt.scatter(row[1]['x'], row[1]['y'], color='grey', s=20)
    for row in df_plot_red.iterrows():
        plt.scatter(row[1]['x'], row[1]['y'], color='r', s=35)
    return
plot_pos(df_avg_pos, [2432652, 2426250, 2428364, 2428706, 2433150], barn_filename)
#%%
def plot_pos_animation(df, high_tags, tags, milk_start):
    fig, ax = plot_barn(barn_filename)
    
    
    
    red_scatter = ax.scatter([], [], color='r', s=35)
    grey_scatter = ax.scatter([], [], color='grey', s=20)
    
    def update(frame):
        df_pos = before_milk_pos(df, milk_start, tags, minutes_before=frame)
        df_plot_red = df_pos.loc[df_pos['tag_id'].isin(high_tags)]
        df_plot_grey = df_pos.loc[~df_pos['tag_id'].isin(high_tags)]
        
        red_scatter.set_offsets(df_plot_red[['x', 'y']].values)
        grey_scatter.set_offsets(df_plot_grey[['x', 'y']].values)
        return red_scatter, grey_scatter
    
    animation = FuncAnimation(fig, update, frames=range(25, -16, -1), interval=250, blit=True)
    animation.save('animations/test.gif', writer='imagemagick')
    plt.show()
    return
plot_pos_animation(df_g2, [2432652, 2426250, 2428364, 2428706, 2433150], tags_g2, milk_start)

#%%
plot_cow(df_milk_1, tags_g2[-1], barn_filename)


