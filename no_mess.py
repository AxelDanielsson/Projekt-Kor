# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 16:24:30 2023

@author: folke
"""
import matplotlib.pyplot as plt
from pycowview.data import csv_read_FA
from pycowview.plot import plot_cow
from pycowview.plot import plot_barn
import pandas as pd 
import numpy as np

from matplotlib.animation import FuncAnimation
from scipy.stats import spearmanr
from datetime import datetime, timedelta

#%%
#
data_filename = 'data/FA_20200922T000000UTC.csv'
barn_filename = 'data/barn.csv'
nrows = 0
df = csv_read_FA(data_filename, nrows)
tags = df['tag_id'].unique()
#%%
def remove_stationary_tags(df, limit=4e3):
    """
    This function removes performance tags and possibly some cows that are not
    interesting.

    Parameters
    ----------
    df : pd.dataframe
        should be a for one entire day
    limit: float
        minimum difference between min and max y to not be removed.
        
    Returns
    -------
    df : pd.dataframe
        dataframe without tags identified as 'stationary'.
    stationary_list : list
        list of tags removed.

    """
    y_min_max = df.groupby('tag_id')['y'].agg(['min', 'max'])

    y_min_max['diff'] = y_min_max['max'] - y_min_max['min']

    stationary_list = y_min_max.loc[y_min_max['diff'] < limit].index.to_list()

    df = df[~df['tag_id'].isin(stationary_list)]

    
    return df, stationary_list

df_new, stationary_list = remove_stationary_tags(df)
tags_cow = [tag for tag in tags if tag not in stationary_list]
#%%
def time_of_day(start_day, time_to_convert, time_zone=2):
    """
    converts time in strange microseconds to time of day based on 
    first row in data for the day

    Parameters
    ----------
    start_day : integer
        time for first position of the day.
    time_to_convert : integer
        time to convert.
    time_zone : integer, optional
        DESCRIPTION. 2 for summer time, 1 for normal.

    Returns
    -------
    time_of_day : datetime.timedelta
        A more readable representation of time_to_convert.

    """
    seconds_diff = (time_to_convert-start_day)/1_000
    time_of_day = timedelta(seconds=seconds_diff+time_zone*3_600)
    return time_of_day
start_day = df['time'].iloc[0]



#%%

def divide_groups(df, tags, x_divide=1670):
    """
    divides cows in two groups

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    tags : TYPE
        DESCRIPTION.
    x_divide : integer, optional
        DESCRIPTION. The default is 1670.

    Returns
    -------
    df_g1 : pd.dataframe
        DESCRIPTION.
    df_g2 : pd.dataframe
        DESCRIPTION.
    tags_g1 : np.array
        DESCRIPTION.
    tags_g2 : np.array
        DESCRIPTION.

    """
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
def entry_exit(df, tags, start_day, x_divide=1670, length=350, tag_lim=1000):
    low_lim = x_divide-length
    high_lim = x_divide+length
    tag_count = df['tag_id'].value_counts()
    max_y = df.groupby('tag_id')['y'].max()
    active_tags = [tag for tag in tags if tag_count[tag] > tag_lim and max_y[tag] > 3000]
    entry_times = {tag:None for tag in active_tags}
    exit_times = {tag:None for tag in active_tags}
    
    
    pos_mask = (df.y < 2100) & (df.x < high_lim) & (df.x > low_lim)
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

def get_number_in_order(times):
    number_in_order = {tag:None for tag in times.keys()}
    for i, tag in enumerate(times.keys()):
        number_in_order[tag] = i+1
    return number_in_order

entry_order = get_number_in_order(entry_times)


#%%
def write_to_file(filename, tags, entry_times, exit_times, start_day):
    with open(filename, 'w') as f:
        f.write(f'Tag     \t Entry time       \t Exit time \n')
        for tag in entry_times.keys():
            entry_time = time_of_day(start_day, entry_times[tag])
            exit_time = time_of_day(start_day, exit_times[tag])
            f.write(f'{tag} \t {entry_time} \t {exit_time} \n')
    return

write_to_file('overlap2.txt', tags_g2, entry_times, exit_times, start_day)


#%%
def first_entry(entry_times):
    return next(iter(entry_times.items()))[1]

milk_start = first_entry(entry_times)

def last_exit(exit_times):
    return list(exit_times.values())[-1]
milk_end = last_exit(exit_times)
#%%
def before_milk_pos(df, milk_start, tags, minutes=0.5, minutes_before=30):
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
df_avg_pos = before_milk_pos(df_g1, milk_start, list(entry_times.keys()))
#%%
def plot_gantt(entry_times, exit_times, start_day):
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, tag in enumerate(reversed(entry_times.keys())):
        ax.barh(str(tag),
               left=(entry_times[tag]-start_day)/(1000*60*60)+2,
               width=(exit_times[tag]-entry_times[tag])/(1000*60*60),
               color='black',
               edgecolor='black')
    plt.title('Entry and Exit Times, Evening Group 2', fontsize=20)
    plt.yticks([])
    plt.xticks([18, 19, 20], ['16:00', '17:00', '18:00'], fontsize=18)
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
        plt.scatter(row[1]['x'], row[1]['y'], color='b', s=35)
        
    plt.title('Positions 30 minutes before first cow enters milking station')    
    
    plt.yticks([])
    plt.xticks([])
    return
plot_pos(df_avg_pos, [2421773, 2428348, 2431954, 2427562, 2423369], barn_filename)

 #%%
def plot_pos_animation(df, high_tags, low_tags, tags, milk_start):
    fig, ax = plot_barn(barn_filename)
    
    
    
    red_scatter = ax.scatter([], [], color='r', s=35)
    blue_scatter = ax.scatter([], [], color='b', s=35)
    grey_scatter = ax.scatter([], [], color='grey', s=20)
    
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='black')
    
    def update(frame):
        df_pos = before_milk_pos(df, milk_start, tags, minutes_before=frame)
        df_plot_red = df_pos.loc[df_pos['tag_id'].isin(high_tags)]
        df_plot_blue = df_pos.loc[df_pos['tag_id'].isin(low_tags)]
        df_plot_grey = df_pos.loc[~df_pos['tag_id'].isin(low_tags) & 
                                  ~df_pos['tag_id'].isin(high_tags)]
        
        
        grey_scatter.set_offsets(df_plot_grey[['x', 'y']].values)
        red_scatter.set_offsets(df_plot_red[['x', 'y']].values)
        blue_scatter.set_offsets(df_plot_blue[['x', 'y']].values)
        time_text.set_text(f'Minutes before/after first entry: {frame}') 
        plt.xticks([])
        plt.yticks([])
        return red_scatter, grey_scatter
    
    animation = FuncAnimation(fig, update, frames=range(25, -126, -1), interval=350, blit=True)
    animation.save('animations/test_exlong.gif')
    plt.show()
    return
plot_pos_animation(df_g1, [2427905, 2427616, 2432089, 2428902, 2428776], 
                   [2421773, 2428348, 2431954, 2427562, 2423369],
                   tags, milk_start)

#%%
def cum_plot(times):
    conv = 1/(1000*60)
    start = int(first_entry(times)*conv)
    end = int(last_exit(times)*conv)
    x = np.linspace(start, end, end-start)
    y = [sum(value * conv< x_value  for value in times.values()) for x_value in x] 
    plt.plot(x,y)
    return
cum_plot(exit_times)
    


#%%
def positions(df, milk_start, barn_filename, minutes_before_start=30, minutes_before_end=5):
    conv = 60_000
    low_lim = milk_start-minutes_before_start*conv
    high_lim = milk_start-minutes_before_end*conv    
    df = df.loc[(df['time'] > low_lim) & 
                  (df['time'] < high_lim)].copy()
    
    
    area_dict = {
        'upper_middle': [1670, 2482.5, 5851.5, 8738],
        'upper_right': [2482.5, 3340, 5851.5, 8738],
        'middle_middle': [1670, 2482.5, 3242.5, 5851.5],
        'middle_right': [2482.5, 3340, 3242.5, 5851.5],
        'lower' : [1670, 3340, 2200, 8738]
       
    }
    
    
    tags = df['tag_id'].unique()
    cow_dict = {tag:None for tag in tags}
    
    for tag in tags:
        single_cow = df.loc[df['tag_id'] == tag].copy()
        pos = (single_cow['x'].mean(), single_cow['y'].mean())
        for area in area_dict.keys():
            if is_inside_new(pos, area_dict[area]):
                cow_dict[tag] = area
                break
        
    return cow_dict
area_pos = positions(df_g2, milk_start, barn_filename)


#%%
def is_inside_new(pos, area):
    if area[0] < pos[0] < area[1] and area[2] < pos[1] < area[3]:
        return True
    else:
        return False


#%%
def plot_area(cow_dict, extra_pos, barn_filename):
    """
    

   outdated and useless at the moment

    """
    barn = pd.read_csv(barn_filename, delimiter=';')
    
    
    drop_values = ['Base', 'bed8', 'bed9']
    barn = barn[~barn['Unit'].isin(drop_values)]
    for cow in cow_dict.keys():
        fig, ax = plot_barn(barn_filename)
        plt.title(f"{cow}")
        for area in barn[1:].itertuples():
            x = [area.x1, area.x2, area.x3, area.x4]
            y = [area.y1, area.y2, area.y3, area.y4]
            plt.fill(x + [x[0]], y + [y[0]], color='r',
                     alpha=round(cow_dict[cow][area.Unit], 3))
        x_extra, y_extra = zip(*extra_pos[cow])
        plt.scatter(x_extra, y_extra, color='r', alpha=0.5)
    
    return
#plot_area(area_pos, extra_pos, barn_filename)
#%%
def get_spearman_cor(entry_times, exit_times):
    spear = spearmanr(list(entry_times.keys()), list(exit_times.keys()))
    return spear.statistic, spear.pvalue
spear, pvalue = get_spearman_cor(entry_times, exit_times)

#%%

plot_cow(df_milk_1, 2417161, barn_filename)

barn = pd.read_csv(barn_filename, delimiter=';')
#%%
area_dict = {
     'upper_middle': [1670, 2482.5, 5851.5, 8738],
     'upper_right': [2482.5, 3340, 5851.5, 8738],
     'middle_middle': [1670, 2482.5, 3242.5, 5851.5],
     'middle_right': [2482.5, 3340, 3242.5, 5851.5],
     'lower' : [1670, 3340, 2200, 8738]    
 }

fig, ax = plot_barn(barn_filename)
for area in area_dict:
    coordinates = area_dict[area]
    x = [coordinates[0], coordinates[0], coordinates[1], coordinates[1]]
    y = [coordinates[2], coordinates[3], coordinates[3], coordinates[2]]
    plt.fill(x + [x[0]], y + [y[0]], alpha=0.5)


