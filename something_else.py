# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:15:05 2023

@author: folke
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def divide_groups(df, x_divide=1670):
    """
    divides cows in two groups

    Parameters
    ----------
    df : pd.dataframe
        dataframe with tags from both groups.
    x_divide : integer, optional
        DESCRIPTION. The default is 1670.

    Returns
    -------
    df_g1 : pd.dataframe
        dataframe with tags from group 1 only.
    df_g2 : pd.dataframe
        dataframe with tags from group 2 only.
    tags_g1 : np.array
        group 1 tags.
    tags_g2 : np.array
        group 2 tags.

    """
    tags = df['tag_id'].unique()
    x_avg = df[1_000_000:].groupby('tag_id')['x'].mean()
    tags_g1 = np.array([tag for tag in tags if x_avg[tag] >= x_divide])
    tags_g2 = np.array([tag for tag in tags if x_avg[tag] < x_divide])
    df_g1 = df[df['tag_id'].isin(tags_g1)] 
    df_g2 = df[df['tag_id'].isin(tags_g2)]
    return df_g1, df_g2, tags_g1, tags_g2


def milk_window(df, n_splits=24, n_before=1, n_after=2):
    """
    Gets time windows for morning and evening for one group.

    Parameters
    ----------
    df : pd.dataframe
        should be from only one group.
    n_splits : integer, optional
        number of splits to split day in. The default is 48.
    n_before : TYPE, optional
        number of splits before split with lowest y
        average. The default is 2.
    n_after : TYPE, optional
        number of splits after split with lowest y
        average.. The default is 4.

    Returns
    -------
    df_milk_1 : pd.dataframe
        morning milking window.
    df_milk_2 : pd.dataframe
        evening milking window.

    """
    
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


def entry_exit(df, x_divide=1670, length=500, y_lim=2100):
    """

    Parameters
    ----------
    df : pd.dataframe
        should be dataframe from milk_window().
    x_divide : integer, optional
        The default is 1670.
    length : integer, optional
        The default is 350.
    y_lim : integer, optional
        The default is 2100.

    Returns
    -------
    sorted_entry : dictionary
        sorted entry times for each cow.
    sorted_exit : dictionary
        sorted exit times for each cow.

    """
    low_lim = x_divide-length
    high_lim = x_divide+length
    tag_count = df['tag_id'].value_counts()
    max_y = df.groupby('tag_id')['y'].max()
    active_tags = [tag for tag in df['tag_id'].unique() 
                   if tag_count[tag] > 1000 and max_y[tag] > 3000]
    df = df.loc[df['tag_id'].isin(active_tags) == True]
    
    entry_mask = (df.y < y_lim) & (df.x < high_lim) & (df.x > low_lim)
    exit_mask = df.y < 1600 #1375
    
    
    df_entry = df.loc[entry_mask].copy()
    df_exit = df.loc[exit_mask].copy()

    entry_times = {tag:None for tag in active_tags}
    exit_times = {tag:None for tag in active_tags}
    for cow in active_tags:    
        cow_entry = df_entry.loc[df['tag_id'] == cow]
        cow_exit = df_exit.loc[df['tag_id'] == cow]
        entry_times[cow] = cow_entry['time'].iloc[0]
        exit_times[cow] = cow_exit['time'].iloc[-1]
        if entry_times[cow] + 12e5 > exit_times[cow]:
            #time_bar = (cow_entry['time'] >= entry_times[cow] + 12e5)
            #cow_exit = cow_entry.loc[time_bar & (cow_entry['y'] < 1600), :]
            if not cow_exit.empty:
                exit_times[cow] = cow_exit['time'].iloc[-1]
                
                 
            else:
                del exit_times[cow]
                del entry_times[cow]        
    sorted_entry = dict(sorted(entry_times.items(), key=lambda x: x[1]))
    sorted_exit = dict(sorted(exit_times.items(), key=lambda x: x[1]))
    
    items = list(sorted_entry.items())
    del_list = []
    for i in range(10,0,-1):
        prev_value = items[i-1][1]
        current_value = items[i][1]
        if prev_value + 3e5 < current_value:
            for j in range(i):
                del_list.append(items[j][0])
            break
    for tag in del_list:
        print(f"{tag} deleted")
        del sorted_entry[tag]
        del sorted_exit[tag]
        
    return sorted_entry, sorted_exit


def first_entry(entry_times):
    """
    simply gets first value in dictionary

    Parameters
    ----------
    entry_times : dictionary


    """
    return next(iter(entry_times.items()))[1]

def last_exit(exit_times):
    return list(exit_times.values())[-1]
       


def positions(df, milk_start, area_dict, minutes_before_start=30, minutes_before_end=20):
        """
        assign each cow to area of barn before milking

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        milk_start : TYPE
            DESCRIPTION.
        minutes_before_start : integer, optional
            The default is 30.
        minutes_before_end : integer, optional
            The default is 5.

        Returns
        -------
        cow_dict : TYPE
            DESCRIPTION.

        """
        conv = 60_000
        low_lim = milk_start-minutes_before_start*conv
        high_lim = milk_start-minutes_before_end*conv    
        df = df.loc[(df['time'] > low_lim) & 
                      (df['time'] < high_lim)].copy()
        
        
        
        
        
        tags = df['tag_id'].unique()
        cow_dict = {tag:None for tag in tags}
        
        for tag in tags:
            single_cow = df.loc[df['tag_id'] == tag].copy()
            #single_cow['diff'] = single_cow['time'][::-1].diff()*(-0.001)
            #single_cow['diff'].iloc[-1] = 0
            pos = (np.average(single_cow['x']),
                   np.average(single_cow['y']))
            for area in area_dict.keys():
                if is_inside_new(pos, area_dict[area]):
                    cow_dict[tag] = area
                    break
            if cow_dict[tag] is None:
                del cow_dict[tag]
        return cow_dict

def is_inside_new(pos, area):
    """
    help function, slightly different from is_inside() from pycowview.

    Parameters
    ----------
    pos : tuple
        x, y position.
    area : tuple
        x_min, x_max , y_min, y_max.

    Returns
    -------
    bool
        True if pos is in area.

    """
    if area[0] < pos[0] < area[1] and area[2] < pos[1] < area[3]:
        return True
    else:
        return False
    
    
    
    
def get_number_in_order(times):
    number_in_order = {tag:None for tag in times.keys()}
    for i, tag in enumerate(times.keys()):
        number_in_order[tag] = i+1
    return number_in_order


def valid_start(df, tag, time, area_dict):
    pos = positions(df.loc[df['tag_id'] == tag], time, area_dict)
    if not pos:
        return False
    else:
        return True
    




def summary_dataframe(df, area_dict):
    morning, evening = milk_window(df, n_before=2, n_after=3)
      
    
    
    entry_morning, exit_morning = entry_exit(morning)
    morning_valid = False
    while morning_valid is not True:
        first_key, first_time = next(iter(entry_morning.items()))
        print(f"Morning start {first_time}")
        if valid_start(df, first_key, first_time, area_dict):
            morning_valid = True
        else:
            print(f"{first_key} removed")
            del entry_morning[first_key]
            del exit_morning[first_key]
    entry_order_morning = get_number_in_order(entry_morning)
    exit_order_morning = get_number_in_order(exit_morning)
    
    
    
    entry_evening, exit_evening = entry_exit(evening)
    evening_valid = False
    while evening_valid is not True:
        first_key, first_time = next(iter(entry_evening.items()))
        if valid_start(df, first_key, first_time, area_dict):
            evening_valid = True
        else:
            del entry_evening[first_key]
            del exit_evening[first_key]
    
    entry_order_evening = get_number_in_order(entry_evening)  
    exit_order_evening = get_number_in_order(exit_evening)  
    
    cum_plot(entry_morning)
    
    
    start_morning = first_entry(entry_morning)
    start_evening = first_entry(entry_evening)
    print(f"{start_morning} used")
    
    pos_morning = positions(df, start_morning, area_dict)
    pos_evening = positions(df, start_evening, area_dict)
    
    
    
    
    common_keys = entry_morning.keys() or entry_evening.keys() \
        #& entry_evening.keys() & pos_morning.keys() & pos_evening.keys()
   
        
        
        
    
    group_dict = {
    key: (
        entry_morning.get(key, None),
        entry_order_morning.get(key, None),
        exit_morning.get(key, None),
        exit_order_morning.get(key, None),
        entry_evening.get(key, None),
        entry_order_evening.get(key, None),
        exit_evening.get(key, None),
        exit_order_evening.get(key, None),
        pos_morning.get(key, None),
        pos_evening.get(key, None)
    )
    for key in common_keys
    }

    group_df = pd.DataFrame.from_dict(group_dict, orient='index',
                                  columns=['EntryMorning', 'EntryOrderMorning',
                                           'ExitMorning', 'ExitOrderMorning',
                                           'EntryEvening', 'EntryOrderEvening',
                                           'ExitEvening', 'ExitOrderEvening',
                                           'PositionMorning', 'PositionEvening'])
    group_df.index.name = 'tag_id'
    
    return group_df


def cum_plot(times):
    conv = 1/(1000*60)
    times = {key: value for key, value in times.items() if value != np.inf}
    start = int(first_entry(times)*conv)
    end = int(last_exit(times)*conv)
    x = np.linspace(start, end, end-start)
    y = [sum(value * conv< x_value  for value in times.values()) for x_value in x] 
    plt.plot(x,y)
    plt.show()
    return
