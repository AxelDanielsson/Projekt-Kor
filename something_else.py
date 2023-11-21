# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:15:05 2023

@author: folke
"""

import numpy as np
import pandas as pd


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


def milk_window(df, n_splits=48, n_before=2, n_after=4):
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


def entry_exit(df, x_divide=1670, length=350, y_lim=2100):
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
    entry_times = {tag:None for tag in active_tags}
    exit_times = {tag:None for tag in active_tags}
    
    
    pos_mask = (df.y < y_lim) & (df.x < high_lim) & (df.x > low_lim)
    df = df.loc[pos_mask]
    for cow in active_tags:    
        single_cow = df.loc[df['tag_id'] == cow].copy()
        entry_times[cow] = single_cow['time'].iloc[0]
        exit_times[cow] = single_cow['time'].iloc[-1]
    sorted_entry = dict(sorted(entry_times.items(), key=lambda x: x[1]))
    sorted_exit = dict(sorted(exit_times.items(), key=lambda x: x[1]))
    return sorted_entry, sorted_exit


def first_entry(entry_times):
    """
    simply gets first value in dictionary

    Parameters
    ----------
    entry_times : dictionary


    """
    return next(iter(entry_times.items()))[1]
       


def positions(df, milk_start, minutes_before_start=30, minutes_before_end=5):
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
        
        
        area_dict = {
            'upper_middle': (1670, 2482.5, 5851.5, 8738),
            'upper_right': (2482.5, 3340, 5851.5, 8738),
            'middle_middle': (1670, 2482.5, 3242.5, 5851.5),
            'middle_right': (2482.5, 3340, 3242.5, 5851.5),
            'lower' : (1670, 3340, 2200, 8738)
           
        }
        
        
        tags = df['tag_id'].unique()
        cow_dict = {tag:None for tag in tags}
        
        for tag in tags:
            single_cow = df.loc[df['tag_id'] == tag].copy()
            single_cow['diff'] = single_cow['time'][::-1].diff()*(-0.001)
            single_cow.iloc[-1] = 0
            pos = (np.average(single_cow['x']),
                   np.average(single_cow['y']))
            for area in area_dict.keys():
                if is_inside_new(pos, area_dict[area]):
                    cow_dict[tag] = area
                    break
            
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