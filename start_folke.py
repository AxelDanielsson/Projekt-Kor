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
milking_dct = {tag:False for tag in tags_cow}
entry_times = {tag:[] for tag in tags_cow}
exit_times = {tag:[] for tag in tags_cow}
#%%
for row in df_new[:3000000].iterrows():
    tag = row[1]['tag_id']
    if milking_dct[tag] == False:
        if row[1]['y'] <= 2000 and 1400 < row[1]['x'] < 2000:
            entry_times[tag].append(row[1]['time'])
            milking_dct[tag] = True
    else:
        if row[1]['y'] >= 2000 and 1400 < row[1]['x'] < 2000 and entry_times[tag][-1]+45*60_000<row[1]['time']:
            exit_times[tag].append(row[1]['time'])
            milking_dct[tag] = False
         
            
    

#%%

plot_cow(df_new[:2500000], 2420330, barn_filename)

#%%
#fig, ax = plot_barn(barn_filename)
#plt.scatter([1400, 2000], [2000, 2000])