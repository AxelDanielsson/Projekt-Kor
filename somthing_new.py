# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 02:46:04 2023

@author: folke
"""

import pandas as pd
import os
from scipy.stats import spearmanr
from scipy.stats import ttest_1samp

def columns_to_datetime(df, column_names, time_zone=2):
    for column in column_names:
        df[column] = pd.to_datetime(df[column]+(3.6e6*time_zone), unit='ms').dt.time
    return df

def time_to_seconds(time_obj):
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second



direct = 'data/'
prefix = 'group1'
files = [file for file in os.listdir(direct) if file.startswith(prefix)]


week = pd.DataFrame()

#week = pd.concat([pd.read_csv(os.path.join(direct, file)) for file in files],
                 #ignore_index=True)
for file in files:
    temp = pd.read_csv(os.path.join(direct, file))
    temp['MilkingDate'] = f"{file[15:19]}-{file[19:21]}-{file[21:23]}"
    
    temp = columns_to_datetime(temp,
                              ['EntryMorning', 'ExitMorning', 
                               'EntryEvening', 'ExitEvening'])
    
    temp = temp.sort_values(by='EntryMorning')
    first_entry = time_to_seconds(temp['EntryMorning'].iloc[0]) 
    temp['TimeAfterEntry'] = temp['EntryMorning'].apply(lambda x:
                                                 (x.hour * 3600 + x.minute \
                                                  * 60 + x.second - first_entry)).to_list()
    week = pd.concat([week, temp], ignore_index=True)
    

cow_info = pd.read_csv('data/CowInfo.csv')

week = pd.merge(week, cow_info, on=['Tag', 'MilkingDate'])




#week['MorningMinutes'] = (week['ExitMorning']-week['EntryMorning'])*1.666e-5
#week['EveningMinutes'] = (week['ExitEvening']-week['EntryEvening'])*1.666e-5

#%%
time_mean = week['TimeAfterEntry'].mean()
t, p = ttest_1samp(week.loc[week['PositionMorning'] == 'lower_right']['TimeAfterEntry'].to_list(), time_mean)


#%%
chosen_date = '2020-09-20'

day = week.loc[week['MilkingDate'] == chosen_date].copy()



  
    
    
    
    
    
day = day.sort_values(by='EntryMorning')
entry = day['EntryOrderMorning'].to_list()
    
exits = day['ExitOrderMorning'].to_list()

spear, pvalue = spearmanr(entry, exits)








mean_order = week.groupby('Tag')[
    ['EntryOrderMorning', 'ExitOrderMorning',
     'EntryOrderEvening', 'ExitOrderEvening',
     'MorningMinutes', 'EveningMinutes', 'Parity']].mean()
