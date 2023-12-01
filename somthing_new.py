# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 02:46:04 2023

@author: folke
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import spearmanr
from scipy.stats import ttest_1samp
from scipy.stats import f_oneway
from collections import Counter

def columns_to_datetime(df, column_names, time_zone=2):
    for column in column_names:
        df[column] = pd.to_datetime(df[column]+(3.6e6*time_zone), unit='ms').dt.time
    return df

def time_to_seconds(time_obj):
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second

def nearby_cows(df, tag, morning, limit=5000):
    if morning:
        entry_time, exit_time = df.loc[df['tag_id'] == tag, 
                                       ['EntryMorning', 'ExitMorning']].values[0]
        entry_mask = (df['tag_id'] != tag) & \
             (entry_time - limit < df['EntryMorning']) & \
             (df['EntryMorning'] < entry_time + limit)
        exit_mask = (df['tag_id'] != tag) & \
            (exit_time - limit < df['ExitMorning']) & \
            (df['ExitMorning'] < exit_time + limit)
    else:
        entry_time, exit_time = df.loc[df['tag_id'] == tag, 
                                       ['EntryEvening', 'ExitEvening']].values[0]
        entry_mask = (df['tag_id'] != tag) & \
             (entry_time - limit < df['EntryEvening']) & \
             (df['EntryEvening'] < entry_time + limit)

        exit_mask = (df['tag_id'] != tag) & \
            (exit_time - limit < df['ExitEvening']) & \
            (df['ExitEvening'] < exit_time + limit)
            
            
            
    if not df.loc[entry_mask].empty:
        entry_list = df.loc[entry_mask, 'tag_id'].tolist()
    else:
        entry_list = []
        
    if not df.loc[exit_mask].empty:
        exit_list = df.loc[exit_mask, 'tag_id'].tolist()
    else:
        exit_list = []
    return entry_list, exit_list


#%%
direct = 'data/'
prefix = 'group1'
files = [file for file in os.listdir(direct) if file.startswith(prefix)]


week = pd.DataFrame()
tag_conv = pd.read_csv('data/tag_conv.csv')
#week = pd.concat([pd.read_csv(os.path.join(direct, file)) for file in files],
                 #ignore_index=True)
for file in files:
    temp = pd.read_csv(os.path.join(direct, file))
    temp['MilkingDate'] = f"{file[15:19]}-{file[19:21]}-{file[21:23]}"
    
    temp['NearbyMorningEntry'], temp['NearbyMorningExit'] = \
        zip(*[nearby_cows(temp, tag, True) for tag in temp['tag_id']])
    temp['NearbyEveningEntry'], temp['NearbyEveningExit'] = \
        zip(*[nearby_cows(temp, tag, False) for tag in temp['tag_id']])
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
week = pd.merge(week, tag_conv, on='tag_id')
week = pd.merge(week, cow_info, left_on=['tag_string', 'MilkingDate'], right_on=['Tag', 'MilkingDate'])




#week['MorningMinutes'] = (week['ExitMorning']-week['EntryMorning'])*1.666e-5
#week['EveningMinutes'] = (week['ExitEvening']-week['EntryEvening'])*1.666e-5

#%%
time_mean = week['TimeAfterEntry'].mean()
ttest_1samp(week.loc[week['PositionMorning'].isin(['lower_left', 'lower_right'])]['TimeAfterEntry'].to_list(), time_mean)
#ttest_1samp(week.loc[week['Parity'] == 5]['TimeAfterEntry'], time_mean)
#%%
f_oneway(week.loc[week['PositionMorning'] == 'lower_left']['TimeAfterEntry'].to_list()+
         week.loc[week['PositionMorning'] == 'lower_right']['TimeAfterEntry'].to_list(),
         week.loc[week['PositionMorning'] == 'middle_left']['TimeAfterEntry'].to_list(),
         week.loc[week['PositionMorning'] == 'middle_right']['TimeAfterEntry'].to_list(),
         week.loc[week['PositionMorning'] == 'upper_left']['TimeAfterEntry'].to_list(),
         week.loc[week['PositionMorning'] == 'upper_right']['TimeAfterEntry'].to_list())


#%%
chosen_date = '2020-09-19'

day = week.loc[week['MilkingDate'] == chosen_date].copy()
day_mean = day['TimeAfterEntry'].mean()

ttest_1samp(day.loc[day['PositionMorning'].isin(['lower_left', 'lower_right'])]['TimeAfterEntry'].to_list(), day_mean)
#ttest_1samp(week.loc[week['PositionMorning'] == 'lower_right']['TimeAfterEntry'], day_mean)
  
    
 #%%   
    
means = []
for key in area_dict.keys():
    means.append(week.loc[week['PositionMorning'] == key]['TimeAfterEntry'].mean())

plt.bar(list(area_dict.keys()), means)
plt.title('Comparison of seconds passed after first cow enters')
plt.ylabel('Mean seconds after first entry')
plt.xlabel('Part of barn')
plt.xticks(fontsize=8)

#%%

areas = [
    week.loc[week['PositionMorning'] == 'upper_right']['TimeAfterEntry'].to_list(),
    week.loc[week['PositionMorning'] == 'upper_left']['TimeAfterEntry'].to_list(),
    week.loc[week['PositionMorning'] == 'middle_right']['TimeAfterEntry'].to_list(),
    week.loc[week['PositionMorning'] == 'middle_left']['TimeAfterEntry'].to_list(),
    week.loc[week['PositionMorning'] == 'lower_right']['TimeAfterEntry'].to_list(),
    week.loc[week['PositionMorning'] == 'lower_left']['TimeAfterEntry'].to_list(),
    ]
plt.boxplot(areas)
plt.title('Comparison of seconds passed after first cow enters')
plt.xlabel('Part of barn')
plt.ylabel('Seconds after first entry')
plt.ylim([-5,2000])
plt.xticks([1, 2, 3, 4, 5, 6], list(area_dict.keys()), fontsize=8)

#%%
fig, ax = plt.subplots()

for area in area_dict.keys():
    ax.scatter(week.loc[week['PositionMorning'] == area]['Parity'], week.loc[week['PositionMorning'] == area]['TimeAfterEntry'], label=area)

ax.set_xlabel('DIM')  # Replace with your actual x-axis label
ax.set_ylabel('Time After Entry')  # Replace with your actual y-axis label
ax.legend()
#%%
mean_order = week.groupby('Tag')[
    ['EntryOrderMorning', 'ExitOrderMorning',
      'EntryOrderEvening', 'ExitOrderEvening',
       'Parity', 'DIM']].mean()
nearby_entry = {tag:[] for tag in week['tag_id'].unique()}
nearby_exit = {tag:[] for tag in week['tag_id'].unique()} 

for tag in nearby_entry.keys():
    single_cow = week.loc[week['tag_id'] == tag]
    for _, row in single_cow.iterrows():
        nearby_entry[tag].extend([row['NearbyMorningEntry'], row['NearbyEveningEntry']])
        nearby_exit[tag].extend([row['NearbyMorningExit'], row['NearbyEveningExit']])
    
    nearby_entry[tag] = [item for sublist in nearby_entry[tag] for item in sublist]
    nearby_exit[tag] = [item for sublist in nearby_exit[tag] for item in sublist]
#%%
for cow, exit_tags in nearby_exit.items():
    frequency_counter = Counter(exit_tags)
    unique_values = list(frequency_counter.keys())
    unique_values = [str(element) for element in unique_values]
    frequencies = list(frequency_counter.values())
    plt.bar(unique_values, frequencies)
    plt.title(cow)
    plt.show()