# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 02:46:04 2023

@author: folke
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import spearmanr
from scipy.stats import ttest_1samp
from scipy.stats import f_oneway
from collections import Counter
#%% 
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


def order(df, sort_column, order_column, pos_column):
    df = df.sort_values(by=sort_column)
    df[order_column] = 0
    count = 1 
    for index, row in df.iterrows():
        if row[pos_column] is not None:
            df.at[index, order_column] = count
            count += 1
        else:
            df.at[index, order_column] = None
    return df
    


#%%
direct = 'data/'
prefix = 'group2'
files = [file for file in os.listdir(direct) if file.startswith(prefix)]


if prefix == 'group2':
    area_dict = {
        'upper_left': (1670, 2482.5, 5851.5, 8738),
        'upper_right': (2482.5, 3340, 5851.5, 8738),
        'middle_left': (1670, 2482.5, 3242.5, 5851.5),
        'middle_right': (2482.5, 3340, 3242.5, 5851.5),
        'lower_left' : (1670, 2482.5, 2100, 3242.5),
        'lower_right' : (2482.5, 3340, 2595, 3242.5)
    }
else:
    area_dict = {
        'upper_left': (0, 881, 5851.5, 8738),
        'upper_right': (881, 1670, 5851.5, 8738),
        'middle_left': (0, 881, 3242.5, 5851.5),
        'middle_right': (881, 1670, 3242.5, 5851.5),
        'lower_left' : (0, 881, 2100, 3242.5), 
        'lower_right': (881, 1670, 2595, 3242.5)
    }


week = pd.DataFrame()
tag_conv = pd.read_csv('data/tag_conv.csv')

for file in files:
    temp = pd.read_csv(os.path.join(direct, file))
    #date to merge with CowInfo
    temp['MilkingDate'] = f"{file[15:19]}-{file[19:21]}-{file[21:23]}"
    
    
    
    #minutes between cow entering and exiting
    temp['MorningMinutes'] = (temp['ExitMorning']-temp['EntryMorning'])*1.666e-5
    temp['EveningMinutes'] = (temp['ExitEvening']-temp['EntryEvening'])*1.666e-5
    
    #find cows entering or leaving together
    temp['NearbyMorningEntry'], temp['NearbyMorningExit'] = \
        zip(*[nearby_cows(temp, tag, True) for tag in temp['tag_id']])
    temp['NearbyEveningEntry'], temp['NearbyEveningExit'] = \
        zip(*[nearby_cows(temp, tag, False) for tag in temp['tag_id']])
    temp = columns_to_datetime(temp,
                              ['EntryMorning', 'ExitMorning', 
                               'EntryEvening', 'ExitEvening'])

    #time after the first cow enters
    temp = temp.sort_values(by='EntryMorning')
    first_entry = time_to_seconds(temp['EntryMorning'].iloc[0]) 
    temp['SecondsAfterMorning'] = temp['EntryMorning'].apply(lambda x:
                                                 (x.hour * 3600 + x.minute \
                                                  * 60 + x.second - first_entry)).to_list()
    
    temp = temp.sort_values(by='EntryEvening')
    first_entry = time_to_seconds(temp['EntryEvening'].iloc[0]) 
    temp['SecondsAfterEvening'] = temp['EntryEvening'].apply(lambda x:
                                                 (x.hour * 3600 + x.minute \
                                                  * 60 + x.second - first_entry)).to_list()
    
    temp = order(temp, 'ExitEvening', 'ExitOrderEvening', 'PositionEvening')    
    temp = order(temp, 'EntryEvening', 'EntryOrderEvening', 'PositionEvening')    
    temp = order(temp, 'ExitMorning', 'ExitOrderMorning', 'PositionMorning')    
    temp = order(temp, 'EntryMorning', 'EntryOrderMorning', 'PositionMorning')
    
    #add day to week
    week = pd.concat([week, temp], ignore_index=True)
    
#merge with CowInfo
cow_info = pd.read_csv('data/CowInfo.csv')
week = pd.merge(week, tag_conv, on='tag_id')
week = pd.merge(week, cow_info, left_on=['tag_string', 'MilkingDate'], right_on=['Tag', 'MilkingDate'])






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

ttest_1samp(day.loc[day['PositionMorning'].isin(['lower_left', 'lower_right'])]['TimeAfterEntry'].to_list(), day_mean)
#ttest_1samp(week.loc[week['PositionMorning'] == 'lower_right']['TimeAfterEntry'], day_mean)
  
    
#%%   
# Time after entry boxplot area
lists_to_plot = []
keys = list(area_dict.keys())
for key in keys:
    temp_list = [time for time in week.loc[week['PositionMorning'] == key]['SecondsAfterMorning'].to_list() if time < 2000]
    temp_list += [time for time in week.loc[week['PositionEvening'] == key]['SecondsAfterEvening'].to_list() if time < 2000]
    lists_to_plot.append(temp_list)
lists_to_plot[-2] = lists_to_plot[-2] + lists_to_plot[-1]
lists_to_plot.pop(-1)
#sns.violinplot(data=lists_to_plot)
plt.boxplot(lists_to_plot, showfliers=False, whiskerprops=dict(linestyle='-', linewidth=1.5, color='black', solid_capstyle='round'))
plt.title(f"Seconds passed after the first cow enters {prefix}", fontsize=14)
plt.ylabel('Seconds')
plt.xlabel('Part of the barn')
plt.xticks([1, 2, 3, 4, 5], [keys[0], keys[1], keys[2], keys[3], 'lower'], fontsize=10)
plt.gca().set_facecolor('whitesmoke')


ttest_1samp(lists_to_plot[-1], (week['SecondsAfterMorning'].mean()+week['SecondsAfterEvening'].mean())/2)
#%%
# Time after entry boxplot parity

masks = [week.Parity == 1, week.Parity == 2, week.Parity == 3, week.Parity > 3]

lists_to_plot = []


for mask in masks:
    temp_list = [time for time in week.loc[mask, 'SecondsAfterMorning'].to_list() 
                 if time < 2000]
    temp_list += [time for time in week.loc[mask, 'SecondsAfterEvening'].to_list() 
                  if time < 2000]
    lists_to_plot.append(temp_list)


plt.boxplot(lists_to_plot, showfliers=True, whiskerprops=dict(linestyle='-', linewidth=1.5, color='black', solid_capstyle='round'))
plt.title(f"Seconds passed after first entry {prefix}", fontsize=14)
plt.ylabel('Seconds')
plt.xlabel('Parity')
plt.xticks([1, 2, 3, 4], ['1', '2', '3', '4+'], fontsize=12)
plt.gca().set_facecolor('whitesmoke')


#%%
masks = [week.Parity == 1, week.Parity == 2, week.Parity == 3, week.Parity > 3]

lists_to_plot = []


for mask in masks:
    temp_list = week.loc[mask, 'MorningMinutes'].dropna().to_list() + \
        week.loc[mask, 'EveningMinutes'].dropna().to_list()
    lists_to_plot.append(temp_list)


plt.boxplot(lists_to_plot)
plt.ylabel('Minutes')
plt.xlabel('Parity')
plt.xticks([1, 2, 3, 4], ['1', '2', '3', '4+'], fontsize=12)
plt.gca().set_facecolor('whitesmoke')



#%%
df_density = week.dropna()
dim = df_density['DIM'].to_list() * 2
time = df_density['MorningMinutes'].to_list() + df_density['EveningMinutes'].to_list()

sns.kdeplot(x=df_density['DIM'].to_list(), y=df_density['MorningMinutes'].to_list(), cmap="Blues", fill=True, thresh=0, levels=100)
plt.scatter(df_density['DIM'].to_list(), df_density['MorningMinutes'].to_list())

#%%
mean_order = week.groupby('Tag')[
    ['EntryOrderMorning', 'ExitOrderMorning',
      'EntryOrderEvening', 'ExitOrderEvening',
       'Parity', 'DIM']].mean()
mean_order['AvgChange'] = (mean_order['EntryOrderMorning'] + mean_order['EntryOrderEvening'] \
    - mean_order['ExitOrderMorning'] -mean_order['ExitOrderEvening'])*0.5
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