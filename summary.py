# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 02:46:04 2023

@author: folke
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import ttest_1samp
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

def week_summary(direct, prefix):
    files = [file for file in os.listdir(direct) if file.startswith(prefix)]
    
    
    if prefix == 'group1':
        area_dict = {
            'Upper left': (1670, 2482.5, 5851.5, 8738),
            'Upper right': (2482.5, 3340, 5851.5, 8738),
            'Middle left': (1670, 2482.5, 3242.5, 5851.5),
            'Middle right': (2482.5, 3340, 3242.5, 5851.5),
            'Lower' : (1670, 3340, 2595, 3242.5)
        }
        
        area_colors = {
            'Upper right': 'green',
            'Upper left': 'blue',
            'Middle right': 'purple',
            'Middle left': 'red',
            'Lower': 'orange'
        }
    else:
        area_dict = {
            'Upper right': (881, 1670, 5851.5, 8738),
            'Upper left': (0, 881, 5851.5, 8738),
            'Middle right': (881, 1670, 3242.5, 5851.5),
            'Middle left': (0, 881, 3242.5, 5851.5),
            
            'Lower': (0, 1670, 2595, 3242.5)
        }
        area_colors = {
            'Upper right': 'blue',
            'Upper left': 'green',
            'Middle right': 'red',
            'Middle left': 'purple',
            'Lower': 'orange'
        }
    
    
    week = pd.DataFrame()
    
    
    for file in files:
        temp = pd.read_csv(os.path.join(direct, file))
        #date to merge with CowInfo
        temp['MilkingDate'] = f"{file[15:19]}-{file[19:21]}-{file[21:23]}"
        
        temp = order(temp, 'ExitEvening', 'ExitOrderEvening', 'PositionEvening')    
        temp = order(temp, 'EntryEvening', 'EntryOrderEvening', 'PositionEvening')    
        temp = order(temp, 'ExitMorning', 'ExitOrderMorning', 'PositionMorning')    
        temp = order(temp, 'EntryMorning', 'EntryOrderMorning', 'PositionMorning')
        
        
        temp['EntryDiff'] = [[abs(value - other_values) for other_values in 
                              temp['EntryMorning']] for value in temp['EntryMorning']]
        temp['ExitDiff'] = [[abs(value - other_values) for other_values in 
                              temp['ExitMorning']] for value in temp['ExitMorning']]
        
        
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
        
        
        
        
        
        #add day to week
        week = pd.concat([week, temp], ignore_index=True)
        
    #merge with CowInfo
    tag_conv = pd.read_csv('data/tag_conv.csv')
    cow_info = pd.read_csv('data/CowInfo.csv')
    week = pd.merge(week, tag_conv, on='tag_id')
    week = pd.merge(week, cow_info, left_on=['tag_string', 'MilkingDate'], right_on=['Tag', 'MilkingDate'])
    week['Parity'] = week['Parity'].apply(lambda x: 3 if x > 3 else x)
    week['DIM'] = week['DIM'].apply(lambda x: 'Early' if x < 50 else ('Late' if x > 149 else 'Mid'))
    lower_mask = week['PositionMorning'].isin(['Lower left', 'Lower right'])
    week.loc[lower_mask, 'PositionMorning'] = 'Lower'
    
    lower_mask = week['PositionEvening'].isin(['Lower left', 'Lower right'])
    week.loc[lower_mask, 'PositionEvening'] = 'Lower'
    
    return week, area_dict, area_colors

direct = 'data/'
prefix = 'group2'
week, area_dict, area_colors = week_summary(direct, prefix)

#%%
def ttest(week, area):
    week=week.dropna()
    t, p = ttest_1samp(week.loc[week['PositionEvening'] == area]['SecondsAfterEvening'],
            week['SecondsAfterEvening'].mean())
    return t, p
tvalue, pvalue = ttest(week, 'Lower')
#%%   
# Time after entry violin plot area
def violin_entry_area(week, session, area_dict, area_colors, prefix):
    lists_to_plot = []
    keys = list(area_dict.keys())
    for key in keys:
        if session == 'morning':
            temp_list = [time for time in week.loc[week['PositionMorning'] == key]['SecondsAfterMorning'].apply(lambda x: x/60).to_list() if time < 35]
        else:
            temp_list = [time for time in week.loc[week['PositionEvening'] == key]['SecondsAfterEvening'].apply(lambda x: x/60).to_list() if time < 35]
        lists_to_plot.append(temp_list)
    
    ax = sns.violinplot(data=lists_to_plot, palette=area_colors.values(), cut=0, bw=0.2)
    
    
    for i, artist in enumerate(ax.collections):
        # Calculate the number of samples in each category
        total_samples = len(lists_to_plot[i])
        
        # Display the sample count above the violin plot
        ax.text(i+0.09, ax.get_ylim()[1]-0.4, f'n={total_samples}', ha='left', va='center')
    
    plt.title(f"Time passed after the first cow enters, {session} {prefix[:-1]} {prefix[-1]}", fontsize=14)
    plt.ylabel('Minutes')
    plt.xlabel('Part of the barn')
    plt.xticks([0, 1, 2, 3, 4], area_colors.keys(), fontsize=10)
    plt.gca().set_facecolor('whitesmoke')
    plt.show()
    
violin_entry_area(week, 'evening', area_dict, area_colors)


#%%
def avg_change_plot(week, prefix):
    agg_func = {
        'EntryOrderMorning': 'mean',
        'ExitOrderMorning': 'mean',
        'EntryOrderEvening': 'mean',
        'ExitOrderEvening': 'mean',
        'Parity': 'first', 
        'DIM': 'first'
        }
    filtered_week = week.dropna().copy()
    filtered_week = filtered_week.groupby('tag_id').filter(lambda x: len(x) == 7)
    custom_palette = sns.color_palette("colorblind")
    
    mean_order = filtered_week.groupby('tag_id').agg(agg_func)
    mean_order['AvgChange'] = (-mean_order['EntryOrderMorning'] -mean_order['EntryOrderEvening'] \
        + mean_order['ExitOrderMorning'] +mean_order['ExitOrderEvening'])*0.5
    hue_order = ['Early', 'Mid', 'Late']  
    sns.swarmplot(data=mean_order, x='Parity', y='AvgChange', hue='DIM', 
                  palette=custom_palette, hue_order=hue_order)
    plt.title(f"Average change in queue group {prefix[-1]}", fontsize=18)
    plt.xticks([0,1,2], ['1', '2', '3+'])
    plt.show()
avg_change_plot(week, prefix)


#%% 
def order_parity_plot(week, prefix, area_dict):
    area_times = {key: [] for key in area_dict.keys()}
    for key in area_times:
        morning_data = week.loc[week['PositionMorning'] == key]
        evening_data = week.loc[week['PositionEvening'] == key]
        
        concatenated_data = pd.concat([morning_data[['EntryOrderMorning', 'Parity']],
                                       evening_data[['EntryOrderEvening', 'Parity']]], 
                                      ignore_index=True)
        concatenated_data['Entry Order'] = concatenated_data['EntryOrderMorning'].fillna(concatenated_data['EntryOrderEvening'])
        area_times[key] = concatenated_data[['Entry Order', 'Parity']]
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    fig.delaxes(axes[1, 2])
    
    axes = axes.flatten()
    
    
    for i, (key, ax) in enumerate(zip(area_times.keys(), axes)):
        sns.violinplot(data=area_times[key], x='Parity', y='Entry Order',
                       ax=ax, cut=0, hue='Parity', palette='colorblind', bw_method=0.2,
                       legend=False)
        ax.set_title(f'{key}', fontsize=18)
        ax.set_xticklabels(['1', '2', '3+'], fontsize=16)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
        ax.set_xlabel(ax.get_xlabel(), fontsize=14)
        ax.set_ylabel(ax.get_ylabel(), fontsize=14)
        for j, value in enumerate(list(week['Parity'].unique())):
            num_samples = len(area_times[key].loc[area_times[key]['Parity'] == value])
            ax.text(j, ax.get_ylim()[1]*1.1, f'n={num_samples}', ha='left', va='bottom', fontdict={'size': 14})
    
    fig.suptitle(f"Entry order for different parities from each area group {prefix[-1]}", fontsize=28)
    plt.tight_layout()
    plt.show()
order_parity_plot(week, prefix, area_dict)
#%%

def entry_exit_scatter(week, prefix):
    sns.scatterplot(data=week, x='EntryOrderMorning', y='ExitOrderMorning',
                    hue='Parity', palette='colorblind', alpha=1)
    plt.title(f"Entry and Exit Order Evenings for Group {prefix[-1]}", fontsize=16)
    legend_labels = {1: '1', 2: '2', 3: '3+'}  
    handles = plt.gca().get_legend().legendHandles
    plt.legend(handles, legend_labels.values(), title='Parity')
    plt.xlabel('Entry order', fontsize=14)
    plt.ylabel('Exit order', fontsize=14)
    plt.show()




#%%
def cows_barplot(week, prefix):
    week_copy = week.sort_values(by='MilkingDate').copy()
    order = ['Early', 'Mid', 'Late']
    sns.countplot(data=week_copy, x='MilkingDate', hue='DIM', hue_order=order,
                  palette='colorblind')
    plt.title(f'Cows in queue group {prefix[-1]}', fontsize=16)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks([0, 20, 40, 60, 80, 100])
    plt.ylabel('Number of cows', fontsize=14)
    plt.xlabel('Date')
    plt.legend()
    sns.set_theme()
    plt.show()
    
    
cows_barplot(week, prefix)
#%%
def nearby_heatmap(week, prefix):
    agg_func = {
        'AllNearby': 'sum',
        'Parity': 'first',
        }
    
    filtered_week = week.dropna().copy()
    filtered_week = filtered_week.groupby('tag_id').filter(lambda x: len(x) == 7)
    filtered_week['AllNearby'] = filtered_week['NearbyMorningEntry'] + filtered_week['NearbyMorningExit'] + \
        filtered_week['NearbyEveningEntry'] + filtered_week['NearbyEveningExit']
    
    tags = filtered_week['tag_id'].unique()
    nearby_sum = pd.DataFrame(filtered_week.groupby('tag_id').agg(agg_func).reset_index())
    
    nearby_sum_exploded = nearby_sum.explode('AllNearby')
    heatmap_data = pd.crosstab(nearby_sum_exploded['tag_id'], nearby_sum_exploded['AllNearby']).fillna(0)
    heatmap_data = heatmap_data[heatmap_data.columns.intersection(tags)]
    
    
    for i in range(len(tags)):
        heatmap_data.iloc[i, i] = None
        
    
        
    sns.heatmap(data=heatmap_data)
    
    plt.title(f'Times Pairs Cows Entered or Exited Together group {prefix[-1]}')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.show()

nearby_heatmap(week, prefix)





