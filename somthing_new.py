# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 02:46:04 2023

@author: folke
"""

import pandas as pd
import os

def columns_to_datetime(df, column_names, time_zone=2):
    for column in column_names:
        df[column] = pd.to_datetime(df[column]+(3.6e6*time_zone), unit='ms').dt.time
    return df



direct = 'data/'
prefix = 'group2'
files = [file for file in os.listdir(direct) if file.startswith(prefix)]


week = pd.DataFrame()

#week = pd.concat([pd.read_csv(os.path.join(direct, file)) for file in files],
                 #ignore_index=True)
for file in files:
    temp = pd.read_csv(os.path.join(direct, file))
    temp['MilkingDate'] = f"{file[15:19]}-{file[19:21]}-{file[21:23]}"
    week = pd.concat([week, temp], ignore_index=True)
    

cow_info = pd.read_csv('data/CowInfo.csv')

week = pd.merge(week, cow_info, on=['Tag', 'MilkingDate'])




week['MorningMinutes'] = (week['ExitMorning']-week['EntryMorning'])*1.666e-5
week['EveningMinutes'] = (week['ExitEvening']-week['EntryEvening'])*1.666e-5

week = columns_to_datetime(week,
                          ['EntryMorning', 'ExitMorning', 
                           'EntryEvening', 'ExitEvening'])


mean_order = week.groupby('Tag')[
    ['EntryOrderMorning', 'ExitOrderMorning',
     'EntryOrderEvening', 'ExitOrderEvening',
     'MorningMinutes', 'EveningMinutes', 'Parity']].mean()
