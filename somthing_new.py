# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 02:46:04 2023

@author: folke
"""

import pandas as pd
import os

def columns_to_datetime(df, column_names, time_zone=2):
    for column in column_names:
        df[column] = pd.to_datetime(df[column]+(3.6e6*time_zone), unit='ms')
    return df



direct = 'data/'
prefix = 'group1'
files = [file for file in os.listdir(direct) if file.startswith(prefix)]

week = pd.DataFrame()

week = pd.concat([pd.read_csv(os.path.join(direct, file)) for file in files],
                 ignore_index=True)

week['MorningMinutes'] = (week['ExitMorning']-week['EntryMorning'])*1.666e-5
week['EveningMinutes'] = (week['ExitEvening']-week['EntryEvening'])*1.666e-5

week = columns_to_datetime(week,
                          ['EntryMorning', 'ExitMorning', 
                           'EntryEvening', 'ExitEvening'])


mean_order = week.groupby('TagId')[
    ['EntryOrderMorning', 'ExitOrderMorning',
     'EntryOrderEvening', 'ExitOrderEvening']].mean()
