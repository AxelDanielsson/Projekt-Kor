# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 02:46:04 2023

@author: folke
"""

import pandas as pd
import os

direct = 'data/'
prefix = 'group1'
files = [file for file in os.listdir(direct) if file.startswith(prefix)]

week = pd.DataFrame()

week = pd.concat([pd.read_csv(os.path.join(direct, file)) for file in files],
                 ignore_index=True)



mean_order = week.groupby('TagId')[
    ['EntryOrderMorning', 'ExitOrderMorning',
     'EntryOrderEvening', 'ExitOrderEvening']].mean()