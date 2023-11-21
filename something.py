# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:04:09 2023

@author: folke
"""

from something_else import first_entry, entry_exit, milk_window, divide_groups,\
    positions, remove_stationary_tags
from pycowview.data import csv_read_FA
import os
import pandas as pd


def main():
    
    direct = 'data/'
    prefix = 'FA_'
    files = [file for file in os.listdir(direct) if file.startswith(prefix)]
    
    for file in files:
    
    
        nrows = 0
        df = csv_read_FA(os.path.join(direct, file), nrows)
        df, _ = remove_stationary_tags(df)
        df_group1, df_group2, _, _ = divide_groups(df)
          
        group1_morning, group1_evening = milk_window(df_group1)
        group2_morning, group2_evening = milk_window(df_group2)
          
          
        group1_entry_morning, group1_exit_morning = entry_exit(group1_morning)
        group1_entry_evening, group1_exit_evening = entry_exit(group1_evening)
          
          
        # group2_entry_morning, group2_exit_evening = entry_exit(group2_morning)
        # group2_entry_evening, group2_exit_evining = entry_exit(group2_evening)
        
        
        group1_start_morning = first_entry(group1_entry_morning)
        group1_start_evening = first_entry(group1_entry_evening)
        # group2_start_morning = first_entry(group2_entry_morning)
        # group2_start_evening = first_entry(group2_entry_evening)
        
        group1_pos_morning = positions(df, group1_start_morning)
        group1_pos_evening = positions(df, group1_start_evening)
        
        # group2_pos_morning = positions(df, group2_start_morning)
        # group2_pos_evening = positions(df, group2_start_evening)
        
        
        common_keys_group1 = group1_entry_morning.keys() \
            & group1_entry_evening.keys()
        group1_dict = {
            key: (
        group1_entry_morning[key],
        group1_exit_morning[key],
        group1_entry_evening[key],
        group1_exit_evening[key],
        group1_pos_morning[key],
        group1_pos_evening[key]
        )
        for key in common_keys_group1
        }

        
        group1_df = pd.DataFrame.from_dict(group1_dict, orient='index',
                        columns=['EntryMorning', 'ExitMorning',
                                 'EntryEvening', 'ExitEvening',
                                 'PositionMorning', 'PositionEvening'])
        group1_df.index.name = 'TagId'
        group1_df.to_csv(f"data/group1_order_{file[3:11]}.csv")
        print(f"finished {file}")
        
if __name__ == "__main__":
    main() 