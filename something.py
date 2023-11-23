# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:04:09 2023

@author: folke
"""

from something_else import first_entry, entry_exit, milk_window, divide_groups,\
    positions, remove_stationary_tags, get_number_in_order, summary_dataframe
from pycowview.data import csv_read_FA
import os
import pandas as pd
from time import perf_counter


def main():
    start = perf_counter()
    direct = 'data/'
    prefix = 'FA_'
    files = [file for file in os.listdir(direct) if file.startswith(prefix)]
    for file in files:
    
    
        nrows = 0
        df = csv_read_FA(os.path.join(direct, file), nrows)
        df, _ = remove_stationary_tags(df)
        df_group1, df_group2, _, _ = divide_groups(df)
        
        area_dict_group1 = {
            'upper_left': (1670, 2482.5, 5851.5, 8738),
            'upper_right': (2482.5, 3340, 5851.5, 8738),
            'middle_left': (1670, 2482.5, 3242.5, 5851.5),
            'middle_right': (2482.5, 3340, 3242.5, 5851.5),
            'lower' : (1670, 3340, 2100, 3242.5)
        }
        area_dict_group2 = {
            'upper_left': (0, 881, 5851.5, 8738),
            'upper_right': (881, 1670, 5851.5, 8738),
            'middle_middle': (0, 881, 3242.5, 5851.5),
            'middle_right': (881, 1670, 3242.5, 5851.5),
            'lower' : (0, 1670, 2100, 3242.5)
        }
                  
        group1_df = summary_dataframe(df_group1, area_dict_group1)
        group2_df = summary_dataframe(df_group2, area_dict_group2)
        group1_df.to_csv(f"data/group1_summary_{file[3:11]}.csv")
        group2_df.to_csv(f"data/group2_summary_{file[3:11]}.csv")
        print(f"finished {file} after {perf_counter()-start} s")
        
        
if __name__ == "__main__":
    main() 