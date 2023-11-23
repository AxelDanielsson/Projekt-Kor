from scipy.stats import spearmanr
from pandas import read_csv

df = read_csv('data/group1_summary_20200919.csv')

entry_morning = df['EntryMorning'].tolist()
exit_morning = df['ExitMorning'].tolist()

spearMorning = spearmanr(entry_morning,exit_morning)
print(spearMorning)