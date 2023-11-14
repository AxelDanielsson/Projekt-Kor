
from no_mess import entry_exit, df_milk_1, tags_g2, start_day



entry_times, exit_times = entry_exit(df_milk_1, tags_g2, start_day)


entrytags = list(entry_times.keys())
exittags = list(exit_times.keys())
match = 0
nomatch = 0
i = 0
spann = 20

for tag in entrytags:
    if spann == 0:
        if tag == exittags[i]:
            match += 1
        else:
            nomatch += 1
    if spann != 0:
        if i < spann:
            if tag in exittags[0:i + spann]:
                match += 1
            else:
                nomatch += 1
        if i >= spann:
            if i + spann > len(exittags):
                if tag in exittags[i - spann:i]:
                    match += 1
                else:
                    nomatch += 1
            else:
                if tag in exittags[i - spann:i + spann]:
                    match += 1
                else:
                    nomatch += 1

    i += 1

print('matches ', match)
print('not matching ', nomatch)