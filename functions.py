####################################################
# Functions from other project used in our project #
####################################################

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as pat


# function from project cowview.data
# reads csv-file
def csv_read_FA(filename, nrows):
    if nrows == 0:
        df = pd.read_csv(filename, header=None)
    else:
        df = pd.read_csv(filename, nrows=nrows, header=None)
    df.columns = ['data_entity', 'tag_id', 'tag_string', 'time', 'x', 'y', 'z']
    return df


# function from project cowview.plot
# function to plot the outline of the barn
def plot_barn(filename):
    df = pd.read_csv(filename, skiprows=0, sep=';', header=0)
    df.columns = ['Unit', 'x1', 'x2', 'x3', 'x4', 'y1', 'y2', 'y3', 'y4']
    units = list(df['Unit'])
    x_1 = list(df['x1'])
    x_2 = list(df['x2'])
    x_3 = list(df['x3'])
    x_4 = list(df['x4'])
    y_1 = list(df['y1'])
    y_2 = list(df['y2'])
    y_3 = list(df['y3'])
    y_4 = list(df['y4'])

    fig, ax = plt.subplots(1, figsize=(6, 6))
    for i in range(len(units)):
        art = pat.Rectangle((x_1[i], min(y_1[i], y_2[i])), x_3[i] - x_1[i], max(y_1[i], y_2[i]) - min(y_1[i], y_2[i]), fill=False)
        ax.add_patch(art)
        # print(ax.patches)
    ax.set_xlim(x_1[0] - 2000, x_3[0] + 2000)
    ax.set_ylim(y_1[0] - 2000, y_2[0] + 2000)
    return fig, ax


# function from project cowview.extras
# extract position data from dataframe
def positions(df):
    x = list(df['x'])
    y = list(df['y'])
    z = list(df['z'])
    return x,y,z

# function from project cowview.plot
# function to plot the position of a cow (based on tag_id) for FA-data
def plot_cow(df, tag_id, filename_barn):
    fig, ax = plot_barn(filename_barn)
    if hasattr(tag_id, "__len__"):
        for i in tag_id:
            temp = df.loc[df['tag_id'] == i]
            x,y,z = positions(temp)
            plt.plot(x,y,'o--', markersize = 2)
    else:
        temp = df.loc[df['tag_id'] == tag_id]
        x,y,z = positions(temp)
        plt.plot(x,y,'o--', markersize = 2)
    plt.show()


# function from project cowview.animation
def animate_cows(df, cowID_1, cowID_2, barn_filename, save_path='n'):

    cow_1 = df.loc[df['tag_id'] == cowID_1]

    cow_2 = df.loc[df['tag_id'] == cowID_2]

    x1, y1, z1 = positions(cow_1)
    x2, y2, z2 = positions(cow_2)

    time1 = list(cow_1['time'])
    time2 = list(cow_2['time'])

    timestrings1= []
    timestrings2 = []

    for timestamp1 in time1:
        timestrings1.append(datetime.fromtimestamp((timestamp1/1000)-7200))

    for timestamp2 in time2:
        timestrings2.append(datetime.fromtimestamp((timestamp2 / 1000) - 7200))

    f, ax1 = plot_barn(barn_filename)


    ax1.change_geometry(2, 1, 1)
    ax2 = f.add_subplot(212)

    plt.tight_layout()

    pos1 = ax1.get_position().bounds
    pos2 = ax2.get_position().bounds

    new_pos1 = [pos1[0], 0.25, pos1[2], 0.7]
    new_pos2 = [pos2[0], pos2[1], pos2[2], 0.1]

    ax1.set_position(new_pos1)
    ax2.set_position(new_pos2)

    xdata1, ydata1 = [], []
    ln1, = ax1.plot([], [], '-')
    xdata2, ydata2 = [], []
    ln2, = ax1.plot([], [], '-')

    d1, = ax1.plot([], [], 'co', label='Cow '+ str(cowID_1))
    d2, = ax1.plot([], [], 'yo', label='Cow '+ str(cowID_2))

    ax1.legend(loc='upper left')

    dist, time = [], []
    dist_plot, = ax2.plot([], [], 'r-')

    def run_animation():
        ani_running = True
        i = 0
        j = 0
        def onClick(event): #If the window is clicked, the gif pauses
            nonlocal ani_running
            if ani_running:
                ani.event_source.stop()
                ani_running = False
            else:
                ani.event_source.start()
                ani_running = True

        def init():
            ax1.set_xlim(0, 3340)
            ax1.set_ylim(0, 8738)
            ax2.set_xlim(timestrings1[0], timestrings1[len(timestrings1)-1])

            ax2.set_ylim(0, 10000)
            date1 = timestrings1[0]
            date2 = timestrings1[len(timestrings1)-1]
            ax1.set_title("Plot of two cows between " + date1.strftime("%d %b %Y %H:%M") + " - " +
                          date2.strftime("%d %b %Y %H:%M"), fontsize=8)
            ax2.set_ylabel('Distance(cm)')
            ax2.set_xlabel('Time of day')

            return ln1, ln2, d1, d2, dist_plot

        def update(frame):
            nonlocal i
            nonlocal j
            if not pause:
                if time1[i] <= time2[j]:
                    if i == len(time1) - 1:  # if at end of times_1
                        j = j + 1
                        xdata2.append(x2[j])  # new distance
                        ydata2.append(y2[j])
                        xdata1.append(x1[i])  # new distance
                        ydata1.append(y1[i])
                    else:
                        i = i + 1
                        xdata1.append(x1[i])  # new distance
                        ydata1.append(y1[i])
                        xdata2.append(x2[j])  # new distance
                        ydata2.append(y2[j])
                else:
                    if j == len(time2) - 1:  # if at end of times_2
                        i = i + 1
                        xdata1.append(x1[i])  # new distance
                        ydata1.append(y1[i])
                        xdata2.append(x2[j])  # new distance
                        ydata2.append(y2[j])
                    else:
                        j = j + 1
                        xdata2.append(x2[j])  # new distance
                        ydata2.append(y2[j])
                        xdata1.append(x1[i])  # new distance
                        ydata1.append(y1[i])

                ln1.set_data(xdata1, ydata1) #Uppdate the plot with the data
                d1.set_data(x1[i], y1[i])
                ln2.set_data(xdata2, ydata2)
                d2.set_data(x2[j], y2[j])
                dist.append(math.sqrt(math.pow(x1[i]-x2[j], 2) + math.pow(y1[i]-y2[j], 2)))

                if time1[i]<time2[j]: #Uppdate the correct time
                    time.append((timestrings1[i]))
                else:
                    time.append((timestrings2[j]))


                dist_plot.set_data(time, dist)

            return ln1, ln2, d1, d2, dist_plot

        f.canvas.mpl_connect('button_press_event', onClick)
        ani = FuncAnimation(f, update, frames=len(time1)+len(time2)-2, init_func=init, blit=True, interval=1, repeat=False) #Main animationfunction
        if save_path != 'n': #If a filename is given, the gif is saved
            try:
                ani.save(save_path)
            except:
                print('Wrong filepath')

        plt.show()

    run_animation()

# Function compares the placement of cows in the entry que with the exit que.
# It takes 2 dict of tags with entry and exit times. The span determine how many cows in front of or behind in
# the exit que the entry placement is compared with
def compare_entry_exit(entrytags, exittags, span):
    match = 0
    nomatch = 0
    i = 0
    for tag in entrytags:
        if span == 0:
            if tag == exittags[i]:
                match += 1
            else:
                nomatch += 1
        if span != 0:
            if i < span:
                if tag in exittags[0:i + span]:
                    match += 1
                else:
                    nomatch += 1
            if i >= span:
                if i + span > len(exittags):
                    if tag in exittags[i - span:i]:
                        match += 1
                    else:
                        nomatch += 1
                else:
                    if tag in exittags[i - span:i + span]:
                        match += 1
                    else:
                        nomatch += 1
        i += 1
    print('matches with span ', span, ':', match)
    print('not matching with span ', span, ':', nomatch)
