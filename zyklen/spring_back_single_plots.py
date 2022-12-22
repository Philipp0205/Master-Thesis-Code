import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Plots all files in the directory as single force-time plots.
def each_tra_to_springback_plot(input_directory, output_directory):
    dirs = [x[0] for x in os.walk(input_directory)]

    for d in dirs:
        if 'cleared' in d:

            # Get all names of folder
            file_names = os.listdir(f'{d}/')

            for name in file_names:
                # Load csv
                # df = pd.read_csv(f'{input_directory}{name}', delimiter=';')
                df = pd.read_csv(f'{d}/{name}', delimiter=';')
                # calculate_springback(df, name, output_directory)
                parent_path = Path(d).parent
                calculate_springback(df, name, f'{parent_path}/results/')


def calculate_springback(df, name, output_directory):
    fig, ax1 = plt.subplots()

    # Remove all negative values of ['Standardkraft'] and save it in df
    # df_without_zero = df[df['Standardkraft'] > 0]

    # Get point where the force is max
    max_force = df['Standardkraft'].max()
    max_force_row = df.loc[df['Standardkraft'] == max_force]

    max_distance = df['Standardweg'].max()
    max_distance_rows = df.loc[round(df['Standardweg'], 2) == round(max_distance, 2)]

    # Get last row of max_distance_rows
    last_row = max_distance_rows.iloc[-1]
    last_row_x = last_row['Prüfzeit']
    last_row_y = last_row['Standardkraft']
    last_row_distance = last_row['Standardweg']

    # Get point where the force is below 1 and after the max force
    df_after_max = df[df['Prüfzeit'] > max_force_row['Prüfzeit'].values[0]]

    min_force_df = df_after_max[df_after_max['Standardkraft'] < 1]

    # If min_force_df is an empty dataframe, then do not plot anything
    if min_force_df.empty:
        return

    min_force_row = min_force_df.iloc[0]
    # min_force_row = df_after_max[df_after_max['Standardkraft'] < 1]
    min_force = min_force_row['Standardkraft']


    x = df['Prüfzeit']
    y = df['Standardkraft']
    y2 = df['Standardweg']

    max_x = max_force_row['Prüfzeit']
    min_x = min_force_row['Prüfzeit']

    max_y = max_force_row['Standardkraft']
    min_y = min_force_row['Standardkraft']

    max_distance = max_force_row['Standardweg']
    min_distance = min_force_row['Standardweg']

    springback = max_force_row['Standardweg'].values[0] - min_force_row['Standardweg']

    # Label the max force point with the springback value
    ax2 = plt.twinx()
    ax1.plot(x, y, label=f'sb: {round(springback, 3)}')
    ax1.scatter(last_row_x, last_row_y, color='green', marker='x')
    ax1.scatter(min_x, min_y, color='red', marker='x')
    ax1.annotate(f'{round(min_force, 2)} N, \n{round(min_distance, 2)} mm', (min_x, min_y))
    ax1.annotate(f'{round(last_row_distance, 2)} mm', (last_row_x, last_row_y))
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Force [N]')
    ax1.legend()

    ax2.set_ylabel('Distance (mm)')
    # ax2.plot(x, y2, color='grey')

    plt.savefig(f'{output_directory}{name}_springback.png', dpi=300)


def plot_time_force(df):
    x = df['Prüfzeit']
    y = df['Standardkraft']

    plt.plot(x, y)
    plt.savefig('pictures/all_single.png')


if __name__ == '__main__':
    each_tra_to_springback_plot()
