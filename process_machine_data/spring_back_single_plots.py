import os
from pathlib import Path

import matplotlib
import learner.data_preprocessing as pp

from process_machine_data.springback_summary_plot import calculate_springback_model

import pandas as pd
import matplotlib.pyplot as plt
# project constants
import misc.project_constants as pc


# import science plots

# Plots all files in the directory as single force-time plots.
def each_tra_to_springback_plot(input_directory, output_directory):
    # Get all directories in the input directory
    dirs = [x[0] for x in os.walk(input_directory)]

    springback_models = []

    # Iterate through all directories and look for cleared directory
    for d in dirs:
        if pc.cleared_directory in d:

            # Get all names of folder
            file_names = os.listdir(f'{d}/')

            for name in file_names:
                # Load csv
                df = pd.read_csv(f'{d}/{name}', delimiter=';')
                # calculate_springback(df, name, output_directory)
                parent_path = Path(d).parent
                # calculate_springback(df, name, f'{parent_path}/{pc.results_directory}/')
                model = calculate_spring_back_2(df, name,
                                                f'{parent_path}/{pc.results_directory}/')

                springback_models.append(model)

    # All springbacksmodels to one csv
    springback_models_df = pd.DataFrame(springback_models)

    springback_models_df.to_csv(f'{output_directory}/springback_models.csv', index=False)

    return springback_models


# Calculates the springback for a single file
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

    # Get first row where force is below 1
    # !!!
    min_force_df = df_after_max[df_after_max['Standardkraft'] < 1.8]

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
    ax1.annotate(f'{round(min_force, 2)} N, \n{round(min_distance, 2)} mm',
                 (min_x, min_y))
    ax1.annotate(f'{round(last_row_distance, 2)} mm', (last_row_x, last_row_y))
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Force [N]')
    ax1.legend()

    ax2.set_ylabel('Distance (mm)')
    # ax2.plot(x, y2, color='grey')
    plt.title(name)

    plt.savefig(f'{output_directory}{name}_springback.png', dpi=400, transparent=True)
    plt.clf()


def calculate_spring_back_2(df, name, output_directory):
    fig, ax1 = plt.subplots()

    # plt.style.use(['science', 'bright'])

    spring_back_model = calculate_springback_model(df, name)
    spring_back = spring_back_model.spring_back

    # Remove rows from df where 'Prüfzeit' is 0
    df = df[df['Prüfzeit'] >= 0]

    x = df['Prüfzeit']
    y = df['Standardkraft']
    y2 = df['Standardweg']

    first_point = spring_back_model.start_point
    last_point = spring_back_model.end_point

    first_point_x = first_point['Prüfzeit']
    first_point_y = first_point['Standardkraft']

    last_point_x = last_point['Prüfzeit']
    last_point_y = last_point['Standardkraft']

    # Label the max force point with the springback value
    ax2 = plt.twinx()
    l1 = ax1.plot(x, y, label='Force')
    # label=f'sb: {round(spring_back, 3)} threshold: '
    #       f'{round(spring_back_model.force_threshold, 1)}'
    #        )

    ax1.scatter(first_point_x, first_point_y, color='green', marker='x')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Force [N]')

    ax2.set_ylabel('Punch travel [mm]')
    l2 = ax2.plot(x, y2, color='grey', label='punch travel')
    ax1.scatter(last_point_x, last_point_y, color='red', marker='x')

    ax1.legend(handles=l1 + l2)
    # ax1.set_title(name)
    ax2.set_title(f'{name} sb: {round(spring_back, 3)}')

    plt.savefig(f'{output_directory}{name}_springback.png', dpi=600, transparent=True)
    plt.clf()

    matplotlib.pyplot.close('all')

    return spring_back_model

def plot_time_force(df):
    x = df['Prüfzeit']
    y = df['Standardkraft']

    plt.plot(x, y)
    plt.savefig('pictures/all_single.png')


if __name__ == '__main__':
    root_directory = pp.get_root_directory()

    data_directory = f'{root_directory}/data/dataset/dip_test/'
    output_directory = f'{data_directory}results/'

    each_tra_to_springback_plot(data_directory, output_directory)
