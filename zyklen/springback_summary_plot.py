import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random

from zyklen.models.distance_springback_diagram_model import DistanceSpringbackDiagramModel
from zyklen.models.springback_model import SpringbackModel


# Plots all springbacks into one distance-springback graph
def all_springbacks_plot(input_directory, output_directory):
    dirs = [x[0] for x in os.walk(input_directory)]

    springback_diagrams = []

    for d in dirs:
        if 'cleared' in d:
            springback_models = []
            parent_path = Path(d).parent
            file_names = os.listdir(f'{d}/')

            print(f'plotting {d}')
            for file_name in file_names:
                df = pd.read_csv(f'{d}/{file_name}', delimiter=';')

                # If dataframe is empty, skip it
                if df.empty:
                    continue

                springback_models.append(calculate_springback_model(df, file_name))

            # If springback_models is empty, do not plot it
            # springback_diagrams.append(plot_all_springbacks(springback_models, output_directory))
            springback_diagrams.append(
                plot_all_springbacks(parent_path.name, springback_models, f'{parent_path}/results/'))

    all_springbacks_consolidated(springback_diagrams, output_directory)


def calculate_springback_model(df, name):
    # Remove all negative values of ['Standardkraft'] and save it in df
    df_without_zero = df[df['Standardkraft'] > 0]

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
    df_after_max = df_without_zero[df_without_zero['Prüfzeit'] > max_force_row['Prüfzeit'].values[0]]

    # Index error if there is no value below 1
    try:
        min_force_row = df_after_max[df_after_max['Standardkraft'] < 1].iloc[0]
    except IndexError:
        min_force_row = df_after_max[df_after_max['Standardkraft'] < 3].iloc[0]

    min_force = min_force_row['Standardkraft']
    min_force_distance = min_force_row['Standardweg']

    # max_distance = max_force_row['Standardweg']
    springback = last_row_distance - min_force_distance

    springback_model = SpringbackModel(name, max_distance, springback)
    return springback_model


def plot_all_springbacks(name, springback_models, output_directory):
    fig, ax1 = plt.subplots()

    # Get all distances of springback_models
    x_distances = [model.distance for model in springback_models]
    # Get all springbacks of springback_models
    y_springbacks = [model.springback for model in springback_models]

    distance_springback_diagram_models = []

    ax1.plot(x_distances, y_springbacks, 'o', color='tab:blue')
    ax1.set_xlabel('Distance [mm]')
    ax1.set_ylabel('Springback [mm]')
    ax1.set_title('Springback of all springback models')

    # Save values in separate csv file
    create_new_csv_with_springback_yp_values(output_directory, x_distances, y_springbacks)

    # Annotate each point with its name
    for i, txt in enumerate([model.name for model in springback_models]):
        ax1.annotate(txt, (x_distances[i], y_springbacks[i]), fontsize=4)

    rounded_distances = [round(elem, 2) for elem in x_distances]
    unique_distances = sorted(list(set(rounded_distances)))

    mean_springbacks = []

    for distance in unique_distances:
        # Get all springbacks with distance distance
        # springbacks_for_distance = [model.distance == distance for model in springback_models]
        # Get all springbacks of springback_models with distance distance
        springbacks_for_distance = [model.springback for model in springback_models if
                                    round(model.distance, 0) == distance]
        mean_springback = mean(springbacks_for_distance)

        mean_springbacks.append(mean_springback)

    axes = plt.gca()
    y_limit = max(y_springbacks) + 0.2
    axes.set_ylim([0, y_limit])

    plt.grid()
    plt.savefig(f'{output_directory}all-springbacks.png', dpi=600)

    return DistanceSpringbackDiagramModel(name, x_distances, y_springbacks)


def all_springbacks_consolidated(springack_diagrams, output_directory):
    fig, ax1 = plt.subplots()

    print(f'plotting {len(springack_diagrams)} diagrams')

    mean_diagrams = []

    for diagram in springack_diagrams:
        r = lambda: random.randint(0, 255)
        color = '#%02X%02X%02X' % (r(), r(), r())

        ax1.scatter(diagram.distances, diagram.springbacks, label=diagram.name, marker='o', color=color)
        ax1.set_xlabel('Distance [mm]')
        ax1.set_ylabel('Springback [mm]')

        mean_diagrams.append(plot_mean_springbacks_consolidated(diagram, output_directory))

    for mean_diagram in mean_diagrams:
        ax1.plot(mean_diagram.distances, mean_diagram.springbacks, color='tab:grey', linestyle='--')

    plt.title('Springback of all springback models')
    plt.legend()
    plt.grid()
    plt.savefig(f'{output_directory}all-springbacks-consolidated.png')


def plot_mean_springbacks_consolidated(sprinback_diagram, output_directory):
    # Get all distances of springback_models
    x_distances = sprinback_diagram.distances
    y_springbacks = sprinback_diagram.springbacks

    mean_springbacks = []

    rounded_distances = [round(elem, 2) for elem in x_distances]
    unique_distances = sorted(list(set(rounded_distances)))

    for distance in unique_distances:
        # Get all springbacks in the springback_diagram with where the distance is distance
        springbacks_for_distance = [springback for springback in y_springbacks if
                                    round(x_distances[y_springbacks.index(springback)], 2) == distance]

        mean_springback = mean(springbacks_for_distance)
        mean_springbacks.append(mean_springback)

    # Create dataframe from unique_distances and mean_springbacks
    df = pd.DataFrame({'distance': unique_distances, 'springback': mean_springbacks})

    return DistanceSpringbackDiagramModel('', unique_distances, mean_springbacks)


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


# Create a csv file where one column is the distance and the other column is the mean springback.
def create_new_csv_with_springback_yp_values(output_directory, x_distances, y_springbacks):
    if len(x_distances) != len(y_springbacks):
        raise Exception('x_distances and y_springbacks must have the same length')

    # Get parent directory of output_directory
    parent_directory = Path(output_directory).parent
    thickness = get_thickness_from_file_name(str(parent_directory.name))
    die_opening = get_die_opening_from_file_name(str(parent_directory.name))

    df = pd.DataFrame(
        {'distance': x_distances, 'springback': y_springbacks, 'thickness': thickness, 'die_opening': die_opening})

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Add csv folder to parent directory
    csv_directory = f'{parent_directory}/csv/'

    if not os.path.exists(csv_directory):
        os.makedirs(csv_directory)

    df.to_csv(f'{csv_directory}springbacks.csv', index=False)


def get_thickness_from_file_name(file_name):
    # Get the thickness from the file name, remove first chart
    thickness = file_name.split('_')[0]
    thickness = thickness[1:]
    # Change "," to "." in thickness
    thickness = thickness.replace(',', '.')
    return thickness


def get_die_opening_from_file_name(file_name):
    # Get the thickness from the file name
    die_opening = file_name.split('_')[1]
    # Remove first char from die_opening
    die_opening = die_opening[1:]

    return die_opening


def consolidate_all_data_into_one_file():
    # Recursivly get all csv files in the csv folder and subfolders
    csv_files = glob.glob('**/csv/*.csv', recursive=True)
    # filter for files that are named springbacks.csv
    csv_files = [file for file in csv_files if 'springbacks.csv' in file]

    # Create a new csv file
    df = pd.DataFrame()

    for csv_file in csv_files:
        # Read csv file
        df_csv = pd.read_csv(csv_file)
        # Add csv file to df
        df = df.append(df_csv)

    # Save df to csv file
    df.to_csv('data/dataset/consolidated.csv', index=False)


if __name__ == '__main__':
    all_springbacks_plot()