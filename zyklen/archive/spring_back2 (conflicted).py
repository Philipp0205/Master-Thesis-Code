import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_all_single_bends():
    # Get all names of folder
    file_names = os.listdir('../data/cycle_vs_single_test/single_measurements/cleared')

    for name in file_names:
        # Load csv
        df = pd.read_csv('../data/cycle_vs_single_test/single_measurements/cleared/' + name, delimiter=';')
        calculate_springback(df, name)


def calculate_springback(df, name):
    # Get point where the force is max
    max_force = df['Standardkraft'].max()
    max_force_row = df.loc[df['Standardkraft'] == max_force]

    # Get point where the force is min and after the max force
    min_force = df.loc[(df['Standardkraft'] < 0) & (df['Prüfzeit'] > max_force_row['Prüfzeit'].values[0])]
    x = df['Prüfzeit']
    y = df['Standardkraft']

    plt.plot(x, y)
    plt.scatter(max_force_row['Prüfzeit'], max_force_row['Standardkraft'], color='red', marker='x')
    plt.scatter(min_force['Prüfzeit'], min_force['Standardkraft'], color='red', marker='x')
    plt.savefig(f'pictures/all_single_bends_springback/{name}_springback.png')
    # clear plot
    plt.clf()




def plot_time_force(df):
    x = df['Prüfzeit']
    y = df['Standardkraft']

    plt.plot(x, y)
    plt.savefig('pictures/all_single.png')


if __name__ == '__main__':
    plot_all_single_bends()




