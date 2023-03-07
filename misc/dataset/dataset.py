import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import matplotlib
import scienceplots
import matplotlib.patches as mpatch
import numpy as np
import learner.data_preprocessing as preprocessing


def visualize_dataset():
    fig = plt.figure()
    x = np.arange(10)

    # Load conolidated dataset
    # dataset = pd.read_csv('data/dataset/consolidated.csv', delimiter=',')

    dataset = preprocessing.get_data()
    print(dataset.head())

    # Get thickness and die_opening for each row
    thicknesses_and_die_openings = dataset[['thickness', 'die_opening']]

    # Remove duplicate rows from thicknesses_and_die_openings
    thicknesses_and_die_openings = thicknesses_and_die_openings.drop_duplicates()

    # Select samples where die_opening is 30
    test_data = thicknesses_and_die_openings[thicknesses_and_die_openings['die_opening'] == 30]
    train_data = thicknesses_and_die_openings[thicknesses_and_die_openings['die_opening'] != 30]

    x_test = test_data['thickness']
    y_test = test_data['die_opening']

    x_train = train_data['thickness']
    y_train = train_data['die_opening']

    # x = thicknesses_and_die_openings['thickness']
    # y = thicknesses_and_die_openings['die_opening']

    # Plot the dataset
    plt.style.use(['science', 'scatter', 'grid', 'bright'])

    # Plot the dataset with seaborn with diamond markers
    plt.figure(dpi=1000)

    # seaborn.scatterplot(x=x, y=y, marker='D', s=40, color='lightcoral')
    seaborn.scatterplot(x=x_test, y=y_test, marker='D', s=40, color='blue', label='Test')
    seaborn.scatterplot(x=x_train, y=y_train, marker='D', s=40, color='lightcoral', label='Train')

    # rectangle_test = plt.Rectangle((0.4, 26), 2.7, 8, linestyle='dashed', fc='none', ec="red",
    #                                label='Test',
    #                                linewidth=0.8)
    # rectangle_train = plt.Rectangle((0.4, 8.5), 2.7, 17, linestyle='dashed', fc='none', ec="blue",
    #                                 label='Train',
    #                                 linewidth=0.8)
    #
    # rectangle_train2 = plt.Rectangle((0.4, 34.6), 2.7, 16.9, linestyle='dashed', fc='none',
    #                                  ec="blue",
    #                                  linewidth=0.8)

    # Shrink current axis's height by 10% on the bottom
    box = plt.gca().get_position()
    plt.gca().set_position([box.x0, box.y0 + box.height * 0.1,
                            box.width, box.height * 0.9])

    # plt.gca().add_patch(rectangle_test)
    # plt.gca().add_patch(rectangle_train)
    # plt.gca().add_patch(rectangle_train2)

    # q: How to make the legend transparent?
    # a: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
               fancybox=True, ncol=5)

    plt.legend(frameon=False)

    legend = plt.legend()
    legend.get_frame().set_edgecolor('b')
    legend.get_frame().set_linewidth(0.0)

    plt.xlabel('Thickness $t$ [mm]')
    plt.ylabel('Die opening $V$ [mm]')

    root = preprocessing.get_root_directory()

    plt.savefig('results/test_train_split1.png', transparent=True)

    plt.clf()


if __name__ == '__main__':
    visualize_dataset()
