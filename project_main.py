import learner.data_preprocessing as preprocessing

from learner.logistic_regression import logistic_regression
from learner.reports.reports_main import create_reports

from zyklen.springback_summary_plot import consolidate_all_data_into_one_file



if __name__ == '__main__':
    # tra_input_directory = 'data/cycle_vs_single_test/single_measurements/messreihe_3/tra/'
    # tra_output_directory = 'data/cycle_vs_single_test/single_measurements/messreihe_3/cleared/'
    # springback_output_directory = 'data/cycle_vs_single_test/single_measurements/messreihe_3/results/'

    parent_folder = 'data/dataset/V50/'

    tra_input_directory = f'{parent_folder}tra/'
    tra_output_directory = f'{parent_folder}cleared/'
    springback_output_directory = f'{parent_folder}results/'
    # springback_output_directory = f'data/results/'

    # iterate_through_folder_and_convert_tra_to_csv(parent_folder)
    # each_tra_to_springback_plot(parent_folder, springback_output_directory)
    # all_springbacks_plot(parent_folder, springback_output_directory)

    consolidate_all_data_into_one_file()

    # visualize_dataset()

    print('Done!')
