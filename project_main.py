from process_machine_data.spring_back_single_plots import each_tra_to_springback_plot

from process_machine_data.springback_summary_plot import \
    consolidate_all_data_into_one_file, all_springbacks_plot

import learner.data_preprocessing as dp
from process_machine_data.tra_to_csv import iterate_through_folder_and_convert_tra_to_csv

if __name__ == '__main__':
    # tra_input_directory = 'data/cycle_vs_single_test/single_measurements/messreihe_3
    # /tra/'
    # tra_output_directory = 'data/cycle_vs_single_test/single_measurements/messreihe_3
    # /cleared/'
    # springback_output_directory =
    # 'data/cycle_vs_single_test/single_measurements/messreihe_3/results/'

    root = dp.get_root_directory()
    dataset_folder = f'{root}/data/dataset/V30/'

    parent_folder = f'{root}/data/dataset/V30/'

    tra_input_directory = f'{parent_folder}tra/'
    tra_output_directory = f'{parent_folder}cleared/'
    springback_output_directory = f'{parent_folder}results/'
    # springback_output_directory = f'data/results/'

    # iterate_through_folder_and_convert_tra_to_csv(parent_folder)
    each_tra_to_springback_plot(parent_folder, springback_output_directory)
    # all_springbacks_plot(parent_folder, springback_output_directory)

    # consolidate_all_data_into_one_file()
    print('Done!')
