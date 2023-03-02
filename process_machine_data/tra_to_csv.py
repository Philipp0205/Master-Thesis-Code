import os
from pathlib import Path

import pandas as pd
import learner.data_preprocessing as pp


def tra_to_csv(input_directory, output_directory):
    file_names = os.listdir(input_directory)

    for file_name in file_names:
        # Read file and delete the first 163 lines (they are not needed) and save acii
        # encoding western european
        with open(f'{input_directory}{file_name}', 'r', encoding='cp1252') as f:
            lines = f.readlines()
            lines = lines[163:]
            # Create dataframe from lines split by ';'
            df = pd.DataFrame([line.split(';') for line in lines])
            # Remove second row from dataframe (it contains the units)
            df = df.drop(df.index[1])
            # Replace ',' with '.' in dataframe
            df = df.replace(',', '.', regex=True)
            # Save dataframe to csv file with ';' as delimiter and '.' as decimal
            # without column numbers
            df.to_csv(f'{output_directory}/{file_name}.csv', sep=';', decimal='.',
                      index=False)

            print(f'{output_directory}/{file_name}.csv')

        # Open file and delete the first row (it contains the units) save it again
        # Workaround because pandas add a row ....
        with open(f'{output_directory}/{file_name}.csv', 'r') as f:
            lines = f.readlines()
            lines = lines[1:]
            # Remove '"' from lines
            lines = [line.replace('"', '') for line in lines]
            # Delete empty lines
            lines = [line for line in lines if line != '\n']
            with open(f'{output_directory}/{file_name}.csv', 'w') as f:
                f.writelines(lines)


def iterate_through_folder_and_convert_tra_to_csv(input_folder):
    dirs = [x[0] for x in os.walk(input_folder)]
    for d in dirs:
        if 'tra' in d:
            path = Path(f'{d}')

            input_dir = f'{d}/'
            cleared_dir = f'{path.parent}/cleared'

            tra_to_csv(input_dir, cleared_dir)


if __name__ == '__main__':
    root_dir = pp.root_directory()
    input_folder = f'{root_dir}/data/dataset/dip_test/'

    iterate_through_folder_and_convert_tra_to_csv(input_folder)
