import glob

import pandas as pd
import learner.data_preprocessing as pp


def consolidate_all_data_into_one_file(output_directory):
    # Recursivly get all csv files in the csv folder and subfolders
    csv_files = glob.glob(f'{output_directory}/**/*.csv', recursive=True)
    # filter for files that are named springbacks.csv
    csv_files = [file for file in csv_files if 'springbacks.csv' in file]

    # Create a new csv file
    df = pd.DataFrame()

    for csv_file in csv_files:
        # Read csv file
        df_csv = pd.read_csv(csv_file)
        # Add csv file to df
        # Concat df with df_csv
        df = pd.concat([df, df_csv])


        # df = df.append(df_csv)

    # Save df to csv file
    df.to_csv(f'{output_directory}/consolidated.csv', index=False)


if __name__ == '__main__':
    root_directory = pp.root_directory()
    output_directory = f'{root_directory}/data/dataset'
    consolidate_all_data_into_one_file(output_directory)
