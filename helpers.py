import argparse
import csv

import numpy as np
import pandas as pd


def check_null_column(col, percentage=80):
    """
    Checks if a column has null values above the percentage
    """
    col_size = len(col)
    null_values = 0
    for value in col:
        if value is None or value == 'na':
            null_values += 1
    if (null_values / col_size) * 100 >= percentage:
        return True
    return False


def check_same_value_column(col):
    """
    Checks if a column has same values for every row
    """
    col_set = set(col)
    if np.NaN in col_set:
        col_set.remove(np.NaN)  # do not count NaN values
    return len(col_set) == 1


def float_row(row):
    """
    Converts strings to floats for a given row
    """
    new_row = row
    for i in range(1, len(row)):  # pass id column
        if row[i] == 'neg':
            new_row[i] = 0
        elif row[i] == 'pos':
            new_row[i] = 1
        elif row[i] == 'na':
            new_row[i] = np.NaN
        else:
            new_row[i] = float(row[i])
    return new_row


def get_data_from_csv(path):
    """
    Gets headers, columns and rows from given csv file path
    """
    headers, cols, rows = [], [], []
    with open(path, newline='') as file:
        reader = csv.reader(file)
        first = True
        for row in reader:
            if first:
                headers = row
                for j in range(len(row)):
                    cols.append([])
                first = False
                continue
            rows.append(float_row(row))
            for j in range(len(headers)):
                cols[j].append(row[j])
    return headers, cols, rows


def read_csv_with_pandas(path):
    """
    Reads the csv file as pandas dataframe and applies required conversions
    """
    df = pd.read_csv(path)
    df.replace(to_replace='na', value=0, inplace=True)
    df.replace(to_replace='neg', value=0, inplace=True)
    df.replace(to_replace='pos', value=1, inplace=True)
    for col in df.columns[2:]:
        df[col] = df[col].astype(float)
    return df


def init_parser(args):
    """
    init argument parser with given arguments
    """
    parser = argparse.ArgumentParser()
    for arg in args:
        name = arg.pop('name')
        parser.add_argument(name, **arg)
    return parser.parse_args()
