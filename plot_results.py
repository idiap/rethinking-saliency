#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

import argparse

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

from os import walk

plt.rcParams.update({'font.size': 12})

parser = argparse.ArgumentParser(description='Arguments for Saliency metric')

# Optimization arguments
parser.add_argument( '--csv_path', type=str,
                    help="Path where csv files are saved")

args = parser.parse_args()

def plot(csv_list, out_file, legends, xlabel, ylabel):
    """
    csv_list: list of csv files to plot, one per legend label
    legends: tuple of legends of csv files in csv_list
    plotval: "loss" or "accuracy"        
    """
    i = 0
    for c in csv_list:
        df = pd.read_csv(c, header=None).T
        plt.plot(df[0].values, df[1].values, linewidth=4)
        i+=1

    plt.ylim((0., 1.0))
    plt.legend(legends, frameon=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.clf()


if __name__ == "__main__":
    save_path = 'results/'

    ## list files in csv folder
    filenames = next(walk(args.csv_path), (None, None, []))[2]  # [] if no file

    plot(filenames, out_file = save_path, legends=filenames, xlabel='Masking Fraction', ylabel = 'Fraction of Images Correctly Classified')
