"""
Script to create bar graphs for comparing error in pose graph and other filters.
"""

import matplotlib.pyplot as plt
import numpy as np
import os, sys, csv
import rospkg
from glob import glob

def read_errs_from_file(fname:str):
    """
    Read the avg errs from the csv file. There should be one float per line, for 10 lines, and nothing else.
    """
    with open(fname, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        errs = [float(row[0]) for row in csv_reader]
        return errs

def create_bar_plot(pgs, ekf, naive, run_name:str):
    """
    Create a bar plot using the pgs data and the filter data.
    One of ekf, naive will be None.
    @param run_name - subdir name of data, also desired filename of saved plot.
    """
    filter_type = "EKF-SLAM" if naive is None else "Naive"
    filter_errs = ekf if filter_type == "EKF-SLAM" else naive

    barWidth = 0.25
    fig = plt.subplots()
    # fig = plt.subplots(figsize=(12, 8))
    # bar heights correspond to err values.
    # Set position of bar on X axis
    br1 = np.arange(len(pgs))
    br2 = [x + barWidth for x in br1]
    # Make the plot
    plt.bar(br1, pgs, color="purple", width=barWidth, edgecolor='grey', label='Pose-Graph SLAM')
    plt.bar(br2, filter_errs, color="green", width=barWidth, edgecolor='grey', label=filter_type)
    # Adding Xticks
    plt.xlabel('Run number', fontsize = 15) #fontweight='bold'
    plt.ylabel('Average position error (m)', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(pgs))],
               [i+1 for i in range(len(pgs))])
    plt.legend(loc="upper left")
    # build plot title.
    title = "High" if "high" in run_name else "Low"
    title += " Noise, " + filter_type + " vs "
    title += "One-Time-" if "one" in run_name else "Iterative-"
    title += "PGS"
    plt.title(title)
    # plt.show()
    plt.savefig(base_pkg_path+"/plots/err_comparisons/{:}.png".format(run_name), format='png')

    # Print the relative overall avg errors to console for use elsewhere.
    print(run_name + ":\n\tPGS: {:.4f}\n\t{:}: {:.4f}".format(sum(pgs)/len(pgs), filter_type, sum(filter_errs)/len(filter_errs)))

def main():
    """
    For all relevant directories, create and save a bar graph.
    """
    rospack = rospkg.RosPack()
    global base_pkg_path
    base_pkg_path = rospack.get_path('base_pkg')
    data_dir = base_pkg_path + "/data"
    # get all subdirectories.
    subdirs = glob(data_dir+"/*/", recursive = False)
    for dir in subdirs:
        # plot the filter and PGS results on a bar graph.
        pgs_errs = read_errs_from_file(dir+'pose_graph_result.csv')
        # read either an ekf or naive filter file.
        ekf_errs, naive_errs = None, None
        try:
            ekf_errs = read_errs_from_file(dir+'ekf.csv')
        except:
            naive_errs = read_errs_from_file(dir+'naive.csv')
        
        # save graph to plots directory, named after the run settings.
        run_name = dir[:-1].split("/")[-1]
        create_bar_plot(pgs_errs, ekf_errs, naive_errs, run_name)


if __name__ == '__main__':
    main()