#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import fitseq_methods

# try running with command
# python3 ./fitseq_run.py -i simu_0_EvoSimulation_Read_Number.csv -t fitmut_input_time_points.csv -o test

###########################
##### PARSE ARGUMENTS #####
###########################

parser = argparse.ArgumentParser(description='Estimate fitness of phenotypes in a competitive pooled growth experiment',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
parser.add_argument('-i', '--input', type=str, required=True,
                    help='a .csv file: with each column being the read number per barcode at each sequenced time-point')

parser.add_argument('-t', '--t_seq', type=str, required=True,
                    help='a .csv file of 2 columns:'
                         '1st column: sequenced time-points evaluated in number of generations, '
                         '2nd column: total effective number of cells of the population for each sequenced time-point.')
                    
parser.add_argument('-a', '--opt_algorithm', type=str, default='bfgs',
                    choices = ['differential_evolution', 'nelder_mead','bfgs'], 
                    help='choose optmization algorithm')

parser.add_argument('-n', '--maximum_iteration_number', type=int, default=50,
                    help='maximum number of iterations')

parser.add_argument('-p', '--parallelize', type=bool, default=True,
                    help='whether to use multiprocess module to parallelize inference across lineages')
                    
parser.add_argument('-o', '--output_filename', type=str, default='output',
                    help='prefix of output .csv files')

args = parser.parse_args()


#####
read_num_seq = np.array(pd.read_csv(args.input, header=None), dtype=float)

csv_input = pd.read_csv(args.t_seq, header=None)
t_seq = np.array(csv_input[0][~pd.isnull(csv_input[0])], dtype=float)
cell_depth_seq = np.array(csv_input[1][~pd.isnull(csv_input[1])], dtype=float)

parallelize = args.parallelize
max_iter_num = args.maximum_iteration_number
opt_algorithm = args.opt_algorithm
output_filename = args.output_filename

my_obj = fitseq_methods.FitSeq(read_num_seq = read_num_seq,
                               t_seq = t_seq,
                               cell_depth_seq = cell_depth_seq,
                               opt_algorithm = opt_algorithm,
                               max_iter_num = max_iter_num,
                               parallelize = parallelize,
                               output_filename = output_filename)

my_obj.function_main()

