#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import fitseq2_methods

# try running with command
# python3 ./fitseq2_methods.py -i simu_0_EvoSimulation_Read_Number.csv -t fitmut_input_time_points.csv -o test

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

parser.add_argument('-dt', '--delta_t', type=float, default=0,
                    help='number of generations between bottlenecks')

parser.add_argument('-c', '--c', type=float, default=1, 
                    help='half of variance introduced by cell growth and cell transfer')

parser.add_argument('-a', '--opt_algorithm', type=str, default='differential_evolution',
                    choices = ['differential_evolution', 'nelder_mead'], 
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

delta_t = args.delta_t
if delta_t == 0:
    delta_t = t_seq[1] - t_seq[0]
c = args.c # per cycle
parallelize = args.parallelize
opt_algorithm = args.opt_algorithm
max_iter_num = args.maximum_iteration_number
output_filename = args.output_filename

my_obj = fitseq2_methods.FitSeq(read_num_seq = read_num_seq,
                                t_seq = t_seq,
                                cell_depth_seq = cell_depth_seq,
                                delta_t = delta_t,
                                c = c,
                                opt_algorithm = opt_algorithm,
                                max_iter_num = max_iter_num,
                                parallelize = parallelize,
                                output_filename = output_filename)


my_obj.function_main()

