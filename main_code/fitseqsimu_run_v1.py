#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import fitseqsimu_methods_v1

# try running with command 
#python fitmutsimu_methods_v1.py -l 100000 -t simu_input_time_points.csv -s simu_input_mutation_fitness.csv -o test


###########################
##### PARSE ARGUMENTS #####
###########################

parser = argparse.ArgumentParser(description = 'Simulated pooled growth of a complexed phenotyphic population', formatter_class = argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-t', '--t_seq', type=str, required=True,
                    help = 'a .csv file, with:'
                           '1st column: sequenced time-points evaluated in number of generations, '
                           '2nd+ column: average number of reads per barcode for each sequenced time-point (accept multiple columns for multiple sequencing runs).')

parser.add_argument('-s', '--fitness', type=str, required='', 
                    help = 'a .csv file, with:'
                           '1st column: fitness of each genotype' 
                           '2nd column: initial cell number of each genotype')
    
parser.add_argument('-d', '--dna_copies', type=int, default=500, 
                    help = 'average genome copy number per barcode used as template in PCR')

parser.add_argument('-p', '--pcr_cycles', type=int, default=25, 
                    help = 'number of cycles in PCR')

parser.add_argument('-o', '--output_filename', type=str, default='output', 
                    help = 'prefix of output .csv files')

args = parser.parse_args()


##########    
csv_input = pd.read_csv(args.t_seq, header=None)
t_seq = np.array(csv_input[0][~pd.isnull(csv_input[0])], dtype=int)
read_num_average_seq_bundle = np.array(csv_input.loc[:,1:csv_input.shape[1]], dtype=int)

csv_input = pd.read_csv(args.fitness, header=None)
s_array = np.array(csv_input[0][~pd.isnull(csv_input[0])], dtype=float)
n0_array = np.array(csv_input[1][~pd.isnull(csv_input[1])], dtype=float)

dna_copies = args.dna_copies
pcr_cycles = args.pcr_cycles
output_filename = args.output_filename

my_obj = fitseqsimu_methods_v1.FitSeqSimu(t_seq = t_seq,
                                          read_num_average_seq_bundle = read_num_average_seq_bundle,
                                          s_array = s_array,
                                          n0_array = n0_array, 
                                          dna_copies = dna_copies,
                                          pcr_cycles = pcr_cycles,
                                          output_filename = output_filename)

my_obj.function_main()

