#!/usr/bin/env python3
import csv
import copy
import itertools
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm


np.random.seed(1)
class GrowthSimulation:
    def __init__(self,
            t_seq,
            read_num_average_seq_bundle,
            s_array,
            n0_array,
            dna_copies,
            pcr_cycles,
            output_filenamepath_prefix
            ):
        """
        """
        self.t_seq = t_seq
        self.seq_num = len(self.t_seq)
        self.evo_num = self.t_seq[-1]
        self.delta_t = self.t_seq[1] - self.t_seq[0]
        
        self.s_array = s_array
        self.n0_array = n0_array

        self.lineages_num = len(s_array)

        self.cell_num_average_bottleneck = np.mean(self.n0_array)

        self.dna_copies = dna_copies
        self.pcr_cycles = pcr_cycles
        
        self.read_num_average_seq_bundle = read_num_average_seq_bundle
        self.bundle_num = self.read_num_average_seq_bundle.shape[1]
        # double check for single (not bundle)
        
        self.output_filenamepath_prefix = output_filenamepath_prefix
        
        self.x_mean_seq = np.zeros(self.evo_num + 1, dtype=float)  # mean fitness
        
        
    
    def function_growth_lineage(self, data_lineage_dict):
        """
        Growth
        """
        if (not isinstance(data_lineage_dict, dict) or not data_lineage_dict): #final layer
            if data_lineage_dict[0] > 0:
                #data_lineage_dict[0] = 2 * data_lineage_dict[0] * (1 + data_lineage_dict[1])
                # Writgtian fitness
                data_lineage_dict[0] = 2 * data_lineage_dict[0] * np.exp(data_lineage_dict[1])
                # Mathusian fitness
            return data_lineage_dict
    
        else:
            for (item_key, item_value) in data_lineage_dict.items():
                data_lineage_dict[item_key] = self.function_growth_lineage(item_value)
            return data_lineage_dict
            
    
    
    def function_sampling_lineage(self, data_lineage_dict):
        """
        Sampling
        """
        if (not isinstance(data_lineage_dict, dict) or not data_lineage_dict): #final layer
            if data_lineage_dict[0] > 0:
                data_lineage_dict[0] = np.random.poisson(data_lineage_dict[0] * self.ratio)
            return data_lineage_dict
    
        else:
            for (item_key, item_value) in data_lineage_dict.items():
                data_lineage_dict[item_key] = self.function_sampling_lineage(item_value)
            return data_lineage_dict
            
    

    def function_amplifying_lineage(self, data_lineage_dict):
        """
        Amplifying
        """
        if (not isinstance(data_lineage_dict, dict) or not data_lineage_dict): #final layer
            if data_lineage_dict[0] > 0:
                for r in range(self.pcr_cycles):
                    data_lineage_dict[0] = np.random.poisson(2 * data_lineage_dict[0])
            return data_lineage_dict
    
        else:
            for (item_key, item_value) in data_lineage_dict.items():
                data_lineage_dict[item_key] = self.function_amplifying_lineage(item_value)
            return data_lineage_dict
    

    
    def function_update_data_population(self, data_population_dict, t, phase_choice):
        """
        """
        for i in range(self.lineages_num):
            if phase_choice == 'growth':
                data_population_dict[i] = self.function_growth_lineage(data_population_dict[i])
            
            elif phase_choice == 'sampling':
                data_population_dict[i] = self.function_sampling_lineage(data_population_dict[i])
                
            elif phase_choice == 'amplifying':
                data_population_dict[i] = self.function_amplifying_lineage(data_population_dict[i])
        
        return data_population_dict
            

    
    def function_extract_data_info_lineage(self, data_lineage_dict):
        """
        Extract mutation information from a single lineage
        """
        if (not isinstance(data_lineage_dict, dict) or not data_lineage_dict):
            self.data_info_lineage['cell_number'].append(data_lineage_dict[0])
            self.data_info_lineage['cell_fitness'].append(data_lineage_dict[1])
        else:
            for (item_key, item_value) in data_lineage_dict.items():
                self.function_extract_data_info_lineage(item_value)

    
    
    def function_calculate_mean_fitness(self, t):
        """
        """
        tmp_numerator = 0    # numerator/denominator
        tmp_denominator = 0
        
        for i in range(self.lineages_num):
            self.data_info_lineage = {
                'cell_fitness':[],
                'cell_number':[]
                }
                                                
            self.function_extract_data_info_lineage(self.data_dict[i])
        
            tmp_numerator += np.dot(
                self.data_info_lineage['cell_fitness'],
                self.data_info_lineage['cell_number']
                )
            tmp_denominator += np.sum(self.data_info_lineage['cell_number'])
         
        self.x_mean_seq[t] = tmp_numerator/tmp_denominator
        
    
    
    def function_extract_data_info(self, data_dict):
        """
        Extract the cell number of all lineages
        """
        data_info = {'all': np.zeros(self.lineages_num, dtype=float)}
        
        for i in range(self.lineages_num):            
            self.data_info_lineage = {
                'cell_fitness':[],
                'cell_number':[]
                }

            self.function_extract_data_info_lineage(data_dict[i])
            
            data_info['all'][i] = np.sum(self.data_info_lineage['cell_number'])
            
        return data_info
    

    
    def funciton_save_data(self):
        """
        """
        ####################
        data_other = {
            'Time_Points': self.t_seq,
            'Mean_Fitness': self.x_mean_seq[self.t_seq],
            'lineages_Number': [self.lineages_num],
            'gDNA_Copies': [self.dna_copies],
            'PCR_cycles': [self.pcr_cycles],
            'Fitness': self.s_array,
            'Average_Read_Depth': self.read_num_average_seq_bundle
            }
        
        tmp = list(itertools.zip_longest(*list(data_other.values())))
        filenamepath = '{}_EvoSimulation_Other_Info.csv'.format(self.output_filenamepath_prefix)
        with open(filenamepath, 'w') as f:
            w = csv.writer(f)
            w.writerow(data_other.keys())
            w.writerows(tmp)
        
        
        tmp = pd.DataFrame(self.cell_num_seq_dict['all'])
        filenamepath = '{}_EvoSimulation_Saturated_Cell_Number.csv'.format(self.output_filenamepath_prefix)
        tmp.to_csv(filenamepath, index=False, header=False)

        tmp = pd.DataFrame(self.bottleneck_cell_num_seq_dict['all'])
        filenamepath = '{}_EvoSimulation_Bottleneck_Cell_Number.csv'.format(self.output_filenamepath_prefix)
        tmp.to_csv(filenamepath, index=False, header=False)
        
        for bundle_idx in range(self.bundle_num):
            tmp = pd.DataFrame(self.read_num_seq_bundle_dict[bundle_idx]['all'])
            filenamepath = '{}_{}_EvoSimulation_Read_Number.csv'.format(self.output_filenamepath_prefix, bundle_idx)
            tmp.to_csv(filenamepath, index=False, header=False)

        

    def simulate(self):
        """
        """
        # Step 1: growth (to simulate the whole process of the pooled growth)
        self.cell_num_seq_dict = dict()
        self.bottleneck_cell_num_seq_dict = dict()
        self.read_num_seq_bundle_dict = {
            i: dict() for i in range(self.bundle_num)
            }

        for t in tqdm(range(self.evo_num + 1)):
            #Step 2-1: initialization (to simulate the process of evolution initialization)
            if t == 0: 
                # data at one generation (particularly, number of cell transferred for t=0, delta_t...)
                self.data_dict = {
                    i: [self.n0_array[i], self.s_array[i]]
                    for i in range(self.lineages_num)
                    }
                # [a0, a1]: -- a0: number of individuals
                #           -- a1: fitness of each individual (sum of fitness of all mutations)
                
                self.data_saturated_dict = {
                    i: [self.n0_array[i]*2**self.delta_t, self.s_array[i]]
                    for i in range(self.lineages_num)
                    }

            else: 
                # Step 2-2: cell growth (to simulate growth of of one generation)
                self.data_dict = self.function_update_data_population(self.data_dict, t, 'growth')
                # -- growth part1: simulate deterministic growth

                depth = np.sum(self.function_extract_data_info(self.data_dict)['all'])
                self.ratio = 2 ** (np.mod(t-1, self.delta_t) + 1) * self.cell_num_average_bottleneck * self.lineages_num / depth
                self.data_dict = self.function_update_data_population(self.data_dict, t, 'sampling')
                # -- growth part2: add growth noise
                            
     
                # Step 2-3: cell transfer (to simulate sampling of cell transfer at the bottleneck)
                mode_factor = np.mod(t, self.delta_t)
                if mode_factor == 0: # bottlenecks
                    self.data_saturated_dict = copy.deepcopy(self.data_dict)
                    self.ratio = 1 / 2 ** self.delta_t
                    self.data_dict = self.function_update_data_population(self.data_dict, t, 'sampling')
                    # -- sampling
                                    
            
            # -- Calculate the mean_fitness at the generation t
            self.function_calculate_mean_fitness(t)
                
            
            mode_factor = np.mod(t, self.delta_t)
            if mode_factor == 0:
                k = int(t/self.delta_t)

                # -- Save data of samples at bottleneck
                output_bottleneck = self.function_extract_data_info(self.data_dict)
                for key in output_bottleneck.keys():
                    if key not in self.bottleneck_cell_num_seq_dict.keys():
                        self.bottleneck_cell_num_seq_dict[key] = np.zeros((self.lineages_num, self.seq_num), dtype=int)
                    self.bottleneck_cell_num_seq_dict[key][:,k] = output_bottleneck[key]
                
                # -- Save data of samples at saturated (the samples are used for the following DNA extraction and PCR)
                output_saturated = self.function_extract_data_info(self.data_saturated_dict)
                for key in output_saturated.keys():
                    if key not in self.cell_num_seq_dict.keys():
                        self.cell_num_seq_dict[key] = np.zeros((self.lineages_num, self.seq_num), dtype=int)
                    self.cell_num_seq_dict[key][:,k] = output_saturated[key]

                # Step 2-4: DNA extraction (to simulate the processes of extracting genome DNA)
                depth = np.sum(output_saturated['all'])
                self.ratio = self.dna_copies * self.lineages_num / depth
                self.data_dna_dict = copy.deepcopy(self.data_saturated_dict)
                self.data_dna_dict = self.function_update_data_population(self.data_dna_dict, t, 'sampling') # -- sampling
                
                # Step 2-5: PCR (to simulate the processes of running PCR)
                self.data_pcr_dict = copy.deepcopy(self.data_dna_dict)
                self.data_pcr_dict = self.function_update_data_population(self.data_pcr_dict, t, 'amplifying')
            
                # Step 2-6: sequencing (to simulate the processes of sequencing)
                depth = np.sum(self.function_extract_data_info(self.data_pcr_dict)['all'])
                for bundle_idx in range(self.bundle_num):
                    self.read_num_average_seq = self.read_num_average_seq_bundle[:,bundle_idx]  # might need to change
                    self.ratio = self.read_num_average_seq[k] * self.lineages_num / depth
                    self.data_sequencing_dict = copy.deepcopy(self.data_pcr_dict)
                    self.data_sequencing_dict = self.function_update_data_population(self.data_sequencing_dict, t, 'sampling')
                        
                    # -- Save read number data for sequencing results
                    output = self.function_extract_data_info(self.data_sequencing_dict)
                    for idx in output.keys():
                        if idx not in self.read_num_seq_bundle_dict[bundle_idx].keys():
                            self.read_num_seq_bundle_dict[bundle_idx][idx] = np.zeros((self.lineages_num, self.seq_num), dtype=int)
                        self.read_num_seq_bundle_dict[bundle_idx][idx][:,k] = output[idx]
        
        
        # Step: save data
        self.funciton_save_data() 
        


def process(
        t_seq,
        read_num_average_seq_bundle,
        s_array,
        n0_array,
        dna_copies,
        pcr_cycles,
        output_filenamepath_prefix
        ):
    """
    """
    my_obj = GrowthSimulation(
        t_seq,
        read_num_average_seq_bundle,
        s_array,
        n0_array,
        dna_copies,
        pcr_cycles,
        output_filenamepath_prefix
        )
    
    my_obj.simulate()
    



def main():
    """
    """
    parser = argparse.ArgumentParser(
        description='Simulated pooled growth of a complexed phenotyphic population',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    parser.add_argument(
        '-t', '--t_seq',
        type=str,
        required=True,
        help='a .csv file, with:'
             '1st column: sequenced time-points evaluated in number of generations, '
             '2nd+ column: average number of reads per barcode for each sequenced time-point (accept multiple columns for multiple sequencing runs).'
        )

    parser.add_argument(
        '-s', '--fitness',
        type=str,
        required=True,
        help='a .csv file, with:'
             '1st column: fitness of each genotype'
             '2nd column: initial cell number of each genotype'
        )
    
    parser.add_argument(
        '-d', '--dna_copies',
        type=int,
        default=500,
        help='average genome copy number per barcode used as template in PCR'
        )

    parser.add_argument(
        '-p', '--pcr_cycles',
        type=int,
        default=25,
        help='number of cycles in PCR'
        )

    parser.add_argument(
        '-o', '--output_filenamepath_prefix',
        type=str,
        default='output',
        help='prefix filenamepath of output files'
        )

    args = parser.parse_args()
    
    csv_input = pd.read_csv(args.t_seq, header=None)
    t_seq = np.array(csv_input[0][~pd.isnull(csv_input[0])], dtype=int)
    read_num_average_seq_bundle = np.array(csv_input.loc[:,1:csv_input.shape[1]], dtype=int)

    csv_input = pd.read_csv(args.fitness, header=None)
    s_array = np.array(csv_input[0][~pd.isnull(csv_input[0])], dtype=float)
    n0_array = np.array(csv_input[1][~pd.isnull(csv_input[1])], dtype=float)

    dna_copies = args.dna_copies
    pcr_cycles = args.pcr_cycles
    output_filenamepath_prefix = args.output_filenamepath_prefix
    
   
    process(
        t_seq,
        read_num_average_seq_bundle,
        s_array,
        n0_array,
        dna_copies,
        pcr_cycles,
        output_filenamepath_prefix
        )


if __name__=="__main__":
    main()
