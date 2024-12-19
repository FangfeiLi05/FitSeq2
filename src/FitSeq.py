#!/usr/bin/env python3
import csv
import copy
import time
import itertools
import argparse

import numpy as np
import pandas as pd
from scipy import special
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import differential_evolution
from scipy.stats import linregress
from tqdm import tqdm
from multiprocess import Pool


# Fitness inference object
class FitSeq:
    def __init__(self,
            read_num_seq,
            t_seq,
            cell_depth_seq,
            opt_algorithm,
            max_iter_num,
            parallelize,
            output_filenamepath_prefix
            ):
        
        # Preparing inputs
        self.read_num_seq = read_num_seq
        self.read_depth_seq = np.sum(self.read_num_seq, axis=0)
        self.lineages_num = np.shape(self.read_num_seq)[0]
        self.t_seq = t_seq
        self.seq_num = len(self.t_seq)
        self.cell_depth_seq = cell_depth_seq
        self.ratio = np.true_divide(self.read_depth_seq, self.cell_depth_seq)
        self.read_num_seq[self.read_num_seq < 1] = 1
        # approximate distribution is not accurate when r < 1

        self.opt_algorithm = opt_algorithm
        self.max_iter_num = max_iter_num
        self.parallelize = parallelize
        self.output_filenamepath_prefix = output_filenamepath_prefix

        # Set bounds for the optimization
        self.bounds = Bounds(-1, 1)

        # Define other variables
        self.kappa_seq = 2.5 * np.ones(self.seq_num)

        self.regression_num = 2
        self.read_freq_seq = self.read_num_seq / self.read_depth_seq



    ##########
    def function_sum_term(self):
        """
        Pre-calculate a term (i.e. sum_term) to reduce calculations in estimating the number of reads.
        """
        self.sum_term_t_seq = np.zeros(self.seq_num-1, dtype=float) # from tkminus1 to tk
        sum_term_extend_t_seq_tmp = 0
        for k in range(self.seq_num-1):
            sum_term_extend_t_seq_tmp = (self.t_seq[k+1]-self.t_seq[k]) * (self.x_mean_seq[k+1] + self.x_mean_seq[k])/2
            self.sum_term_t_seq[k] =  np.exp(-sum_term_extend_t_seq_tmp)
        
        self.sum_term_sqrt_t_seq = np.sqrt(self.sum_term_t_seq)



    def function_read_num_theory_scaler(self, s):
        """
        Estimate read number all time points for a lineage given s.
        Inputs: s (scaler)
        Output: {'read_number': (array, vector)}
        """
        if type(s) != np.float64:
            #print(type(s))
            s = list(s)[0] #temporary
        
        read_num_seq_lineage_theory = np.zeros(self.seq_num, dtype=float)
        read_num_seq_lineage_theory[0] = self.read_num_seq_lineage[0]
        
        for k in range(1, self.seq_num):
            sum_term_tk_minus_tkminus1 = self.sum_term_t_seq[k-1]
            all_tmp1 = np.exp(s * (self.t_seq[k] - self.t_seq[k-1])) * sum_term_tk_minus_tkminus1
            #read_num_seq_lineage_theory[k] = np.multiply(self.read_num_seq_lineage[k-1]/self.ratio[k-1]*self.ratio[k], all_tmp1)
            read_num_seq_lineage_theory[k] = self.read_num_seq_lineage[k-1]/self.ratio[k-1]*self.ratio[k] *  all_tmp1
            
            #print(type(s), len(s))
        output = {'read_number': read_num_seq_lineage_theory}

        return output
    


    def function_read_num_theory_array(self, s_array):
        """
        Estimate read number all time points for a lineage given s.
        Inputs: s_array (array, vector)
        Output: {'read_number': (array, 2D matrix)}
        """
        s_len = len(s_array)
        
        read_num_seq_lineage_theory = np.zeros((s_len, self.seq_num), dtype=float)
        read_num_seq_lineage_theory[:,0] = cself.read_num_seq_lineage[0]

        for k in range(1, self.seq_num):
            sum_term_tk_minus_tkminus1 = self.sum_term_t_seq[k-1]
            all_tmp1 = np.exp(s_array * (self.t_seq[k] - self.t_seq[k-1])) * sum_term_tk_minus_tkminus1
            cell_num_seq_lineage_theory[:,k] = np.multiply(self.read_num_seq_lineage[k-1]/self.ratio[k-1]*self.ratio[k], all_tmp1)
    
        output = {'read_number': read_num_seq_lineage_theory}
        return output



    def function_prior_loglikelihood_scaler(self, s):
        """
        Calculate log-likelihood value of a lineage given s.
        Inputs: s (scaler)
        Output: log-likelihood value of all time poins (scaler)
        """        
        output = self.function_read_num_theory_scaler(s)
        read_num_seq_lineage_theory = output['read_number']
        read_num_seq_lineage_theory[read_num_seq_lineage_theory < 1] = 1
        
        tmp_kappa_reverse = 1/self.kappa_seq

        tmp_theory = read_num_seq_lineage_theory
        tmp_observed = self.read_num_seq_lineage
        tmp_observed_reverse = 1/tmp_observed
        ive_ele = 2* np.multiply(np.sqrt(np.multiply(tmp_theory, tmp_observed)), tmp_kappa_reverse)
 
        tmp_part1 = np.log(tmp_kappa_reverse)
        tmp_part2 = 1/2 * np.log(np.multiply(tmp_theory, tmp_observed_reverse))
        tmp_part3 = -np.multiply(tmp_theory + tmp_observed, tmp_kappa_reverse)
        tmp_part4 = np.log(special.ive(1, ive_ele)) + ive_ele

        log_likelihood_seq_lineage = tmp_part1 + tmp_part2 + tmp_part3 + tmp_part4
        log_likelihood_lineage = np.sum(log_likelihood_seq_lineage, axis=0)

        return log_likelihood_lineage
        
     

    
    ##########
    def function_prior_loglikelihood_array(self, s_array):
        """
        Calculate log-likelihood value of a lineage given s.
        Inputs: s_array (array, vector)
        Output: log-likelihood value of all time poins (array, vector)
        """
        s_len = len(s_array)

        output = self.function_read_num_theory_array(s_array)
        read_num_seq_lineage_theory = output['read_number'] #(s_len, eq_num)
        read_num_seq_lineage_theory[read_num_seq_lineage_theory < 1] = 1
        
        tmp_kappa_reverse = np.tile(1/self.kappa_seq, (s_len, 1))

        tmp_theory = read_num_seq_lineage_theory
        tmp_observed = np.tile(self.read_num_seq_lineage, (s_len, 1))
        tmp_observed_reverse = np.tile(1/self.read_num_seq_lineage, (s_len, 1))
        ive_ele = 2* np.multiply(np.sqrt(np.multiply(tmp_theory, tmp_observed)), tmp_kappa_reverse)
 
        tmp_part1 = np.log(tmp_kappa_reverse)
        tmp_part2 = 1/2 * np.log(np.multiply(tmp_theory, tmp_observed_reverse))
        tmp_part3 = -np.multiply(tmp_theory + tmp_observed, tmp_kappa_reverse)
        tmp_part4 = np.log(special.ive(1, ive_ele)) + ive_ele

        log_likelihood_seq_lineage = tmp_part1 + tmp_part2 + tmp_part3 + tmp_part4
        log_likelihood_lineage = np.sum(log_likelihood_seq_lineage, axis=2)

        return log_likelihood_lineage
    


    ##########
    def function_prior_loglikelihood_scaler_approx(self, s):
        """
        Calculate log-likelihood value of a lineage given s.
        Inputs: s (scaler)
        Output: log-likelihood value of all time poins (scaler)
        """
        output = self.function_read_num_theory_scaler(s)
        read_num_seq_lineage_theory = output['read_number']
        read_num_seq_lineage_theory[read_num_seq_lineage_theory < 1] = 1
        
        log_likelihood_seq_lineage = (
            0.25*np.log(read_num_seq_lineage_theory)
            - 0.5*np.log(4*np.pi*self.kappa_seq)
            - 0.75*np.log(self.read_num_seq_lineage)
            - (np.sqrt(self.read_num_seq_lineage)
            - np.sqrt(read_num_seq_lineage_theory))**2
            / self.kappa_seq)
        
        log_likelihood_lineage = np.sum(log_likelihood_seq_lineage, axis=0)

        return log_likelihood_lineage



    def function_prior_loglikelihood_array_approx(self, s_array):
        """
        Calculate log-likelihood value of a lineage given s.
        Inputs: s_array (array, vector)
        Output: log-likelihood value of all time poins (array, vector)
        """
        s_len = len(s_array)

        output = self.function_read_num_theory_array(s_array)
        read_num_seq_lineage_theory = output['read_number']
        read_num_seq_lineage_theory[read_num_seq_lineage_theory < 1] = 1
        
        tmp_kappa = np.tile(self.kappa_seq, (s_len, 1))
        tmp_theory = read_num_seq_lineage_theory
        tmp_observed = np.tile(self.read_num_seq_lineage, (s_len, 1))
        
        log_likelihood_seq_lineage = (
            0.25*np.log(tmp_theory)
            - 0.5*np.log(4*np.pi*tmp_kappa)
            - 0.75*np.log(tmp_observed)
            - (np.sqrt(tmp_observed) - np.sqrt(tmp_theory))**2
            / tmp_kappa)
        
        log_likelihood_lineage = np.sum(log_likelihood_seq_lineage, axis=1)

        return log_likelihood_lineage



    def function_prior_loglikelihood_opt(self, s):
        """
        Calculate log-likelihood value of a lineage given s in optimization
        """
        #output = self.function_prior_loglikelihood_scaler(s)
        output = self.function_prior_loglikelihood_scaler_approx(s)

        return -output #minimization only in python



    def function_parallel(self, i): 
        """
        i: lineage label
        Output optimized s for each lineage i
        """
        self.read_num_seq_lineage = self.read_num_seq[i, :]
        
        if self.opt_algorithm == 'differential_evolution':
            opt_output = differential_evolution(
                func=self.function_prior_loglikelihood_opt,
                seed=1,
                bounds=self.bounds
                )
            
        elif self.opt_algorithm == 'nelder_mead': 
            opt_output = minimize(
                self.function_prior_loglikelihood_opt,
                x0=[0.001],
                method='Nelder-Mead',
                bounds=self.bounds,
                options={'ftol': 1e-8, 'disp': False, 'maxiter': 500}
                )

        elif self.opt_algorithm == 'bfgs': 
            opt_output = minimize(
                self.function_prior_loglikelihood_opt,
                x0=[0.001],
                method='L-BFGS-B',
                bounds=self.bounds,
                options={'ftol': 1e-8, 'disp': False, 'maxiter': 500}
                )
                                  
        s_opt = opt_output.x[0]
    
        return s_opt

    

    def function_update_mean_fitness(self, k_iter):
        """
        Updated mean fitness
        """
        x_mean = np.sum(self.read_freq_seq * np.tile(self.result_s, (self.seq_num, 1)).T, axis=0)
        self.x_mean_seq_dict[k_iter] = x_mean - x_mean[0]
        #self.result_s = self.result_s - x_mean[0]
        
        self.read_num_seq_theory = np.zeros(np.shape(self.read_num_seq), dtype=float)
        for i in range(self.lineages_num):
            self.read_num_seq_lineage = self.read_num_seq[i, :]
            output = self.function_read_num_theory_scaler(self.result_s[i])
            self.read_num_seq_theory[i,:] = output['read_number']
        
        #x_mean_tmp = np.sum(self.cell_num_seq_theory * np.tile(self.result_s, (self.seq_num, 1)).T, axis=0)
        #x_mean = x_mean_tmp / np.sum(self.cell_num_seq_theory, axis=0)
        #self.x_mean_seq_dict[k_iter] = x_mean - x_mean[0]
        

        
    def function_run_iteration(self):
        """
        Run a single interation
        Run optimization for each lineages to find their optimized s
        """
        if self.parallelize:
            pool_obj = Pool() # might need to change processes=8
            output_tmp = pool_obj.map(self.function_parallel, tqdm(range(self.lineages_num)))
            pool_obj.close()
            self.result_s = np.array(output_tmp, dtype=float)

        else:
            for i in tqdm(range(self.lineages_num)):
                self.result_s[i] = self.function_parallel(i)
        

    
    def function_prior_loglikelihood_iteration(self):
        """ """
        self.prior_loglikelihood = np.zeros(self.lineages_num)
        for i in range(self.lineages_num):
            self.read_num_seq_lineage = self.read_num_seq[i, :]
            #self.prior_loglikelihood[i] = self.function_prior_loglikelihood_scaler(self.result_s[i])
            self.prior_loglikelihood[i] = self.function_prior_loglikelihood_scaler_approx(self.result_s[i])
            
                    

    def function_save_data(self, output_result):
        """
        Save data according to label: if it's saving a step or the final data
        """
        tmp_1 = output_result['FitSeq_Result']
        tmp = list(itertools.zip_longest(*list(tmp_1.values())))
        filenamepath = '{}_FitSeq_Result.csv'.format(self.output_filenamepath_prefix)
        with open(filenamepath, 'w') as f:
            w = csv.writer(f)
            w.writerow(tmp_1.keys())
            w.writerows(tmp)
 
        tmp_2 = output_result['Mean_fitness_Result']
        tmp = list(itertools.zip_longest(*list(tmp_2.values())))
        filenamepath = '{}_FitSeq_Result_Mean_fitness.csv'.format(
            self.output_filenamepath_prefix
            )
        with open(filenamepath, 'w') as f:
            w = csv.writer(f)
            w.writerow(tmp_2.keys())
            w.writerows(tmp)
        
        tmp_3 = output_result['Read_Number_Estimated']
        tmp = pd.DataFrame(tmp_3.astype(int))
        filenamepath = '{}_FitSeq_Result_Read_Number_Estimated.csv'.format(
            self.output_filenamepath_prefix
            )
        tmp.to_csv(filenamepath, index=False, header=False)
                


    def seq(self):
        """
        main function
        """
        start = time.time()
        self.calculate_error = False
        self.prior_loglikelihood_list = []
        self.iter_timing_list = [] # running time for each iteration
        
        #####
        self.result_s = np.zeros(self.lineages_num, dtype=float)
        self.prior_loglikelihood = np.zeros(self.lineages_num, dtype=float)
        self.read_num_seq_theory = np.zeros((self.lineages_num, self.seq_num), dtype=float)
        
        # choice_1 (without linear regression)
        #self.x_mean_seq_dict = {0: 1e-8 * np.ones(self.seq_num, dtype=float)}
        
        # choice_2 (with linear regression)
        # linear regression of the first two time points:        
        if self.regression_num == 2:
            tmp = (self.read_freq_seq[:, 1] - self.read_freq_seq[:, 0]) / (self.t_seq[1] - self.t_seq[0])
        else:
            tmp = [regression_output.slope for i in range(self.lineages_num) for regression_output in
                   [linregress(self.t_seq[0:self.regression_num], np.log(self.read_freq_seq[i, 0:self.regression_num]))]]

        tmp = tmp - np.dot(self.read_freq_seq[:, 0], tmp)  # Normalization
        tmp = np.tile(tmp, (self.seq_num, 1)).T
        self.x_mean_seq = np.sum(tmp * self.read_freq_seq, axis=0)
        self.x_mean_seq_dict = {0: self.x_mean_seq}
        

        for k_iter in range(1, self.max_iter_num+1):
            start_iter = time.time()
            print('--- iteration {} ...'.format(k_iter))

            self.x_mean_seq = self.x_mean_seq_dict[k_iter-1]
            #self.x_mean_seq = np.exp(self.x_mean_seq) - 1
               
            output_result_old = {
                'FitSeq_Result': {
                    'Fitness': np.copy(self.result_s),
                    'Log_Likelihood_Value': np.copy(self.prior_loglikelihood),
                    'Mean_Fitness': np.copy(self.x_mean_seq_dict[k_iter-1]), # prior
                    'Kappa_Value': np.copy(self.kappa_seq),
                    'Inference_Time': np.copy(self.iter_timing_list)
                    },
                'Mean_fitness_Result': copy.deepcopy(self.x_mean_seq_dict),
                'Read_Number_Estimated': np.copy(self.read_num_seq_theory)
                }
               
            self.function_sum_term()
            self.function_run_iteration()
            #self.function_estimation_error()
            self.function_prior_loglikelihood_iteration()
            self.function_update_mean_fitness(k_iter)

            prior_loglikelihood_sum = np.sum(self.prior_loglikelihood)
            self.prior_loglikelihood_list.append(prior_loglikelihood_sum)
            #print('prior_loglikelihood', prior_loglikelihood_sum)
            if len(self.prior_loglikelihood_list) >= 2:
                stop_check = self.prior_loglikelihood_list[-1] - self.prior_loglikelihood_list[-2]
                print(stop_check)
                if stop_check < 0:
                    self.function_save_data(output_result_old)
                    break
                elif k_iter==self.max_iter_num:
                    self.function_save_data(output_result_old)
                
            end_iter = time.time()
            iter_timing = np.round(end_iter - start_iter, 5)
            self.iter_timing_list.append(iter_timing)
            print('    computing time: {} seconds'.format(iter_timing), flush=True)

        
        end = time.time()
        inference_timing = np.round(end - start, 5)
        print('Total computing time: {} seconds'.format(inference_timing), flush=True)




def process(
        read_num_seq,
        t_seq,
        cell_depth_seq,
        opt_algorithm,
        max_iter_num,
        parallelize,
        output_filenamepath_prefix
        ):
    """
    """
    my_obj = FitSeq(
        read_num_seq,
        t_seq,
        cell_depth_seq,
        opt_algorithm,
        max_iter_num,
        parallelize,
        output_filenamepath_prefix
        )
    
    my_obj.seq()
    
    
    
    
def main():
    """
    """
    parser = argparse.ArgumentParser(
        description='Estimate fitness of phenotypes in a competitive pooled growth experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='a .csv file: with each column being the read number per barcode at each sequenced time-point')

    parser.add_argument(
        '-t', '--t_seq',
        type=str,
        required=True,
        help='a .csv file of 2 columns:'
            '1st column: sequenced time-points evaluated in number of generations, '
            '2nd column: total effective number of cells of the population for each sequenced time-point.'
            )
                    
    parser.add_argument(
        '-a', '--opt_algorithm',
        type=str,
        default='bfgs',
        choices = ['differential_evolution', 'nelder_mead','bfgs'],
        help='choose optmization algorithm')

    
    parser.add_argument(
        '-n', '--maximum_iteration_number',
        type=int, default=50,
        help='maximum number of iterations, need to be >= 2'
        )

    parser.add_argument(
        '-p', '--parallelize',
        type=int,
        default=0,
        help='use multiprocess module to parallelize inference across lineages'
        )
                    
    parser.add_argument(
        '-o', '--output_filenamepath_prefix',
        type=str,
        default='output',
        help='prefix of output .csv files'
        )

    args = parser.parse_args()


    read_num_seq = np.array(pd.read_csv(args.input, header=None), dtype=float)

    csv_input = pd.read_csv(args.t_seq, header=None)
    t_seq = np.array(csv_input[0][~pd.isnull(csv_input[0])], dtype=float)
    cell_depth_seq = np.array(csv_input[1][~pd.isnull(csv_input[1])], dtype=float)
   
    parallelize = bool(int(args.parallelize))
    opt_algorithm = args.opt_algorithm
    output_filenamepath_prefix = args.output_filenamepath_prefix
        
    if args.maximum_iteration_number < 2:
        print('The maximum number of iterations need to be >=2, force changing it to 2!')
    max_iter_num = max(args.maximum_iteration_number, 2)
    
    process(
        read_num_seq,
        t_seq,
        cell_depth_seq,
        opt_algorithm,
        max_iter_num,
        parallelize,
        output_filenamepath_prefix
        )


if __name__=="__main__":
    main()
