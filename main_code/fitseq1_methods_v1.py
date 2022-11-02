#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy import special
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import differential_evolution
import itertools
import csv
import time
from multiprocess import Pool
from tqdm import tqdm
from scipy.stats import linregress
import copy


# fitness inference object
class FitSeq:
    def __init__(self, read_num_seq,
                       t_seq,
                       cell_depth_seq,
                       opt_algorithm,
                       max_iter_num,
                       parallelize,
                       output_filename):
        
        # preparing inputs
        self.read_num_seq = read_num_seq
        self.read_depth_seq = np.sum(self.read_num_seq, axis=0)
        self.lineages_num = np.shape(self.read_num_seq)[0]
        self.t_seq = t_seq
        self.seq_num = len(self.t_seq)
        self.cell_depth_seq = cell_depth_seq
        self.ratio = np.true_divide(self.read_depth_seq, self.cell_depth_seq)
        self.cell_num_seq = self.read_num_seq / self.ratio
        self.cell_num_seq[self.cell_num_seq < 1] = 1 
        self.read_num_seq[self.read_num_seq < 1] = 1 # approximate distribution is not accurate when r < 1

        self.opt_algorithm = opt_algorithm
        self.max_iter_num = max_iter_num
        self.parallelize = parallelize
        self.output_filename = output_filename

        # set bounds for the optimization
        self.bounds = Bounds(-1, 1)

        # define other variables
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



    ##########
    def function_cell_num_theory_scaler(self, s):
        """
        Estimate cell number & mutant cell number all time points for a lineage given s and tau. 
        Inputs: s (scaler)
        Output: {'cell_number': (array, vector)}
        """            
        cell_num_seq_lineage_observed = self.cell_num_seq_lineage
        
        cell_num_seq_lineage_theory = np.zeros(self.seq_num, dtype=float)
        cell_num_seq_lineage_theory[0] = cell_num_seq_lineage_observed[0]
        
        for k in range(1, self.seq_num):
            sum_term_tk_minus_tkminus1 = self.sum_term_t_seq[k-1]
            all_tmp1 = np.exp(s * (self.t_seq[k] - self.t_seq[k-1])) * sum_term_tk_minus_tkminus1
            cell_num_seq_lineage_theory[k] = np.multiply(cell_num_seq_lineage_observed[k-1], all_tmp1)       
            
        output = {'cell_number': cell_num_seq_lineage_theory}

        return output
    


    ##########
    def function_cell_num_theory_array(self, s_array):
        """
        Estimate cell number & mutant cell number all time points for a lineage given s and tau. 
        Inputs: s_array (array, vector)
        Output: {'cell_number': (array, 2D matrix)}
        """
        s_len = len(s_array)
            
        cell_num_seq_lineage_observed = np.tile(self.cell_num_seq_lineage, (s_len, 1))
        
        cell_num_seq_lineage_theory = np.zeros((s_len, self.seq_num), dtype=float)
        cell_num_seq_lineage_theory[:,0] = cell_num_seq_lineage_observed[:,0]

        for k in range(1, self.seq_num):
            sum_term_tk_minus_tkminus1 = self.sum_term_t_seq[k-1]
            all_tmp1 = np.exp(s_array * (self.t_seq[k] - self.t_seq[k-1])) * sum_term_tk_minus_tkminus1
            cell_num_seq_lineage_theory[:,k] = np.multiply(cell_num_seq_lineage_observed[:,k-1], all_tmp1)       
    
        output = {'cell_number': cell_num_seq_lineage_theory}

        return output



    ##########
    def function_prior_loglikelihood_scaler(self, s):
        """
        Calculate log-likelihood value of a lineage given s and tau.
        Inputs: s(scaler)
        Output: log-likelihood value of all time poins (scaler)
        """        
        output = self.function_cell_num_theory_scaler(s)
        cell_num_seq_lineage_theory = output['cell_number']
        read_num_seq_lineage_theory = np.multiply(cell_num_seq_lineage_theory, self.ratio)
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
        Calculate log-likelihood value of a lineage given s and tau.
        Inputs: s_array (array, vector)
        Output: log-likelihood value of all time poins (array, vector)
        """
        s_len = len(s_array)

        output = self.function_cell_num_theory_array(s_array)
        cell_num_seq_lineage_theory = output['cell_number'] #(s_len, eq_num)
        read_num_seq_lineage_theory = np.multiply(cell_num_seq_lineage_theory, np.tile(self.ratio, (s_len, 1)))
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
        Inputs: s(scaler)
        Output: log-likelihood value of all time poins (scaler)
        """
        output = self.function_cell_num_theory_scaler(s)
        cell_num_seq_lineage_theory = output['cell_number']
        read_num_seq_lineage_theory = np.multiply(cell_num_seq_lineage_theory, self.ratio)
        read_num_seq_lineage_theory[read_num_seq_lineage_theory < 1] = 1
        

        log_likelihood_seq_lineage = (0.25 * np.log(read_num_seq_lineage_theory) - 0.5 * np.log(4 * np.pi * self.kappa_seq)
                                      - 0.75 * np.log(self.read_num_seq_lineage)
                                      - (np.sqrt(self.read_num_seq_lineage) - np.sqrt(read_num_seq_lineage_theory)) ** 2 / self.kappa_seq)
        
        log_likelihood_lineage = np.sum(log_likelihood_seq_lineage, axis=0)

        return log_likelihood_lineage




    ##########
    def function_prior_loglikelihood_array_approx(self, s_array):
        """
        Calculate log-likelihood value of a lineage given s and tau.
        Inputs: s_array (array, vector)
        Output: log-likelihood value of all time poins (array, vector)
        """
        s_len = len(s_array)

        output = self.function_cell_num_theory_array(s_array)
        cell_num_seq_lineage_theory = output['cell_number']
        read_num_seq_lineage_theory = np.multiply(cell_num_seq_lineage_theory, np.tile(self.ratio, (s_len, 1)))
        read_num_seq_lineage_theory[read_num_seq_lineage_theory < 1] = 1
        
        tmp_kappa = np.tile(self.kappa_seq, (s_len, 1))
        tmp_theory = read_num_seq_lineage_theory
        tmp_observed = np.tile(self.read_num_seq_lineage, (s_len, 1))
        
        log_likelihood_seq_lineage = (0.25 * np.log(tmp_theory) - 0.5 * np.log(4 * np.pi * tmp_kappa)
                                      - 0.75 * np.log(tmp_observed)
                                      - (np.sqrt(tmp_observed) - np.sqrt(tmp_theory)) ** 2 / tmp_kappa)
        
        log_likelihood_lineage = np.sum(log_likelihood_seq_lineage, axis=1)

        return log_likelihood_lineage



    ##########
    def function_prior_loglikelihood_opt(self, s):
        """
        Calculate posterior log-likelihood value of a lineage given s and tau in optimization
        """
        #output = self.function_prior_loglikelihood_scaler(s)
        output = self.function_prior_loglikelihood_scaler_approx(s)

        return -output #minimization only in python



    ##########
    def function_parallel(self, i): 
        """
        i: lineage label
        calculate probability first, then for adaptive lineage output optimized s and tau
        """
        self.read_num_seq_lineage = self.read_num_seq[i, :]
        self.cell_num_seq_lineage = self.cell_num_seq[i, :]
        
        if self.opt_algorithm == 'differential_evolution':
            opt_output = differential_evolution(func = self.function_prior_loglikelihood_opt,
                                                seed = 1,
                                                bounds = self.bounds)
            
        elif self.opt_algorithm == 'nelder_mead': 
            opt_output = minimize(self.function_prior_loglikelihood_opt, 
                                  x0=[0.001], 
                                  method='Nelder-Mead', 
                                  bounds=self.bounds, 
                                  options={'ftol': 1e-8, 'disp': False, 'maxiter': 500})

        elif self.opt_algorithm == 'bfgs': 
            opt_output = minimize(self.function_prior_loglikelihood_opt, 
                                  x0=[0.001], 
                                  method='L-BFGS-B', 
                                  bounds=self.bounds, 
                                  options={'ftol': 1e-8, 'disp': False, 'maxiter': 500})
                                  
        s_opt = opt_output.x[0]
    
        return s_opt

    

    ##########
    def function_update_mean_fitness(self, k_iter):
        """
        Updated mean fitness & mutant fraction
        """
        self.mutant_fraction_numerator = np.zeros(self.seq_num, dtype=float)
        self.cell_num_seq_theory = np.zeros(np.shape(self.read_num_seq), dtype=float)
       
        for i in range(self.lineages_num):
            self.read_num_seq_lineage = self.read_num_seq[i, :]
            self.cell_num_seq_lineage = self.cell_num_seq[i, :]
            output = self.function_cell_num_theory_scaler(self.result_s[i])
            self.cell_num_seq_theory[i,:] = output['cell_number']
        
        #x_mean_tmp = np.sum(self.cell_num_seq_theory * np.tile(self.result_s, (self.seq_num, 1)).T, axis=0)
        #x_mean = x_mean_tmp / np.sum(self.cell_num_seq_theory, axis=0)
        x_mean_tmp = np.sum(self.cell_num_seq * np.tile(self.result_s, (self.seq_num, 1)).T, axis=0)
        x_mean = x_mean_tmp / np.sum(self.cell_num_seq, axis=0)

        self.x_mean_seq_dict[k_iter] = x_mean - x_mean[0]
        #self.x_mean_seq_dict[k_iter] = x_mean

        self.read_num_seq_theory = self.cell_num_seq_theory * self.ratio

        
    

    ##########
    def function_run_iteration(self):
        """
        run a single interation
        """
        # Calculate proability for each lineage to find adaptive lineages, 
        # Then run optimization for adaptive lineages to find their optimized s & tau for adaptive lineages
        if self.parallelize:
            pool_obj = Pool() # might need to change processes=8
            output_tmp = pool_obj.map(self.function_parallel, tqdm(range(self.lineages_num)))
            pool_obj.close()
            self.result_s = np.array(output_tmp)

        else:
            for i in range(self.lineages_num):
                output = self.function_parallel(i)
                self.result_s[i] = output
        

    

    ##########
    def function_prior_loglikelihood_iteration(self):
        """ """
        self.prior_loglikelihood = np.zeros(self.lineages_num)
        for i in range(self.lineages_num):
            self.read_num_seq_lineage = self.read_num_seq[i, :]
            self.cell_num_seq_lineage = self.cell_num_seq[i, :]
            self.prior_loglikelihood[i] = self.function_prior_loglikelihood_scaler(self.result_s[i])



    #####
    def function_save_data(self, output_result):
        """
        Save data according to label: if it's saving a step or the final data
        """
        tmp_1 = output_result['FitSeq_Result']
        tmp = list(itertools.zip_longest(*list(tmp_1.values())))
        with open(self.output_filename + '_FitSeq_Result.csv', 'w') as f:
            w = csv.writer(f)
            w.writerow(tmp_1.keys())
            w.writerows(tmp)
 
        tmp_2 = output_result['Mean_fitness_Result']
        tmp = list(itertools.zip_longest(*list(tmp_2.values())))
        with open(self.output_filename + '_Mean_fitness_Result.csv', 'w') as f:
            w = csv.writer(f)
            w.writerow(tmp_2.keys())
            w.writerows(tmp)
        
        tmp_3 = output_result['Read_Number_Estimated']
        tmp = pd.DataFrame(tmp_3.astype(int))
        tmp.to_csv(self.output_filename + '_Read_Number_Estimated.csv',
                   index=False, header=False)
                


    #####
    def function_main(self):
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
        

        # change this to the following:
        # linear regression of the first two time points:        
        if self.regression_num == 2:
            tmp = (self.read_freq_seq[:, 1] - self.read_freq_seq[:, 0]) / (self.t_seq[1] - self.t_seq[0])
        else:
            tmp = [regression_output.slope for i in range(self.lineages_num) for regression_output in
                   [linregress(self.t_seq[0:self.regression_num], np.log(self.read_freq_seq[i, 0:self.regression_num]))]]

        tmp = tmp - np.dot(self.read_freq_seq[:, 0], tmp)  # Normalization
        tmp = np.tile(tmp, (self.seq_num, 1)).T
        self.x_mean_seq = np.sum(tmp * self.read_freq_seq, axis=0)
        #self.x_mean_seq = np.exp(self.x_mean_seq) - 1

        #self.x_mean_seq_dict = {0: 1e-8 * np.ones(self.seq_num, dtype=float)}
        self.x_mean_seq_dict = {0: self.x_mean_seq}

        for k_iter in range(1, self.max_iter_num+1):
            start_iter = time.time()
            print(f'--- iteration {k_iter} ...')

            self.x_mean_seq = self.x_mean_seq_dict[k_iter-1]
            #self.x_mean_seq = np.exp(self.x_mean_seq) - 1
               
            output_result_old = {'FitSeq_Result': {'Fitness': np.copy(self.result_s),
                                                    'Log_Likelihood_Value': np.copy(self.prior_loglikelihood), 
                                                    'Mean_Fitness': np.copy(self.x_mean_seq_dict[k_iter-1]), # prior
                                                    'Kappa_Value': np.copy(self.kappa_seq), 
                                                    'Inference_Time': np.copy(self.iter_timing_list)}, 
                                 'Mean_fitness_Result': copy.deepcopy(self.x_mean_seq_dict),
                                 'Read_Number_Estimated': np.copy(self.read_num_seq_theory)}
               
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
                
            end_iter = time.time()
            iter_timing = np.round(end_iter - start_iter, 5)
            self.iter_timing_list.append(iter_timing)
            print(f'    computing time: {iter_timing} seconds', flush=True)

        
        end = time.time()
        inference_timing = np.round(end - start, 5)
        print(f'Total computing time: {inference_timing} seconds',flush=True)
