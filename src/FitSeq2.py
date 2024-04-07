#!/usr/bin/env python3
import csv
import copy
import time
import itertools
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocess import Pool
from scipy import special
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import differential_evolution



# fitness inference object
class FitSeq2:
    def __init__(self,
            read_num_seq,
            t_seq,
            cell_depth_seq,
            delta_t,
            c,
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

        self.delta_t = delta_t
        self.opt_algorithm = opt_algorithm
        self.max_iter_num = max_iter_num
        self.parallelize = parallelize
        self.output_filenamepath_prefix = output_filenamepath_prefix

        # Set bounds for the optimization
        self.bounds = Bounds(
            [1e-8, -1],
            [np.max(self.read_num_seq[:,0])/self.ratio[0]*10, 1]
            )
        
        # Define other variables
        self.kappa_seq = 2.5 * np.ones(self.seq_num)
        
        #self.regression_num = 2
        self.read_freq_seq = self.read_num_seq / self.read_depth_seq

        #self.noise_c = c / self.delta_t
        # noise_c: noise per generation
        # c: noise per cycle
        
        self.noise_c_seq = (self.t_seq[1:] - self.t_seq[:-1]) / self.delta_t * c
        # noise_c_seq: noise per cycle
        
        read_num_mean = np.mean(self.read_num_seq[:,0])
        cell_num_mean = self.cell_depth_seq[0] / self.lineages_num
        self.noise_beta = (read_num_mean/cell_num_mean + 2*read_num_mean/100 + 1)/2
        
        


    def function_sum_term(self):
        """
        Pre-calculate a term (i.e. sum_term) to reduce calculations in estimating the number of reads.
        """
        self.sum_term_t_seq = np.zeros(self.seq_num-1, dtype=float) # from tkminus1 to tk
        for k in range(self.seq_num-1):
            sum_term_t_seq_tmp = (self.t_seq[k+1]-self.t_seq[k]) * (self.x_mean_seq[k+1] + self.x_mean_seq[k])/2 
            self.sum_term_t_seq[k] =  np.exp(-sum_term_t_seq_tmp)    



    def function_epsilon_term(self, s):
        """ """
        epsilon_term_t_seq = np.zeros(self.seq_num-1, dtype=float) # from tkminus1 to tk
        for k in range(self.seq_num-1):
            sum_term_tkplus1_minus_tk = self.sum_term_t_seq[k]
            tmp_1 = np.exp(s * (self.t_seq[k+1]-self.t_seq[k]))
            epsilon_term_t_seq[k] =  sum_term_tkplus1_minus_tk * tmp_1
        
        epsilon_term_sqrt_t_seq = np.sqrt(epsilon_term_t_seq)
        
        return {0: epsilon_term_t_seq, 1: epsilon_term_sqrt_t_seq}




    def function_prior_loglikelihood_scaler_subfunction(self, n0, s):
        """" """
        output_epsilon_term = self.function_epsilon_term(s)  # need to add this!!!
        epsilon_term_t_seq = output_epsilon_term[0]
        epsilon_term_sqrt_t_seq = output_epsilon_term[1]

        gamma_vec = np.sqrt(self.read_num_seq_lineage[1:])

        rho_vec = np.sqrt(self.ratio[1:])
        rho_square_vec = self.ratio[1:]
        
        b_vector = np.multiply(rho_vec, gamma_vec) / self.noise_beta 
        b_vector[0] += epsilon_term_sqrt_t_seq[0] * np.sqrt(n0) /self.noise_c_seq[0] 
        
        tmp1 = np.true_divide(1, self.noise_c_seq) 
        tmp2 = rho_square_vec / self.noise_beta
        tmp3 = np.true_divide(epsilon_term_t_seq, self.noise_c_seq)
        diag_center = tmp1 + tmp2 + np.concatenate((tmp3[1:], [0]))
        
        tmp4 = np.true_divide(epsilon_term_sqrt_t_seq, self.noise_c_seq)
        diag_side = tmp4[1:]

        M_matrix = np.diag(diag_center) - np.diag(diag_side, k=1) - np.diag(diag_side, k=-1)
        
        #nu_vec = np.matmul(np.linalg.inv(M_matrix), b_vector)
        nu_vec = np.linalg.solve(M_matrix, b_vector)
        
        tmp_a = np.multiply(epsilon_term_sqrt_t_seq, np.concatenate(([np.sqrt(n0)], nu_vec[:-1])))
        tmp_b = np.true_divide(np.power(nu_vec - tmp_a, 2), self.noise_c_seq)

        tmp_c = np.multiply(rho_vec, nu_vec)
        tmp_d = np.power(gamma_vec - tmp_c, 2) / self.noise_beta

        output = -np.sum(tmp_b + tmp_d)

        return {'optimal_value': output, 'optimal': np.square(nu_vec)}
        


    def function_prior_loglikelihood_scaler(self, n0, s):
        """
        Calculate log-likelihood value of a lineage given s and n0.
        Inputs: s (scaler)
        Output: log-likelihood value of all time poins (scaler)
        """
        tmp_kappa_reverse = 1/self.noise_beta
        tmp_theory = n0 * self.ratio[0]
        tmp_observed = self.read_num_seq_lineage[0]
        tmp_observed_reverse = 1/tmp_observed
        ive_ele = 2* np.multiply(np.sqrt(np.multiply(tmp_theory, tmp_observed)), tmp_kappa_reverse)
        tmp_part1 = np.log(tmp_kappa_reverse)
        tmp_part2 = 1/2 * np.log(np.multiply(tmp_theory, tmp_observed_reverse))
        tmp_part3 = -np.multiply(tmp_theory + tmp_observed, tmp_kappa_reverse)
        tmp_part4 = np.log(special.ive(1, ive_ele)) + ive_ele

        f_r0_c_n0_log = tmp_part1 + tmp_part2 + tmp_part3 + tmp_part4

        tmp = self.function_prior_loglikelihood_scaler_subfunction(n0, s)
        integrand_log = tmp['optimal_value'] + f_r0_c_n0_log
            
        return integrand_log
     
    

    def function_prior_loglikelihood_opt(self, x):
        """
        Calculate log-likelihood value of a lineage given s and n0 in optimization
        """
        output = self.function_prior_loglikelihood_scaler(x[0], x[1])

        return -output #minimization only in python



    def function_parallel(self, i): 
        """
        i: lineage label
        Output optimized s and n0 for each lineage i
        """
        self.read_num_seq_lineage = self.read_num_seq[i, :]
        
        if self.opt_algorithm == 'differential_evolution':
            opt_output = differential_evolution(
                func = self.function_prior_loglikelihood_opt,
                seed = 1,
                bounds = self.bounds
                )

        elif self.opt_algorithm == 'nelder_mead': 
            opt_output = minimize(
                self.function_prior_loglikelihood_opt,
                x0 = [100, 0.1] ,
                method = 'Nelder-Mead',
                bounds = self.bounds,
                options = {'ftol': 1e-8, 'disp': False, 'maxiter': 500}
                )

        n0_opt = opt_output.x[0]
        s_opt = opt_output.x[1]
            
        return [n0_opt , s_opt]

    
    
    def function_estimation_error_lineage(self, n0_opt, s_opt):
        """
        Estimate estimation error of a lineage for optimization
        """
        d_n0, d_s = 1e-6, 1e-8
    
        f_zero = self.function_prior_loglikelihood_opt([n0_opt, s_opt])

        f_plus_n0 = self.function_prior_loglikelihood_opt([n0_opt + d_n0, s_opt])
        f_minus_n0 = self.function_prior_loglikelihood_opt([n0_opt - d_n0, s_opt])
        
        f_plus_s = self.function_prior_loglikelihood_opt([n0_opt, s_opt + d_s])
        f_minus_s = self.function_prior_loglikelihood_opt([n0_opt, s_opt - d_s])
    
        f_plus_n0_s = self.function_prior_loglikelihood_opt([n0_opt + d_n0, s_opt + d_s])
        
        f_n0n0 = (f_plus_n0 + f_minus_n0 - 2*f_zero)/d_n0**2
        f_ss = (f_plus_s + f_minus_s - 2*f_zero)/d_s**2
        f_n0s = (f_plus_n0_s - f_plus_n0 - f_plus_s + f_zero)/d_s/d_n0
    
        curvature_matrix = np.array([[f_n0n0,f_n0s], [f_n0s,f_ss]])
        eigs, eigvecs = np.linalg.eig(curvature_matrix)
        v1, v2 = eigvecs[:,0], eigvecs[:,1]
        lambda1, lambda2 = np.abs(eigs[0]), np.abs(eigs[1])
        
        error_n0_lineage =  max(
            np.abs(v1[0]/np.sqrt(lambda1)),
            np.abs(v2[0]/np.sqrt(lambda2))
            )
        error_s_lineage = max(
            np.abs(v1[1]/np.sqrt(lambda1)),
            np.abs(v2[1]/np.sqrt(lambda2))
            )

        return error_n0_lineage, error_s_lineage

    
   
    def function_estimation_error(self):
        for i in range(self.lineages_num):
            self.read_num_seq_lineage = self.read_num_seq[i, :]
                
            s_opt = self.result_s[i]
            n0_opt = self.result_n0[i]
            self.error_n0[i], self.error_s[i] = self.function_estimation_error_lineage(n0_opt, s_opt)
         
   


    ##########
    def function_update_mean_fitness(self, k_iter):
        """
        Updated mean fitness
        """
        x_mean = np.sum(self.read_freq_seq * np.tile(self.result_s, (self.seq_num, 1)).T, axis=0)
        self.x_mean_seq_dict[k_iter] = x_mean - x_mean[0]
        #self.result_s = self.result_s - x_mean[0]
        
        self.cell_num_seq_theory = np.zeros(np.shape(self.read_num_seq), dtype=float)
        for i in range(self.lineages_num):
            self.read_num_seq_lineage = self.read_num_seq[i, :]
            output = self.function_prior_loglikelihood_scaler_subfunction(self.result_n0[i], self.result_s[i])
            self.cell_num_seq_theory[i,1:] = output['optimal']
            self.cell_num_seq_theory[i,0] = self.result_n0[i]
        self.read_num_seq_theory = self.cell_num_seq_theory * self.ratio
        
        #x_mean_tmp = np.sum(self.cell_num_seq_theory * np.tile(self.result_s, (self.seq_num, 1)).T, axis=0)
        #x_mean = x_mean_tmp / np.sum(self.cell_num_seq_theory, axis=0)
        #self.x_mean_seq_dict[k_iter] = x_mean - x_mean[0]
        

        

    def function_run_iteration(self):
        """
        Run a single interation
        Run optimization for each lineages to find their optimized s & n0
        """
        
        if self.parallelize:
            pool_obj = Pool() # might need to change processes=8
            output_tmp = pool_obj.map(self.function_parallel, tqdm(range(self.lineages_num)))
            pool_obj.close()
            output = np.array(output_tmp)           
            self.result_n0 = output[:,0]
            self.result_s = output[:,1]
        else:
            self.result_n0 = np.zeros(self.lineages_num)
            self.result_s = np.zeros(self.lineages_num)
            for i in range(self.lineages_num):
                output = self.function_parallel(i)
                self.result_n0[i] = output[0]
                self.result_s[i] = output[1]
        

     
    def function_prior_loglikelihood_iteration(self):
        """ """
        self.prior_loglikelihood = np.zeros(self.lineages_num)
        for i in range(self.lineages_num):
            self.read_num_seq_lineage = self.read_num_seq[i, :]
            
            self.prior_loglikelihood[i] = self.function_prior_loglikelihood_scaler(self.result_n0[i], self.result_s[i])



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
        filenamepath = '{}_Mean_fitness_Result.csv'.format(self.output_filenamepath_prefix)
        with open(filenamepath, 'w') as f:
            w = csv.writer(f)
            w.writerow(tmp_2.keys())
            w.writerows(tmp)
        
        tmp_3 = output_result['Read_Number_Estimated']
        tmp = pd.DataFrame(tmp_3.astype(int))
        filenamepath = '{}_Read_Number_Estimated.csv'.format(self.output_filenamepath_prefix)
        tmp.to_csv(filenamepath, index=False, header=False)
         
        
    def seq2(self):
        """
        main function
        """
        start = time.time()
        self.calculate_error = False
        self.prior_loglikelihood_list = []
        self.iter_timing_list = [] # running time for each iteration
        
        #####
        self.result_s = np.zeros(self.lineages_num, dtype=float)
        self.result_n0 = np.zeros(self.lineages_num, dtype=float)
        
        self.error_s = np.zeros(self.lineages_num, dtype=float)
        self.error_n0 = np.zeros(self.lineages_num, dtype=float)
        
        self.prior_loglikelihood = np.zeros(self.lineages_num, dtype=float)
        self.read_num_seq_theory = np.zeros((self.lineages_num, self.seq_num), dtype=float)
        
        # choice_1 (without linear regression)
        self.x_mean_seq_dict = {0: 1e-8 * np.ones(self.seq_num, dtype=float)}
        
        # choice_2 (with linear regression)
        # linear regression of the first two time points:
        #if self.regression_num == 2:
        #    tmp = (self.read_freq_seq[:, 1] - self.read_freq_seq[:, 0]) / (self.t_seq[1] - self.t_seq[0])
        #else:
        #    tmp = [regression_output.slope for i in range(self.lineages_num) for regression_output in
        #           [linregress(self.t_seq[0:self.regression_num], np.log(self.read_freq_seq[i, 0:self.regression_num]))]]

        #tmp = tmp - np.dot(self.read_freq_seq[:, 0], tmp)  # Normalization
        #tmp = np.tile(tmp, (self.seq_num, 1)).T
        #self.x_mean_seq = np.sum(tmp * self.read_freq_seq, axis=0)
        #self.x_mean_seq_dict = {0: self.x_mean_seq}
        

        for k_iter in range(1, self.max_iter_num+1):
            start_iter = time.time()
            print(f'--- iteration {k_iter} ...')

            self.x_mean_seq = self.x_mean_seq_dict[k_iter-1]
     
            output_result_old = {
                'FitSeq_Result': {
                    'Fitness': np.copy(self.result_s),
                    #'Error_Fitness': np.copy(self.error_s),
                    'Cell_Number': np.copy(self.result_n0),
                    #'Error_Cell_Number': np.copy(self.error_n0),
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



def process(
        read_num_seq,
        t_seq,
        cell_depth_seq,
        delta_t,
        c,
        opt_algorithm,
        max_iter_num,
        parallelize,
        output_filenamepath_prefix
        ):
    """
    """
    my_obj = FitSeq2(
        read_num_seq,
        t_seq,
        cell_depth_seq,
        delta_t,
        c,
        opt_algorithm,
        max_iter_num,
        parallelize,
        output_filenamepath_prefix
        )
    
    my_obj.seq2()
    
    
    

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
        help='a .csv file: with each column being the read number per barcode at each sequenced time-point'
        )

    parser.add_argument(
    '-t', '--t_seq',
    type=str,
    required=True,
    help='a .csv file of 2 columns:'
         '1st column: sequenced time-points evaluated in number of generations, '
         '2nd column: total effective number of cells of the population for each sequenced time-point.'
         )

    parser.add_argument(
        '-dt', '--delta_t',
        type=float,
        required=True,
        help='number of generations between bottlenecks'
        )

    parser.add_argument(
        '-c', '--c',
        type=float,
        default=1,
        help='half of variance introduced by cell growth and cell transfer'
        )

    parser.add_argument(
        '-a', '--opt_algorithm',
        type=str,
        default='differential_evolution',
        choices = ['differential_evolution', 'nelder_mead'],
        help='choose optmization algorithm'
        )

    parser.add_argument(
        '-n', '--maximum_iteration_number',
        type=int,
        default=50,
        help='maximum number of iterations'
        )

    parser.add_argument(
        '-p', '--parallelize',
        type=bool,
        default=True,
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

    delta_t = args.delta_t
    c = args.c # per cycle
    parallelize = args.parallelize
    opt_algorithm = args.opt_algorithm
    max_iter_num = args.maximum_iteration_number
    output_filenamepath_prefix = args.output_filenamepath_prefix

    process(
        read_num_seq,
        t_seq,
        cell_depth_seq,
        delta_t,
        c,
        opt_algorithm,
        max_iter_num,
        parallelize,
        output_filenamepath_prefix
        )
        

if __name__=="__main__":
    main()
