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
from tqdm import tqdm
from multiprocess import Pool
from scipy.stats import nbinom


# Fitness inference object
class FitSeq2:
    def __init__(self,r,t,cell_depth,delta_t,c,opt_algorithm,max_iter_num,parallelize,output_prefix):
        
        # Preparing inputs
        self.r = r
        self.read_depth = np.sum(self.r, axis=0)
        self.lineages_num = np.shape(self.r)[0]
        self.t = t
        self.seq_num = len(self.t)
        self.cell_depth = cell_depth
        self.ratio = np.true_divide(self.read_depth, self.cell_depth)
        # self.r[self.r < 1] = 1 # pseudocounts

        self.delta_t = delta_t
        self.opt_algorithm = opt_algorithm
        self.max_iter_num = max_iter_num
        self.parallelize = parallelize
        self.output_prefix = output_prefix

        # Set bounds for the optimization
        self.bounds = Bounds(
            [1e-8, -1],
            [np.max(self.r[:,0])/self.ratio[0]*10, 1]
            )
        
        # Define other variables
        self.kappa = 2.5 * np.ones(self.seq_num)
        
        #self.regression_num = 2
        self.read_freq = self.r / self.read_depth

        # noise_c: noise per cycle
        self.noise_c = c*(self.t[1:] - self.t[:-1])/self.delta_t
        
        r_mean = np.mean(self.r[:,0])
        n_mean = self.cell_depth[0] / self.lineages_num
        self.noise_beta = (r_mean/n_mean + 2*r_mean/100 + 1)/2

    def calc_mean_s_decay(self):
        """
        Pre-calculate the contribution of the mean fitness to mean n, to reduce calculations in estimating the number of reads.
        """
        self.mean_s_decay = np.zeros(self.seq_num-1, dtype=float) # from tkminus1 to tk
        for k in range(self.seq_num-1):
            log_mean_s_decay = (self.t[k+1]-self.t[k]) * (self.s_mean[k+1] + self.s_mean[k])/2 
            self.mean_s_decay[k] =  np.exp(-log_mean_s_decay)    

    def E_term(self, s):
        """
        The factor governing the change in the n from one timepoint to the next, incorporating lineage fitness and mean fitness
        """
        E_term = np.zeros(self.seq_num-1, dtype=float) # from tkminus1 to tk
        for k in range(self.seq_num-1):
            E_term[k] =  self.mean_s_decay[k] * np.exp(s * (self.t[k+1]-self.t[k]))

        return E_term

    def traj_LL_from_n(self, n0, s):
        """"
        maximizes exponent in likelihood function by solving equations linear in the sqrt(n)
        Get log likelihood of a trajectory (after the initial r0)
        """
        E_term = self.E_term(s)
        E_term_sqrt = np.sqrt(E_term)

        gamma_vec = np.sqrt(self.r_lineage[1:])

        rho_vec = np.sqrt(self.ratio[1:])
        rho_square_vec = self.ratio[1:]
        
        b_vec = rho_vec*gamma_vec/self.noise_beta 
        b_vec[0] += E_term_sqrt[0] * np.sqrt(n0)/self.noise_c[0] 
        
        diag_center = 1/self.noise_c + rho_square_vec/self.noise_beta + np.concatenate(((E_term/self.noise_c)[1:], [0]))
        diag_side = (E_term_sqrt/self.noise_c)[1:]

        M_matrix = np.diag(diag_center) - np.diag(diag_side, k=1) - np.diag(diag_side, k=-1)
        
        #nu_vec = np.matmul(np.linalg.inv(M_matrix), b_vec)
        nu_vec = np.linalg.solve(M_matrix, b_vec)
        
        n_LL = (nu_vec - E_term_sqrt*np.concatenate(([np.sqrt(n0)], nu_vec[:-1])))**2/self.noise_c
        r_LL = (gamma_vec - rho_vec*nu_vec)**2 / self.noise_beta

        return {'optimal_value': -np.sum(n_LL + r_LL), 'optimal': np.square(nu_vec)}
        
    def traj_LL(self, n0, s):
        """
        Calculate negative log-likelihood value of a lineage trajectory, given s and n0.
        """
        beta = self.noise_beta
        r0_th = n0 * self.ratio[0] # theory
        r0_obs = self.r_lineage[0] # observed

        # use a negative binomial prior for n0 --- this models an overdispersed Poisson distribution where the
        # variance is a multiple of the mean as expected from a branching process
        r0_loglikelihood = nbinom.logpmf(r0_obs,r0_th/(beta-1),1/beta)
#         if r0_obs==0:
#             r0_loglikelihood = np.log(1/beta) - r0_th/beta # + np.log(r0_th/beta**2)
#         else:
#             ive_arg = 2*np.sqrt(r0_th*r0_obs)/beta
#             log_bessel = np.log(special.ive(1, ive_arg)) + ive_arg
#             r0_loglikelihood = np.log(1/beta) + 1/2*np.log(r0_th/r0_obs) - (r0_th+r0_obs)/beta + log_bessel

        integrand_log = self.traj_LL_from_n(n0, s)['optimal_value'] + r0_loglikelihood
        return integrand_log

    def objective_func(self, x):
        """
        Calculate log-likelihood value of a lineage given s and n0 in optimization
        """
        output = self.traj_LL(x[0], x[1])
        return -output # minimization only in python
     
    def optimize_n0_s(self, i): 
        """
        i: lineage label
        Output optimized s and n0 for each lineage i
        """
        self.r_lineage = self.r[i, :]
        
        if self.opt_algorithm == 'differential_evolution':
            opt_output = differential_evolution(
                func = self.objective_func,
                seed = 1,
                bounds = self.bounds
                )

        elif self.opt_algorithm == 'nelder_mead': 
            opt_output = minimize(
                self.objective_func,
                x0 = [100, 0.1] ,
                method = 'Nelder-Mead',
                bounds = self.bounds,
                options = {'ftol': 1e-8, 'disp': False, 'maxiter': 500}
                )

        n0_opt = opt_output.x[0]
        s_opt = opt_output.x[1]
            
        return [n0_opt , s_opt]
    
    def estimation_error_lineage(self,n0_opt,s_opt,i):
        """
        Estimate estimation error of a lineage for optimization
        """
        self.r_lineage = self.r[i, :]

        d_n0 = min(n0_opt,2e-2)/2
        d_s = 1e-4
    
        f_zero = self.objective_func([n0_opt, s_opt])

        f_plus_n0 = self.objective_func([n0_opt + d_n0, s_opt])
        f_minus_n0 = self.objective_func([n0_opt - d_n0, s_opt])
        
        f_plus_s = self.objective_func([n0_opt, s_opt + d_s])
        f_minus_s = self.objective_func([n0_opt, s_opt - d_s])
    
        f_plus_n0_s = self.objective_func([n0_opt + d_n0, s_opt + d_s])
        
        f_n0n0 = (f_plus_n0 + f_minus_n0 - 2*f_zero)/d_n0**2
        f_ss = (f_plus_s + f_minus_s - 2*f_zero)/d_s**2
        f_n0s = (f_plus_n0_s - f_plus_n0 - f_plus_s + f_zero)/d_s/d_n0
    
        curvature_matrix = np.array([[f_n0n0,f_n0s], [f_n0s,f_ss]])
        # print(curvature_matrix,flush=True)
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
   
    def estimation_error(self):
        for i in range(self.lineages_num):
            self.r_lineage = self.r[i, :]
            self.error_n0[i], self.error_s[i] = self.estimation_error_lineage(self.result_n0[i],self.result_s[i],i)
         
    def update_mean_fitness(self, k_iter):
        """
        Updated mean fitness
        """
        s_mean = np.sum(self.read_freq * np.tile(self.result_s, (self.seq_num, 1)).T, axis=0)
        self.s_mean_dict[k_iter] = s_mean - s_mean[0]
        #self.result_s = self.result_s - s_mean[0]
        
        self.n_theory = np.zeros(np.shape(self.r), dtype=float)
        for i in range(self.lineages_num):
            self.r_lineage = self.r[i, :]
            output = self.traj_LL_from_n(self.result_n0[i], self.result_s[i])
            self.n_theory[i,1:] = output['optimal']
            self.n_theory[i,0] = self.result_n0[i]
        self.r_theory = self.n_theory * self.ratio
        
        #s_mean_tmp = np.sum(self.n_theory * np.tile(self.result_s, (self.seq_num, 1)).T, axis=0)
        #s_mean = s_mean_tmp / np.sum(self.n_theory, axis=0)
        #self.s_mean_dict[k_iter] = s_mean - s_mean[0]
        
    def run_iteration(self):
        """
        Run a single interation
        Run optimization for each lineages to find their optimized s & n0
        """
        
        if self.parallelize:
            pool_obj = Pool() # might need to change processes=8
            output0 = pool_obj.map(self.optimize_n0_s, tqdm(range(self.lineages_num)))
            pool_obj.close()
            output = np.array(output0)
            self.result_n0 = output[:,0]
            self.result_s = output[:,1]
        else:
            self.result_n0 = np.zeros(self.lineages_num)
            self.result_s = np.zeros(self.lineages_num)
            for i in tqdm(range(self.lineages_num)):
                self.result_n0[i],self.result_s[i] = self.optimize_n0_s(i)
     
    def loglikelihood_iteration(self):
        """ """
        self.loglikelihood = np.zeros(self.lineages_num)
        for i in range(self.lineages_num):
            self.r_lineage = self.r[i, :]
            
            self.loglikelihood[i] = self.traj_LL(self.result_n0[i], self.result_s[i])

    def save_data(self, output_result):
        """
        Save data according to label: if it's saving a step or the final data
        """
        fitseq_result = output_result['FitSeq_Result']
        filenamepath = '{}_FitSeq2_Result.csv'.format(self.output_prefix)
        with open(filenamepath, 'w') as f:
            w = csv.writer(f)
            w.writerow(fitseq_result.keys())
            w.writerows(list(itertools.zip_longest(*list(fitseq_result.values()))))
 
        mean_fitness_result = output_result['Mean_fitness_Result']
        filenamepath = '{}_FitSeq2_Result_Mean_fitness.csv'.format(self.output_prefix)
        with open(filenamepath, 'w') as f:
            w = csv.writer(f)
            w.writerow(mean_fitness_result.keys())
            w.writerows(list(itertools.zip_longest(*list(mean_fitness_result.values()))))
        
        estimated_r = output_result['Read_Number_Estimated']

        # can save floats instead of ints to see how setting the minimum read count to 1 affects inferred n
        est_r_pd = pd.DataFrame(estimated_r)#.astype(int))
        filenamepath = '{}_FitSeq2_Result_Read_Number_Estimated.csv'.format(self.output_prefix)
        est_r_pd.to_csv(filenamepath, index=False, header=False)
         
    def infer_fitnesses(self):
        """
        main function
        """
        start = time.time()
        self.calculate_error = False
        self.loglikelihood_list = []
        self.iter_timing_list = [] # running time for each iteration
        
        #####
        self.result_s = np.zeros(self.lineages_num, dtype=float)
        self.result_n0 = np.zeros(self.lineages_num, dtype=float)
        
        self.error_s = np.zeros(self.lineages_num, dtype=float)
        self.error_n0 = np.zeros(self.lineages_num, dtype=float)
        
        self.loglikelihood = np.zeros(self.lineages_num, dtype=float)
        self.r_theory = np.zeros((self.lineages_num, self.seq_num), dtype=float)
        
        # choice_1 (without linear regression)
        self.s_mean_dict = {0: 1e-8 * np.ones(self.seq_num, dtype=float)}
        
        # choice_2 (with linear regression)
        # linear regression of the first two time points:
        # if self.regression_num == 2:
        #    tmp = (self.read_freq[:, 1] - self.read_freq[:, 0]) / (self.t[1] - self.t[0])
        # else:
        #    tmp = [regression_output.slope for i in range(self.lineages_num) for regression_output in
        #           [linregress(self.t[0:self.regression_num], np.log(self.read_freq[i, 0:self.regression_num]))]]

        # tmp = tmp - np.dot(self.read_freq[:, 0], tmp)  # Normalization
        # tmp = np.tile(tmp, (self.seq_num, 1)).T
        # self.s_mean = np.sum(tmp * self.read_freq, axis=0)
        # self.s_mean_dict = {0: self.s_mean}
        

        for k_iter in range(1, self.max_iter_num+1):
            start_iter = time.time()
            print('--- iteration {} ...'.format(k_iter),flush=True)

            self.s_mean = self.s_mean_dict[k_iter-1]
     
            # output result for fitness in terms of growth rate per cycle
            output_result = {
                'FitSeq_Result': {
                    'Fitness_Per_Cycle': self.result_s*self.delta_t,
                    'Error_Fitness': self.error_s*self.delta_t,
                    'Cell_Number': self.result_n0,
                    'Error_Cell_Number': self.error_n0,
                    'Log_Likelihood_Value': self.loglikelihood,
                    'Mean_Fitness': self.s_mean_dict[k_iter-1]*self.delta_t, # prior
                    'Kappa_Value': self.kappa,
                    'Inference_Time': self.iter_timing_list
                    },
                'Mean_fitness_Result': {k:self.s_mean_dict[k]*self.delta_t for k in self.s_mean_dict},
                'Read_Number_Estimated': self.r_theory
                }
               
            self.calc_mean_s_decay()
            self.run_iteration()
            self.estimation_error()
            self.loglikelihood_iteration()
            self.update_mean_fitness(k_iter)
            
            loglikelihood_sum = np.sum(self.loglikelihood)
            self.loglikelihood_list.append(loglikelihood_sum)
            if len(self.loglikelihood_list) >= 2:
                stop_check = self.loglikelihood_list[-1] - self.loglikelihood_list[-2]
                print(stop_check)
                if stop_check < 0:
                    self.save_data(output_result)      
                    break
                elif k_iter==self.max_iter_num:
                    self.save_data(output_result)
 
            end_iter = time.time()
            iter_timing = np.round(end_iter - start_iter, 5)
            self.iter_timing_list.append(iter_timing)
            print('    computing time: {} seconds'.format(iter_timing), flush=True)
        
        
        end = time.time()
        inference_timing = np.round(end - start, 5)
        print('Total computing time: {} seconds'.format(inference_timing), flush=True)

def process(r,t,cell_depth,delta_t,c,opt_algorithm,max_iter_num,parallelize,output_prefix):
    my_obj = FitSeq2(r,t,cell_depth,delta_t,c,opt_algorithm,max_iter_num,parallelize,output_prefix)
    my_obj.infer_fitnesses()
    
def main():
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
        '-t', '--t',
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
        help='maximum number of iterations, need to be >= 2'
        )

    parser.add_argument(
        '-p', '--parallelize',
        type=int,
        default=0,
        help='use multiprocess module to parallelize inference across lineages'
        )
                    
    parser.add_argument(
        '-o', '--output_prefix',
        type=str,
        default='output',
        help='prefix of output .csv files'
        )

    args = parser.parse_args()
    
    r = np.array(pd.read_csv(args.input, header=None), dtype=float)

    csv_input = pd.read_csv(args.t, header=None)
    t = np.array(csv_input[0][~pd.isnull(csv_input[0])], dtype=float)
    cell_depth = np.array(csv_input[1][~pd.isnull(csv_input[1])], dtype=float)

    delta_t = args.delta_t
    c = args.c # per cycle
    parallelize = bool(int(args.parallelize))
    opt_algorithm = args.opt_algorithm
    output_prefix = args.output_prefix
    
    if args.maximum_iteration_number < 2:
        print('The maximum number of iterations need to be >=2, force changing it to 2!')
    max_iter_num = max(args.maximum_iteration_number, 2)
    
    process(r,t,cell_depth,delta_t,c,opt_algorithm,max_iter_num,parallelize,output_prefix)

if __name__=="__main__":
    main()
