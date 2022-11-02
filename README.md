[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Contact Info](https://img.shields.io/badge/Contact-fangfeili0525@gmail.com-blue.svg)]()

# FitSeq2.0

## 1. What is FitSeq2.0?

FitSeq2.0 is a Python-based fitness estimation tool for pooled amplicon sequencing studies. It is an improved version of the MATLAB tool [FitSeq](https://github.com/sashaflevy/Fit-Seq). FitSeq2.0 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

It currently has two main functions:
* `fitseqsimu_run_v1.py` simulates the entire experimental process of competitve pooled growth of a population of genotypes.
* `fitseq2_run_v1.py` calculates the fitness of each genotype from read-count time-series data.

A walk-through is included as the jupyter notebook [here](https://github.com/FangfeiLi05/FitSeq2.0/blob/master/walk_through/walk_through.ipynb).


### 2. How to install FitSeq2.0?

* Python 3 is required. This version has been tested on a MacBook Pro (Apple M1 Chip, 8 GB Memeory), with Python 3.8.5.
* Clone this repository by running `git clone https://github.com/FangfeiLi05/FitSeq2.0.git` in terminal.
* `cd` to the root directory of the project (the folder containing `README.md`).
* Install dependencies by running `pip install -r requirements.txt` in terminal.


### 3. How to use FitSeq2.0?

#### 3.1. Evolution Simulation
`fitseqsimu_run_v1.py` simulates the entire experimental process of competative pooled growth of a population of genotypes with different fitnesses. This simulation includes all sources of noise, including growth noise, noise from cell transfers, DNA extraction, PCR, and sequencing.

##### Options
* `--t_seq` or `-t`: a .csv file, with
  + 1st column: sequenced time-points evaluated in number of generations
  + 2nd+ columns: average number of reads per barcode for each sequenced time-point (accept multiple columns for multiple sequencing runs)
* `--fitness` or `-s`: a .csv file, with
  + 1st column: fitness of each genotype
  + 2nd column: initial cell number of each genotype
* `--dna_copies` or `-d`: average genome template copies per barcode in PCR' (`default: 500`)
* `--pcr_cycles` or `-p`: number of cycles in PCR' (`default: 25`)
* `--output_filename` or `-o`: prefix of output files' (`default: output`)

##### Outputs
* `output_filename_EvoSimulation_Read_Number.csv`: read number per barcode for each time point
* `output_filename_EvoSimulation_Other_Info.csv`: a record of some inputs (also mean fitness of the population)
* `output_filename_EvoSimulation_Bottleneck_Cell_Number.csv`: bottleneck cell number per barcode for each time point
* `output_filename_EvoSimulation_Saturated_Cell_Number.csv`: saturated cell number per barcode for each time point

##### For Help
```
python fitseqsimu_run_v1.py --help
```

##### Example
```
python fitseqsimu_run_v1.py -t simu_input_time_points.csv -s simu_input_fitness.csv -o test
```    


#### 3.2. Fitness Estimation
`fitseq2_run_v1.py` estimates the fitness of each genotype from read-count time-series data. 

##### Options
* `--input` or `-i`: a .csv file, with each column being the read number per barcode at each sequenced time-point
* `--t_seq` or `-t`: a .csv file, with
  + 1st column: sequenced time-points evaluated in number of generations
  + 2nd column: total effective number of cells of the population for each sequenced time-point (cell number at the bottleneck)
* `--delta_t` or `-dt`: number of generations between bottlenecks (`default: the 2nd sequenced time-point - the 1st sequenced time-point`)
* `--c` or `-c`: half of variance introduced by cell growth and cell transfer' (`default: 1`)
* `--maximum_iteration_number` or `-n`: maximum number of iterations (`default: 50`)
* `--opt_algorithm` or `-a`: optmization algorithm (differential_evolution or nelder_mead) (`default: differential_evolution`)
* `--parallelize` or `-p`: whether to use multiprocess module to parallelize inference across lineages (`default: True`)
* `--output_filename` or `-o`: prefix of output files' (`default: output`)

##### Outputs
* `output_filename_FitSeq_Result.csv`: a .csv file, with
  + 1st column of .csv: estimated fitness of each genotype
  + 2nd column of .csv: theoretical estimation error for fitness
  + 3rd column of .csv: estimated initial cell number each genotype
  + 4th column of .csv: theoretical estimation error for initial cell number 
  + 5th column of .csv: maximized likelihood value (log-scale) for each genotype
  + 6th column of .csv: estimated mean fitness per sequenced time point
  + 7th column of .csv: inference time for each iteration
* `output_filename_Mean_fitness_Result.csv`: estimated mean fitness at each iteration
* `output_filename_Read_Number_Estimated.csv`: estimated read number per genotype for each time point

##### For Help
```
python fitseq2_run_v1.py --help
```  

##### Example
```
python fitseq2_run_v1.py -i simu_test_EvoSimulation_Read_Number.csv -t fitseq_input_time_points.csv -o test
```
