[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?style=flat&logo=Jupyter)](https://jupyterlab.readthedocs.io/en/stable)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![github](https://img.shields.io/badge/GitHub-FangfeiLi05-181717.svg?style=flat&logo=github)](https://github.com/my)


# Fit-Seq2 Tutorial

## 1. What is **_FitSeq2_**? :bulb:

**_FitSeq2_** is a Python-based fitness estimation tool for pooled amplicon sequencing studies. It is an improved version of the MATLAB tool [**_Fit-Seq_**](https://github.com/sashaflevy/Fit-Seq). FitSeq2 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

It currently has two functions:

   - GrowthSimulation.py` simulates the entire experimental process of competitve pooled growth of a population of genotypes.

   - `FitSeq2.py` calculates the fitness of each genotype from read-count time-series data.

A walkthrough is included in [this jupyter notebook](https://github.com/FangfeiLi05/FitSeq2/blob/master/test/Gothrough.ipynb).


## 2. How to install it? :computer:

Clone this repository to your local machine.

```console
git clone https://github.com/FangfeiLi05/FitSeq2.git
```

FitSeq2 is a Python pipeline. For convinence, we recommend install FitSeq2 pipeline with [Anaconda](https://www.anaconda.com). We create a Conda virtual environment named `fitseq2`, with Python also installed.

```console
conda create -n fitseq2 python=3
```

Navigate to the root directory (the folder containing `README.md`). Activate Conda environment `fitseq2`, then install all required Python packages by using the file [requirements.txt](./requirements.txt) in the root directory. 

```console
cd </path/to/FitSeq2/root/directory>
conda activate fitseq2
pip3 install -r requirements.txt
conda deactivate
```

In addition, we recommend add the Conda environment `fitseq2` to Jupyter kernelspec. By doing so, you can access this environment in Jupyter Notebook without activating the environment in Terminal. Specifically, in Terminal,
     
```console
cd ~
conda activate fitseq2
python3 -m ipykernel install --user --name=fitseq2
conda deactivate
```

Both of the two modules `GrowthSimulation` and `FitSeq2` can be run in Python (as a module) or in Terminal (as a Python script). To make the running of the two modules more easy, we add the source code directory [src](./src): (a) to PYTHONPATH, (b) to the system path. 
Add the following two lines to the file ".zprofile",
```console
export PATH="</path/to/FitSeq2/src>:$PATH"
export PYTHONPATH="$PYTHONPATH:</path/to/FitSeq2/src>"
```

By doing (a), we can import the modules `GrowthSimulation` and`FitSeq2` in Python, without using `sys.path.append(source_code_directory)`. By doing (b), we can run the Python scripts `GrowthSimulation.py` and `FitSeq2.py` in Terminal without specifying its full path. 



## 3. How to use it? :walking:

### 3.1. Growth Simulation
`GrowthSimulation.py` simulates the entire experimental process of competative pooled growth of a population of genotypes with different fitnesses. This simulation includes all sources of noise, including growth noise, noise from cell transfers, DNA extraction, PCR, and sequencing.

#### Options
* `--t_seq` or `-t`: a .csv file, with
  + 1st column: sequenced time-points evaluated in number of generations
  + 2nd+ columns: average number of reads per barcode for each sequenced time-point (accept multiple columns for multiple sequencing runs)
* `--fitness` or `-s`: a .csv file, with
  + 1st column: fitness of each genotype
  + 2nd column: initial cell number of each genotype
* `--dna_copies` or `-d`: average genome template copies per barcode in PCR' (`default: 500`)
* `--pcr_cycles` or `-p`: number of cycles in PCR' (`default: 25`)
* `--output_filename` or `-o`: prefix of output files' (`default: output`)

#### Outputs
* `output_filename_EvoSimulation_Read_Number.csv`: read number per barcode for each time point
* `output_filename_EvoSimulation_Other_Info.csv`: a record of some inputs (also mean fitness of the population)
* `output_filename_EvoSimulation_Bottleneck_Cell_Number.csv`: bottleneck cell number per barcode for each time point
* `output_filename_EvoSimulation_Saturated_Cell_Number.csv`: saturated cell number per barcode for each time point

#### For Help
```
python3 GrowthSimulation.py --help
```

##### Example
```
python3 GrowthSimulation.py -t simu_input_time_points.csv -s simu_input_fitness.csv -o test
```    


### 3.2. Fitness Estimation
`FitSeq2.py` estimates the fitness of each genotype from read-count time-series data. 

#### Options
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

#### Outputs
* `output_filename_FitSeq_Result.csv`: a .csv file, with
  + 1st column of .csv: estimated fitness of each genotype
  + 2rd column of .csv: estimated initial cell number each genotype
  + 3th column of .csv: maximized likelihood value (log-scale) for each genotype
  + 4th column of .csv: estimated mean fitness per sequenced time point
  + 5th column of .csv: value of noise parameter (kappa) per sequenced time point
  + 6th column of .csv: inference time for each iteration
* `output_filename_Mean_fitness_Result.csv`: estimated mean fitness at each iteration
* `output_filename_Read_Number_Estimated.csv`: estimated read number per genotype for each time point

#### For Help
```
python FitSeq2.py --help
```  

#### Example
```
python FitSeq2.py -i simu_test_EvoSimulation_Read_Number.csv -t fitseq_input_time_points.csv -o test
```