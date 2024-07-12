# Minimum-distance based empirical Bayes estimation for the Poisson mean model

Implementation of the method proposed in the paper: "[Optimal empirical Bayes estimation for the Poisson model via minimum-distance methods](https://arxiv.org/abs/2209.01328)", Soham Jana, Yury Polyanskiy, and Yihong Wu, arXiv:2209.01328v2.

## Requirements
* numpy/scipy
* sklearn
* mathplotlib
* tqdm
* pandas
* scipy
* random


## Info
1) The "plotting_worst_prior" folder includes the file 

* main_plot_poisson_worst_prior.py

which plots (not compute) the worst case prior that maximizes the posterior variance over a given support
and saves the plots in the folder "numerical_results"

2) The "main_mixture_codes" folder includes the main files

* main_poisson_mindist.py: This is used to estimate the prior distribution from training data

* main_poisson_eb.py: This code computes NPMLE, Chi squared, and Squared Hellinger based EB estimator function, given training dataset
  
* main_poisson_worst_prior.py: This is used to estimate the worst-case prior that maximizes the posterior variance


## Cite
Please cite our paper if you use this code in your own work.
```
@article{jana2022optimal,
  title={Optimal empirical Bayes estimation for the Poisson model via minimum-distance methods},
  author={Jana, Soham and Polyanskiy, Yury and Wu, Yihong},
  journal={arXiv preprint arXiv:2209.01328},
  year={2022}
}
```
