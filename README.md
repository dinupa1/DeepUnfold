# Iterative Bayesian Unfolding with DNN 


Particle level information get distorted in the detector level due to detector inefficiencies and detector acceptance. Unfolding try to make corrections to these effects while improving the particle level information.

Here we try to use a deep neural network with iterative Bayesian reweighing for unfolding problem.

This is based on the [OmniFold](https://github.com/ahill187/DeepBayes) algorithm.

For cloning the repository use;

```
https://github.com/dinupa1/DeepUnfold.git
cd DeepUnfold
conda env create -f environment.yml
conda activate DeepUnfold
```

Follow documentation [here](https://github.com/conda-forge/miniforge) to install `miniforge`

## Resources
[OmniFold: A Method to Simultaneously Unfold All Observables](https://arxiv.org/abs/1911.09107)

[A Neural Resampler for Monte Carlo Reweighting with Preserved Uncertainties](https://arxiv.org/abs/2007.11586)
