# Iterative Bayesian DNN 


Particle level information get distorted in the detector level due to detector inefficiencies and detector acceptance. Unfolding try to make corrections to these effects while improving the particle level information.

Here we try to use a deep neural network with iterative Bayesian reweighing for unfolding problem.

This is based on the [DeepBayes](https://github.com/ahill187/DeepBayes) module.

Use following commands to use repo.

```
git clone https://github.com/dinupa1/unfoldML.git
cd unfoldML
conda env create -f environment.yml
conda activate unfoldml
```

Follow documentation [here](https://github.com/conda-forge/miniforge) to install `miniforge`

For more information [here](slides/05_feb_2023.pdf).
