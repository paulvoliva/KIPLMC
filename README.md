# KIPLMC
This is the code base for the reproduction of the experiments presented in "Kinetic Interacting Particle Langevin Monte Carlo" by Paul Valsecchi Oliva and O. Deniz Akyildiz. We present ULA-based diffusion discretisations for the approxiamtion of the marginal MLE.

The repository includes implementations of the KIPLMC1 and KIPLMC2 algorithms, as well as matching models to test. These include a toy case normal model, a Logistic Regression and a Bayesian Neural Network to reproduce Fig. 1-4 from the paper, as well as enabling much experimentation with a simple ''plug and play'' structure.

## Notebooks

There is a `jupyter` notebook included, clearly going through the procedure of obtaining the plots from Fig. 1 and Fig. 2. The data for the Wisconsin Cancer Dataset [[1]](#1) is attached as well, while we note that the dataset for the MNIST dataset was taken from the `keras.datasets` [[2]](#2). 

The core of the algorithms are in the `KIPLMC` package, which includes some implementations of similar algorithms for comparison. The algorithms come with some explanation of how to try them out with other models.

## Citation
The complete paper can be found at https://arxiv.org/abs/2407.05790, which can be cited with
```
@misc{oliva2024kineticinteractingparticlelangevin,
      title={Kinetic Interacting Particle Langevin Monte Carlo}, 
      author={Paul Felix Valsecchi Oliva and O. Deniz Akyildiz},
      year={2024},
      eprint={2407.05790},
      archivePrefix={arXiv},
      primaryClass={stat.CO},
      url={https://arxiv.org/abs/2407.05790}, 
}
```

## References 
<a id="1">[1]</a> W. Wolberg, O. Mangasarian, N. Street, and W. Street. "Breast Cancer Wisconsin (Diagnostic)," UCI Machine Learning Repository, 1993. [Online]. Available: https://doi.org/10.24432/C5DW2B.

<a id="2">[2]</a> Team, K. (n.d.). Keras documentation: MNIST digits classification dataset. [online] keras.io. Available at: https://keras.io/api/datasets/mnist/.
