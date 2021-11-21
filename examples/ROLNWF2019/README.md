# Examples in A Machine Learning Framework for Solving High-Dimensional MFC and MFG

This folder contains the scripts and codes used to perform the experiments described in the paper

```
  @article{ROLNWF2019,
    title = {A Machine Learning Framework for Solving High-Dimensional Mean Field Game and Mean Field Control Problems},
    year = {2019},
    journal = {arXiv preprint arXiv:1912.01825},
    author = {L. Ruthotto, S. Osher, W. Li, L. Nurbekyan, S. Wu Fung},
    pages = {15 pages}
  }
```

## Julia experiments

The results were generated with version 0.1.0 of the MFGnet.jl package. The package dependencies at that time are captured in the Manifest.toml file. We recommend to revert to this version when you are interested in reproducing the results from the paper (note that some randomness in the optimization algorithms may lead to slightly different results).

To run all or parts of the experiments, take a look at the shell script 'runMultilevel.sh'. The overall computation can take some time, depending on your system.  The results we have obtained with this are provided [here](http://www.mathcs.emory.edu/~lruthot/pubs/2020-PNAS-MFG/ROLNWF2019-MFGnet.jl-Results.zip).

From these results, figures, grid representation of the value function, etc. are generated in the files viewOMTResults.jl and viewObstacleResults.jl.

## MATLAB experiments

We performed the comparison between the proposed Lagrangian Machine Learning scheme and a provably convergent Eulerian method in MATLAB. The code needed for this can be found in the subdirectory 'matlab'. These codes depend on the package [FAIR.m](https://github.com/C4IR/FAIR.m/). We have used version df184cdda6dd15cc947b726e2d0cb42aa10fd8b5 with MATLAB R2019b. These codes are not optimized for runtime and we provide our results for download [here](https://www.mathcs.emory.edu/~lruthot/publication/ruthotto-et-al-2020-mfg/).
