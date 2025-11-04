# MFGnet

[![CI](https://github.com/EmoryMLIP/MFGnet.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/EmoryMLIP/MFGnet.jl/actions/workflows/CI.yml)

This repository contains the Julia code used in [*A Machine Learning Framework for Solving High-Dimensional Mean Field Game and Mean Field Control Problems*](https://doi.org/10.1073/pnas.1922204117).

## Installation

This package can be installed using Julia's package manager. To do this, type:

```julia
using Pkg
Pkg.add(PackageSpec(url="https://github.com/EmoryMLIP/MFGnet.jl/"))
```

## Testing

To run the test suite locally:

```julia
using Pkg
Pkg.test("MFGnet")
```

Or use the provided test script:

```bash
julia test_local.jl
```

## Requirements

- Julia 1.6 or later (Julia 1.11+ recommended)
- See `Project.toml` for package dependencies

## Reference

A detailed description of the approach implemented here can be found in:

    @article{ROLNWF2020,
      title = {A Machine Learning Framework for Solving High-Dimensional Mean Field Game and Mean Field Control Problems},
      year = {2020},
      journal = {Proceedings of the National Academy of Sciences},
      author = {L. Ruthotto, S. Osher, W. Li, L. Nurbekyan, S. Wu Fung},
      issue = {117},
      volume = {17},
      url = {https://doi.org/10.1073/pnas.1922204117},
      pages = {9783--9793}
    }


## Acknowledgements

This material is in part based upon work supported by the National Science Foundation under Grant Number 1751636. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.
