# MFGnet

[![CI](https://github.com/EmoryMLIP/MFGnet.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/EmoryMLIP/MFGnet.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/EmoryMLIP/MFGnet.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/EmoryMLIP/MFGnet.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Julia Version](https://img.shields.io/badge/Julia-1.10+-blue.svg)](https://julialang.org/)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

This repository contains the Julia code used in [*A Machine Learning Framework for Solving High-Dimensional Mean Field Game and Mean Field Control Problems*](https://doi.org/10.1073/pnas.1922204117).

## Paper Version

The original code used to produce the results in the PNAS 2020 paper can be found at:
- **Commit**: `da5712c0ae4d52bfcba11c844918233b6554c09f`
- **Date**: December 13, 2019
- **Access**: `git checkout da5712c`

The current version of this repository has been modernized for Julia 1.10+ with updated dependencies (DifferentialEquations.jl, Optimization.jl, and other modern Julia ecosystem packages) and improved functionality, while maintaining the core algorithms from the paper.

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

- Julia 1.10 or later (Julia 1.11+ recommended)
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
