"""

MFGnet.jl -  A Machine Learning Framework for Solving High-Dimensional Mean Field Game and Mean Field Control Problems

A detailed description of the approach implemented here can be found in:

@article{ROLNWF2019,
  title = {A Machine Learning Framework for Solving High-Dimensional Mean Field Game and Mean Field Control Problems},
  year = {2019},
  journal = {arXiv preprint arXiv:1912.01825},
  author = {L. Ruthotto, S. Osher, W. Li, L. Nurbekyan, S. Wu Fung},
  pages = {15 pages}
}
"""
module MFGnet
    using LinearAlgebra
	using Zygote
    using Printf

    include("F.jl")
    include("G.jl")
    include("utils.jl")        # Must come before ResNN (ResNN uses append)
    include("linInter1D.jl")   # Must come before ResNN (ResNN uses linInter1D)
    include("singleLayer.jl")  # Must come before ResNN and NN
    include("ResNN.jl")        # Must come before NN (NN uses Union{SingleLayer,ResNN})
    include("NN.jl")           # Must come before layers.jl
    include("layers.jl")       # Uses NN() in PotentialNN constructor
    include("timeStepping.jl") # Must come before odefun and MFG
    include("odefun.jl")       # Must come before MFG
    include("MFG.jl")          # Uses odefun, timeStepping, layers
    include("param2vec.jl")
    include("Gaussians.jl")
    include("bfgs.jl")


end  # module MFGnet
