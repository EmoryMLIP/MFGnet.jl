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
    using Flux
	using Zygote
    using Printf

    include("F.jl")
    include("G.jl")
    include("layers.jl")
    include("singleLayer.jl")
    include("ResNN.jl")
    include("NN.jl")
    include("odefun.jl")
    include("timeStepping.jl")
    include("linInter1D.jl")
    include("utils.jl")
    include("MFG.jl")
    include("param2vec.jl")
    include("Gaussians.jl")
    include("bfgs.jl")


end  # module MFGnet
