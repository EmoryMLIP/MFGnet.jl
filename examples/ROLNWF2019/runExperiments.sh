#!/bin/sh


# OMT Examples
# 2 dimensions
julia -e "maxIter=500; d=2; nTrain=32^2; nVal=32^2; alph=[1.0;1.0;5.0;1.0;5.0]; include(\"runOMTExperiment.jl\")" > OMT-BFGS-d-2.txt

# 10 dimensions
julia -e "maxIter=500; d=2; nTrain=64^2; nVal=64^2; alph=[1.0;1.0;5.0;1.0;5.0]; include(\"runOMTExperiment.jl\")" > OMT-BFGS-d-10.txt

# 50 dimensions
julia -e "maxIter=500; d=2; nTrain=128^2; nVal=64^2; alph=[1.0;1.0;5.0;1.0;5.0]; include(\"runOMTExperiment.jl\")" > OMT-BFGS-d-50.txt

# 100 dimensions
julia -e "maxIter=500; d=2; nTrain=192^2; nVal=64^2; alph=[1.0;1.0;5.0;1.0;5.0]; include(\"runOMTExperiment.jl\")" > OMT-BFGS-d-100.txt





# Crowd Motion Examples
# 2 dimensions
julia -e "maxIter=500; saveIter=25; nt=4; nTh=2; sampleFreq=25; alph=[1.0;1.0;5.0;10.0;1.0]; nTrain=32^2; nVal=32^2; d=2; Qheight=5e1; saveStr = \"Obstacle-BFGSHighSamples-d-\"*string(d); include(\"runObstacleExperiment.jl\")" | tee Obstacle-BFGSHighSamples-d-2.txt

# 10 dimensions
julia -e "maxIter=500; saveIter=25; nt=4; nTh=2; sampleFreq=25; alph=[1.0;1.0;5.0;10.0;1.0]; nTrain=64^2; nVal=64^2; d=50; Qheight=5e1; saveStr = \"Obstacle-BFGSHighSamples-d-\"*string(d); include(\"runObstacleExperiment.jl\")" | tee Obstacle-BFGSHighSamples-d-10.txt

# 50 dimensions
julia -e "maxIter=500; saveIter=25; nt=4; nTh=2; sampleFreq=25; alph=[1.0;1.0;5.0;10.0;1.0]; nTrain=80^2; nVal=64^2; d=50; Qheight=5e1; saveStr = \"Obstacle-BFGSHighSamples-d-\"*string(d); include(\"runObstacleExperiment.jl\")" | tee Obstacle-BFGSHighSamples-d-50.txt

# 100 dimensions
julia -e "maxIter=500; saveIter=25; nt=4; nTh=2; sampleFreq=25; alph=[1.0;1.0;5.0;10.0;1.0]; nTrain=96^2; nVal=64^2; d=100; Qheight=5e1; saveStr = \"Obstacle-BFGSHighSamples-d-\"*string(d); include(\"runObstacleExperiment.jl\")" | tee Obstacle-BFGSHighSamples-d-100.txt