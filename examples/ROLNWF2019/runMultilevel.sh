#!/bin/sh

# OMT Examples
# 2 dimensions
julia -e "maxIter=[500;500]; d=2; nTrain=[32^2;48^2]; nVal=32^2; nt=2; saveIter=100; alph=[2.0;1.0;5.0;3.0;3.0]; include(\"runOMTExperimentMultilevel.jl\")" | tee OMT-Multilevel-d-2.txt

# no HJB
julia -e "maxIter=[500;500]; d=2; nTrain=[32^2;48^2]; nVal=32^2; nt=2; saveIter=100; alph=[2.0;1.0;5.0;0.0;0.0]; saveStr=\"OMT-Multilevel-noHJB-d-2\"; include(\"runOMTExperimentMultilevel.jl\")" | tee OMT-Multilevel-noHJB-nt-2-d-2.txt

# no HJB nt=2 time steps
julia -e "maxIter=[500;500]; d=2; nTrain=[32^2;48^2]; nVal=32^2; nt=8; saveIter=100; alph=[2.0;1.0;5.0;0.0;0.0]; saveStr=\"OMT-Multilevel-noHJB-nt-8-d-2\"; include(\"runOMTExperimentMultilevel.jl\")" | tee OMT-Multilevel-noHJB-nt-8-d-2.txt

# ADAM
julia -e "maxIter=[5000;5000]; d=2; nTrain=[32^2;48^2]; nVal=32^2; nt=2; saveIter=100; alph=[2.0;1.0;5.0;3.0;3.0]; saveStr=\"OMT-Multilevel-ADAM-d-2\"; include(\"runOMTExperimentMultilevel.jl\")" | tee OMT-Multilevel-ADAM-d-2.txt

# 10 dimensions
julia -e "maxIter=[500;500]; d=10; nTrain=[64^2;80^2]; nVal=64^2; nt=2; saveIter=100; alph=[2.0;1.0;5.0;3.0;3.0]; include(\"runOMTExperimentMultilevel.jl\")" | tee OMT-Multilevel-d-10.txt
# 50 dimensions
julia -e "maxIter=[500;500]; d=50; nTrain=[112^2;128^2]; nVal=64^2; nt=2; saveIter=100; alph=[2.0;1.0;5.0;3.0;3.0]; include(\"runOMTExperimentMultilevel.jl\")" | tee OMT-Multilevel-d-50.txt
# 100 dimensions
julia -e "maxIter=[500;500]; d=100; nTrain=[160^2;192^2]; nVal=64^2; nt=2; saveIter=100; alph=[2.0;1.0;5.0;3.0;3.0]; include(\"runOMTExperimentMultilevel.jl\")" | tee OMT-Multilevel-d-100.txt

# # Crowd Motion Examples
# # 2 dimensions
julia -e "maxIter=[1000;250]; nt=4; alph=[1.0;1.0;5.0;20.0;1.0]; nTrain=[32^2;48^2]; nVal=32^2; d=2; Qheight=5e1; saveStr = \"Obstacle-Multilevel-d-\"*string(d); include(\"runObstacleMultilevel.jl\")" | tee Obstacle-Multilevel-d-2.txt
#
# # 10 dimensions
julia -e "maxIter=[1000;250]; nt=4;   alph=[1.0;1.0;5.0;20.0;1.0]; nTrain=[64^2;80^2]; nVal=64^2; d=50; Qheight=5e1; saveStr = \"Obstacle-Multilevel-d-\"*string(d); include(\"runObstacleMultilevel.jl\")" | tee Obstacle-Multilevel-d-10.txt
#
# # 50 dimensions
julia -e "maxIter=[1000;250];  nt=4; sampleFreq=25; alph=[1.0;1.0;5.0;20.0;1.0]; nTrain=[80^2;96^2]; nVal=64^2; d=50; Qheight=5e1; saveStr = \"Obstacle-Multilevel-d-\"*string(d); include(\"runObstacleMultilevel.jl\")" | tee Obstacle-Multilevel-d-50.txt
#
# # 100 dimensions
julia -e "maxIter=[1000;250];  nt=4; sampleFreq=25; alph=[1.0;1.0;5.0;20.0;1.0]; nTrain=[92^2;112^2]; nVal=64^2; d=100; Qheight=5e1; saveStr = \"Obstacle-Multilevel-d-\"*string(d); include(\"runObstacleMultilevel.jl\")" | tee Obstacle-Multilevel-d-100.txt
