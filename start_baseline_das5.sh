#!/usr/bin/env bash

slurmcfg \
    --head_node fs2.das5.science.uva.nl --remote_user agrotov \
    --remote_scratch_directory /var/scratch/agrotov \
    --resource TitanX \
    --num_nodes 1 \
    --inline_job_descriptor \
    "task {
        working_directory: '/Users/agrotov/Documents/PhD/lerotexperimental'
        command: 'module load blas; module load cuda70/toolkit; th Main.lua -network GoogLeNet_BN -batchSize 64 -nGPU 2 -save GoogLeNet_BN -bufferSize 9600 -LR 0.01 -checkpoint 320000 -weightDecay 1e-4'
    }"

