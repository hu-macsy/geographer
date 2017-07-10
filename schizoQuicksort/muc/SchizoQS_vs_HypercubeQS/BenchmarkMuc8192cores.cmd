#!/bin/bash
#@ job_type = parallel
#@ class = general
#@ node = 512
#@ island_count = 1
#@ tasks_per_node = 16
#@ wall_clock_limit = 1:00:00
#@ job_name = BenchmarkMuc8192Cores
#@ network.MPI = sn_all,not_shared,us
#@ initialdir = $(home)/ShizoQuicksort/muc/SchizoQS_vs_HypercubeQS
#@ output = jobs/job_$(job_name)_$(jobid).out
#@ error = jobs/job_$(job_name)_$(jobid).err
#@ notification=always
#@ notify_user=michael.axtmann@kit.edu
#@ queue

module list

# config=config08192HC.json
# cat ${config}
# poe ~/dsort/Release/src/BenchmarkProbe ${config}

config=config08192S.json
cat ${config}
poe ~/dsort/Release/src/BenchmarkProbe ${config}
