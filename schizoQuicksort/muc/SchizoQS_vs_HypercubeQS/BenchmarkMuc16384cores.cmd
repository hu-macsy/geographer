
#!/bin/bash
#@ job_type = parallel
#@ class = large
#@ node = 1024
#@ island_count = 2
#@ tasks_per_node = 16
#@ wall_clock_limit = 1:00:00
#@ job_name = BenchmarkMuc16384Cores
#@ network.MPI = sn_all,not_shared,us
#@ initialdir = $(home)/ShizoQuicksort/muc/SchizoQS_vs_HypercubeQS
#@ output = jobs/job_$(job_name)_$(jobid).out
#@ error = jobs/job_$(job_name)_$(jobid).err
#@ notification=always
#@ notify_user=michael.axtmann@kit.edu
#@ queue

module list

# config=config16384HC.json
# cat ${config}
# poe ~/dsort/Release/src/BenchmarkProbe ${config}

config=config16384S.json
cat ${config}
poe ~/dsort/Release/src/BenchmarkProbe ${config}
