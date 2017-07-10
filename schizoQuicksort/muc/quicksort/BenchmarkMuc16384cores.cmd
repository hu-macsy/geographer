#!/bin/bash
#@ job_type = parallel
#@ class = large
#@ node = 1024
#@ island_count = 2
#@ tasks_per_node = 16
#@ wall_clock_limit = 0:06:00
#@ job_name = BenchmarkMuc16384Cores
#@ network.MPI = sn_all,not_shared,us
#@ initialdir = $(home)/ShizoQuicksort/muc/quicksort
#@ output = jobs/job_$(job_name)_$(jobid).out
#@ error = jobs/job_$(job_name)_$(jobid).err
#@ notification=always
#@ notify_user=uydof@student.kit.edu
#@ queue

. ~/.bashrc
module list
poe ~/ShizoQuicksort/build/quicksort
