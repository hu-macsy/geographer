#!/bin/bash
#@ job_type = parallel
#@ class = test
#@ node = 16
#@ island_count = 1
#@ tasks_per_node = 16
#@ wall_clock_limit = 0:02:00
#@ job_name = BenchmarkMuc256Cores
#@ network.MPI = sn_all,not_shared,us
#@ initialdir = $(home)/ShizoQuicksort/muc/quicksort
#@ output = job_$(job_name)_$(jobid).out
#@ error = job_$(job_name)_$(jobid).err
#@ notification=always
#@ notify_user=uydof@student.kit.edu
#@ queue

poe ~/ShizoQuicksort/build/quicksort
