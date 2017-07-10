#!/bin/bash
#@ job_type = parallel
#@ class = large
#@ node = 2048
#@ island_count = 4
#@ tasks_per_node = 16
#@ wall_clock_limit = 0:08:00
#@ job_name = BenchmarkMuc32768Cores
#@ network.MPI = sn_all,not_shared,us
#@ initialdir = $(home)/ShizoQuicksort/muc/benchmark
#@ output = jobs/job_$(job_name)_$(jobid).out
#@ error = jobs/job_$(job_name)_$(jobid).err
#@ notification=always
#@ notify_user=uydof@student.kit.edu
#@ queue

poe ~/ShizoQuicksort/build/benchmark_elements
