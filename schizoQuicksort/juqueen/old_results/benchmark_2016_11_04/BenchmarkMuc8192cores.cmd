name=bcastbenchmrk
#@job_name         = MPI_code
#@output           = result_${name}_$(jobid)_$(stepid).out
#@error            = result_${name}_$(jobid)_$(stepid).err
#@environment      = COPY_ALL
#@job_type         = bluegene
#@notification     = always
#@notify_user      = michael.axtmann@kit.edu
#@bg_size         = 512
#@bg_connectivity  = torus
#@wall_clock_limit = 0:30:00
#@queue

runjob --ranks-per-node 16 : ~/ShizoQuicksort/build/benchmark
