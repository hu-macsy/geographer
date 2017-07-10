name=bcastbenchmrk
#@job_name         = MPI_code
#@output           = result_quicksort_$(jobid)_$(stepid).out
#@error            = result_quicksort_$(jobid)_$(stepid).err
#@environment      = COPY_ALL
#@job_type         = bluegene
#@notification     = always
#@notify_user      = michael.axtmann@kit.edu
#@bg_size         = 32
#@bg_connectivity  = torus
#@wall_clock_limit = 0:03:00
#@queue

module list

runjob --ranks-per-node 16 : ~/ShizoQuicksort/build/quicksort
