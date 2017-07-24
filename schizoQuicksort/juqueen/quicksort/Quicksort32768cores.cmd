name=bcastbenchmrk
#@job_name         = MPI_code
#@output           = result_quicksort_32768cores_$(jobid)_$(stepid).out
#@error            = result_quicksort_32768cores_$(jobid)_$(stepid).err
#@environment      = COPY_ALL
#@job_type         = bluegene
#@notification     = always
#@notify_user      = michael.axtmann@kit.edu
#@bg_shape         = 1x1x2x2
#@bg_connectivity  = torus
#@wall_clock_limit = 0:15:00
#@queue

module list

runjob --ranks-per-node 16 : ~/ShizoQuicksort/build/quicksort
