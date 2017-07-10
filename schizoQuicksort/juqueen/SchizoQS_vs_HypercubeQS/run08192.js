#@job_name         = MPI_code
#@comment          = "16 ranks per node, 64 nodes"
#@output           = result_probe_$(jobid)_$(stepid).out
#@error            = result_probe_$(jobid)_$(stepid).err
#@environment      = COPY_ALL
#@job_type         = bluegene
#@notification     = always
#@notify_user      = michael.axtmann@kit.edu
#@bg_size          = 2048
#@bg_connectivity  = torus
#@wall_clock_limit = 1:00:00
#@queue

module list

config=config08192HC.json
cat ${config}
runjob --ranks-per-node 16 : ~/dsort/Release/src/BenchmarkProbe ${config}

config=config08192S.json
cat ${config}
runjob --ranks-per-node 16 : ~/dsort/Release/src/BenchmarkProbe ${config}
