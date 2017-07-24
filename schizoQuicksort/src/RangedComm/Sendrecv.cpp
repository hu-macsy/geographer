/*****************************************************************************
 * This file is part of the Project ShizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#include <mpi.h>
#include "MPI_Ranged.hpp"

int Range::Sendrecv(void *sendbuf,
            int sendcount, MPI_Datatype sendtype,
            int dest, int sendtag,
            void *recvbuf, int recvcount, MPI_Datatype recvtype,
            int source, int recvtag,
            Range::Comm const &comm, MPI_Status *status) {
    return MPI_Sendrecv(sendbuf,
                        sendcount, sendtype,
                        comm.first + dest, sendtag,
                        recvbuf, recvcount, recvtype,
                        comm.first + source, recvtag,
                        comm.mpi_comm, status);
}
