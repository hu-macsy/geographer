/*****************************************************************************
 * This file is part of the Project ShizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#include <mpi.h>
#include "MPI_Ranged.hpp"
#include "Requests.hpp"

int Range::Send(const void *sendbuf, int count, MPI_Datatype datatype, int dest,
        int tag, Range::Comm comm) {
    return MPI_Send(const_cast <void*>(sendbuf), count, datatype,
                    comm.first + dest, tag, comm.mpi_comm);
};

/*
 * Request for the send
 */
class Range_Requests::Isend : public Range::R_Req {
public:
    Isend(const void *sendbuf, int count, MPI_Datatype datatype,
            int dest, int tag, Range::Comm comm);
    int test(int *flag, MPI_Status *status);

private:
    const void *sendbuf;
    int count;
    MPI_Datatype datatype;
    int dest, tag;
    Range::Comm comm;
    bool requested;
    MPI_Request request;
};

void Range::Isend(const void *sendbuf, int count, MPI_Datatype datatype, int dest,
        int tag, Range::Comm comm, Range::Request *request) {
    *request  = std::unique_ptr<R_Req>(new Range_Requests::Isend(sendbuf, count, 
            datatype, dest + comm.first, tag, comm));
};

Range_Requests::Isend::Isend(const void *sendbuf, int count, MPI_Datatype datatype,
        int dest, int tag, Range::Comm comm) : sendbuf(sendbuf), count(count),
        datatype(datatype), dest(dest), tag(tag), comm(comm), requested(false) {
    void* buf = const_cast<void*>(sendbuf);
    MPI_Isend(buf, count, datatype, dest, tag, comm.mpi_comm, &request);
};

int Range_Requests::Isend::test(int *flag, MPI_Status *status) {
    return MPI_Test(&request, flag, status);
};
