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

int Range::Recv(void* buf, int count, MPI_Datatype datatype, int source, int tag,
        Range::Comm comm, MPI_Status *status) {
    return MPI_Recv(buf, count, datatype, comm.first + source, tag, comm.mpi_comm, status);
}

/*
 * Request for the receive
 */
class Range_Requests::Irecv : public Range::R_Req {
public:
    Irecv(void *buffer, int count, MPI_Datatype datatype, int source,
            int tag, Range::Comm comm);
    int test(int *flag, MPI_Status *status);
private:
    void *buffer;
    int count;
    MPI_Datatype datatype;
    int source, tag;
    Range::Comm comm;
    bool receiving;
    MPI_Request request;
};

void Range::Irecv(void* buffer, int count, MPI_Datatype datatype, int source, int tag,
        Range::Comm comm, Range::Request *request) {    
    *request = std::unique_ptr<R_Req>(new Range_Requests::Irecv(buffer, count, 
            datatype, source, tag, comm));
    int x;
    (*request)->test(&x, MPI_STATUS_IGNORE);
};

Range_Requests::Irecv::Irecv(void *buffer, int count, MPI_Datatype datatype,
        int source, int tag, Range::Comm comm) : buffer(buffer), count(count),
        datatype(datatype), source(source), tag(tag), comm(comm), receiving(false) {  
};

int Range_Requests::Irecv::test(int *flag, MPI_Status *status) {    
    if (receiving) {
        return MPI_Test(&request, flag, status);
    }
    if (!receiving) {
        int ready;
        MPI_Status stat;
        Range::Iprobe(source, tag, comm, &ready, &stat);
        if (ready) {
            MPI_Irecv(buffer, count, datatype, stat.MPI_SOURCE, tag, comm.mpi_comm,
                    &request);
            receiving = true;
        }
    }
    return 0;
};
