/*****************************************************************************
 * This file is part of the Project ShizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#include <cassert>
#include <memory>
#include <mpi.h>
#include "MPI_Ranged.hpp"
#include "Requests.hpp"
#include <iostream>

#define W(X) #X << "=" << X << " "

 

Range::Comm::Comm() : mpi_comm(MPI_COMM_NULL), first(-1), last(-1){
}

Range::Comm::Comm(MPI_Comm mpi_comm, int first, int last) : mpi_comm(mpi_comm),
        first(first), last(last) {
}

bool Range::Comm::includesRank(int rank) {
    return ((rank >= first) && (rank <= last));
}

Range::Request::Request() : req_ptr(std::unique_ptr<R_Req>()) {
}

Range::Request::Request(R_Req *req) : req_ptr(req) {
}

Range::R_Req& Range::Request::operator*() {
    return *req_ptr;
}

Range::R_Req* Range::Request::operator->() {
    return req_ptr.get();
}

void Range::Request::operator=(std::unique_ptr<R_Req> req) {   
    req_ptr = std::move(req);
}

void Range::Comm_rank(Range::Comm comm, int *rank) {
    int global_rank;
    MPI_Comm_rank(comm.mpi_comm, &global_rank);
    *rank = global_rank - comm.first;
}

void Range::Comm_size(Range::Comm comm, int *size) {
    *size = comm.last - comm.first + 1;
}

void Range::Create_Comm_from_MPI(MPI_Comm mpi_comm, Range::Comm *rcomm) {
    int size;
    MPI_Comm_size(mpi_comm, &size);
    *rcomm = Range::Comm(mpi_comm, 0, size - 1);
}
    
void Range::Create_Comm(Range::Comm rcomm, int first, int last,
        Range::Comm *new_comm) {
    *new_comm = Range::Comm(rcomm.mpi_comm, rcomm.first + first, rcomm.first + last);
}

int Range::Iprobe(int source, int tag, Range::Comm comm, int *flag, MPI_Status *status) {
    if (source != MPI_ANY_SOURCE)
        source += comm.first;
    MPI_Status tmp_status;
    int return_value = MPI_Iprobe(source, tag, comm.mpi_comm, flag, &tmp_status);
    if (*flag) {
        if (!comm.includesRank(tmp_status.MPI_SOURCE))
            *flag = 0;
        else if (status != MPI_STATUS_IGNORE)
            *status = tmp_status;
    }
    return return_value;
}

int Range::Probe(int source, int tag, Range::Comm comm, MPI_Status *status) {
    int flag = 0;
    while (!flag)
        Range::Iprobe(source, tag, comm, &flag, status);
    return 0;
}

int Range::Test(Range::Request *request, int *flag, MPI_Status *status) {
    *flag = 0;
    return (*request)->test(flag, status);
}

void Range::Testall(int count, Range::Request *array_of_requests, int* flag,
        MPI_Status array_of_statuses[]) {
    *flag = 1;
    for (int i = 0; i < count; i++) {
        int temp_flag;
        if (array_of_statuses == MPI_STATUSES_IGNORE)
            Test(&array_of_requests[i], &temp_flag, MPI_STATUS_IGNORE);
        else
            Test(&array_of_requests[i], &temp_flag, &array_of_statuses[i]);
        if (temp_flag == 0)
            *flag = 0;
    }
}

int Range::Wait(Range::Request *request, MPI_Status *status) {
    int flag = 0, return_value;
    while (flag == 0) {
        return_value = Test(request, &flag, status);
    }
    return return_value;
}

void Range::Waitall(int count, Range::Request array_of_requests[],
        MPI_Status array_of_statuses[]) {
    int flag = 0;
    while (flag == 0) {
        Testall(count, array_of_requests, &flag, array_of_statuses);
    }
}

int Range::get_Rank_from_Status(Range::Comm const &comm, MPI_Status status) {
    return status.MPI_SOURCE - comm.first;
}
