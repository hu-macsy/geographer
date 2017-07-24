/*****************************************************************************
 * This file is part of the Project ShizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#include "../RangedComm/MPI_Ranged.hpp"
#include "RangeOperations.hpp"
#include <cassert>
#include <iostream>

int CollectiveOperations<Range::Comm, Range::Request>::Ibcast(void* buffer, 
        int count, MPI_Datatype datatype, int root, Range::Comm const &comm, 
        Range::Request* request, int tag) {
    Range::Ibcast(buffer, count, datatype, root, tag, comm, request);
    return 0;
}

int CollectiveOperations<Range::Comm, Range::Request>::Bcast(void* buffer, int count,
        MPI_Datatype datatype, int root, Range::Comm const &comm, int tag) {
    Range::Bcast(buffer, count, datatype, root, tag, comm);
    return 0;
}

bool CollectiveOperations<Range::Comm, Range::Request>::implementsIbcast() {
    return true;
}
bool CollectiveOperations<Range::Comm, Range::Request>::implementsIscanAndBcast() {
    return true;
}

int CollectiveOperations<Range::Comm, Range::Request>::Igatherv(void* sendbuf,
        int sendcount, MPI_Datatype sendtype, void* recvbuf, int* recvcounts,
        int* displs, MPI_Datatype recvtype, int root, Range::Comm comm, 
        Range::Request* request, int tag) {
    Range::Igatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs,
            recvtype, root, tag, comm, request);
    return 0;
}

int CollectiveOperations<Range::Comm, Range::Request>::Gatherv(void* sendbuf,
        int sendcount, MPI_Datatype sendtype, void* recvbuf, int* recvcounts,
        int* displs, MPI_Datatype recvtype, int root, Range::Comm comm, 
        int tag)  {
    Range::Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs,
            recvtype, root, tag, comm);
    return 0;
}

int CollectiveOperations<Range::Comm, Range::Request>::Igatherm(void *sendbuf, 
        int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount, int root, 
        std::function<void (void*, void*, void*)> op, Range::Comm comm,
        Range::Request* request,int tag) {
    Range::Igatherm(sendbuf, sendcount, sendtype, recvbuf, recvcount, root, tag,
            op, comm, request);
    return 0;
}

int CollectiveOperations<Range::Comm, Range::Request>::Gatherm(void* sendbuf,
        int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, 
        int root, std::function<void (void*,void*,void*)> op, Range::Comm comm,
        int tag) {
    Range::Gatherm(sendbuf, sendcount, sendtype, recvbuf, recvcount, root, tag,
            op, comm);
    return 0;
}

int CollectiveOperations<Range::Comm, Range::Request>::Ireduce(void* sendbuf,
        void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root,
        Range::Comm comm, Range::Request* request, int tag) {
    Range::Ireduce(sendbuf, recvbuf, count, datatype, tag, op, root, comm, request);
    return 0;
}

int CollectiveOperations<Range::Comm, Range::Request>::Reduce(void* sendbuf,
        void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root,
        Range::Comm comm, int tag) {
    Range::Reduce(sendbuf, recvbuf, count, datatype, tag, op, root, comm);
    return 0;
}

int CollectiveOperations<Range::Comm, Range::Request>::Iscan(void* sendbuf,
        void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, 
        const Range::Comm& comm, Range::Request* request, int tag) {
    Range::Iscan(sendbuf, recvbuf, count, datatype, tag, op, comm, request);
    return 0;
}

int CollectiveOperations<Range::Comm, Range::Request>::Scan(void* sendbuf, 
        void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
        const Range::Comm& comm, int tag) {
    Range::Scan(sendbuf, recvbuf, count, datatype, tag, op, comm);
    return 0;
}

int CollectiveOperations<Range::Comm, Range::Request>::IscanAndBcast(void* sendbuf,
        void* recvbuf_scan, void* recvbuf_bcast, int count, MPI_Datatype datatype,
        MPI_Op op, Range::Comm const &comm, Range::Request *request, int tag) {
    Range::IscanAndBcast(sendbuf, recvbuf_scan, recvbuf_bcast, count, datatype,
            tag, op, comm, request);
    return 0;
}

int CollectiveOperations<Range::Comm, Range::Request>::ScanAndBcast(void *sendbuf,
        void *recvbuf_scan, void *recvbuf_bcast, int count, MPI_Datatype datatype, 
        MPI_Op op, Range::Comm const &comm, int tag) {
    Range::ScanAndBcast(sendbuf, recvbuf_scan, recvbuf_bcast, count, datatype,
            tag, op, comm);
    return 0;
}

int CollectiveOperations<Range::Comm, Range::Request>::Isend(void *sendbuf,
        int count, MPI_Datatype datatype, int dest, int tag, Range::Comm const &comm,
        Range::Request *request) {
    Range::Isend(sendbuf, count, datatype, dest, tag, comm, request);
    return 0;
}

int CollectiveOperations<Range::Comm, Range::Request>::Send(void *sendbuf,
        int count, MPI_Datatype datatype, int dest, int tag, Range::Comm const &comm) {
    return Range::Send(sendbuf, count, datatype, dest, tag, comm);
}

int CollectiveOperations<Range::Comm, Range::Request>::Irecv(void *buffer, 
        int count, MPI_Datatype datatype, int source, int tag, Range::Comm const &comm,
        Range::Request *request) {
    Range::Irecv(buffer, count, datatype, source, tag, comm, request);
    return 0;
}

int CollectiveOperations<Range::Comm, Range::Request>::Recv(void *buf, int count,
        MPI_Datatype datatype, int source, int tag, Range::Comm const &comm, 
        MPI_Status *status) {
    return Range::Recv(buf, count, datatype, source, tag, comm, status);
}

int CollectiveOperations<Range::Comm, Range::Request>::Sendrecv(void *sendbuf,
        int sendcount, MPI_Datatype sendtype,
        int dest, int sendtag,
        void *recvbuf, int recvcount, MPI_Datatype recvtype,
        int source, int recvtag,
        Range::Comm const &comm, MPI_Status *status) {
    return Range::Sendrecv(sendbuf, sendcount, sendtype,
            dest, sendtag,
            recvbuf, recvcount, recvtype,
            source, recvtag,
            comm, status);
}

int CollectiveOperations<Range::Comm, Range::Request>::Iprobe(int source, int tag,
        Range::Comm const &comm, int *flag, MPI_Status *status) {
    return Range::Iprobe(source, tag, comm, flag, status);
}

int CollectiveOperations<Range::Comm, Range::Request>::Probe(int source, int tag,
        Range::Comm const &comm, MPI_Status *status) {
    return Range::Probe(source, tag, comm, status);
}

int CollectiveOperations<Range::Comm, Range::Request>::Ibarrier(Range::Comm comm, 
        Range::Request *request) {
    Range::Ibarrier(comm, request);
    return 0;
}

int CollectiveOperations<Range::Comm, Range::Request>::Barrier(Range::Comm comm) {
    Range::Barrier(comm);
    return 0;
}

int CollectiveOperations<Range::Comm, Range::Request>::Test(Range::Request *request,
        int *flag, MPI_Status *status) {
    return Range::Test(request, flag, status);
}

int CollectiveOperations<Range::Comm, Range::Request>::Wait(Range::Request *request,
        MPI_Status *status) {
    return Range::Wait(request, status);
}

int CollectiveOperations<Range::Comm, Range::Request>::Testall(int count, 
        Range::Request array_of_requests[], int *flag, MPI_Status array_of_statuses[]) {
    Range::Testall(count, array_of_requests, flag, array_of_statuses);
    return 0;
}

int CollectiveOperations<Range::Comm, Range::Request>::Waitall(int count, 
        Range::Request array_of_requests[], MPI_Status array_of_statuses[]) {
    Range::Waitall(count, array_of_requests, array_of_statuses);
    return 0;
}

int CollectiveOperations<Range::Comm, Range::Request>::Comm_rank(
        Range::Comm const &comm, int *rank) {
    Range::Comm_rank(comm, rank);
    return 0;
}

int CollectiveOperations<Range::Comm, Range::Request>::Comm_size(
        Range::Comm const &comm, int *size) {
    Range::Comm_size(comm, size);
    return 0;
}

void CollectiveOperations<Range::Comm, Range::Request>::Create_Comm_from_MPI(
        MPI_Comm mpi_comm, Range::Comm* comm) {
    Range::Create_Comm_from_MPI(mpi_comm, comm);
}

void CollectiveOperations<Range::Comm, Range::Request>::Create_Comm(
        Range::Comm const &comm, int first, int last, Range::Comm *new_comm) {    
    Range::Create_Comm(comm, first, last, new_comm);
}
void CollectiveOperations<Range::Comm, Range::Request>::createNewCommunicators(
        Range::Comm const &comm, int left_end, int right_start, Range::Comm *left,
        Range::Comm *right, bool mpi_split) {
    if (mpi_split) {
     /*   MPI_Comm mpi_comm = comm.mpi_comm;
        int rank, size;
        MPI_Comm_rank(mpi_comm, &rank);
        MPI_Comm_size(mpi_comm, &size);
        
        int color1, color2;
        if (rank <= left_end)
            color1 = 1;
        else {
            color1 = MPI_UNDEFINED;
        }

        if (rank >= right_start)
            color2 = 2;
        else {
            color2 = MPI_UNDEFINED;
        }

        MPI_Comm mpi_left, mpi_right;        
        MPI_Comm_split(comm.mpi_comm, color1, rank, &mpi_left);
        MPI_Comm_split(comm.mpi_comm, color2, rank, &mpi_right);
        *left = Range::Comm(mpi_left, 0, left_end);
        *right = Range::Comm(mpi_right, 0, size - 1 - right_start); */    
    } else {
        int size;
        Range::Comm_size(comm, &size);
        Range::Create_Comm(comm, 0, left_end, left);
        Range::Create_Comm(comm, right_start, size - 1, right);
    }
}

int CollectiveOperations<Range::Comm, Range::Request>::getSource(
        Range::Comm const &comm, MPI_Status status) {
    return Range::get_Rank_from_Status(comm, status);
}

int CollectiveOperations<Range::Comm, Range::Request>::freeComm(Range::Comm *) {
    /*if (comm->mpi_comm != MPI_COMM_WORLD && comm->mpi_comm != MPI_COMM_NULL)
	return MPI_Comm_free(&(comm->mpi_comm));*/
    return 0;
}

