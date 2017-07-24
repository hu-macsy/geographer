/*****************************************************************************
 * This file is part of the Project ShizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#include <mpi.h>
#include "MPIOperations.hpp"
#include <vector>
#include <cassert>

int CollectiveOperations<MPI_Comm, MPI_Request>::Ibcast(void* buffer, int count,
        MPI_Datatype datatype, int root, MPI_Comm const &comm, MPI_Request *request,
        int tag) {
#ifndef NO_IBCAST
    return MPI_Ibcast(buffer, count, datatype, root, comm, request);
#else
    (void) buffer;
    (void) count;
    (void) datatype;
    (void) root;
    (void) comm;
    (void) request;
    (void) tag;
    assert(false);
    return -1;
#endif
}

int CollectiveOperations<MPI_Comm, MPI_Request>::Bcast(void* buffer, int count, 
        MPI_Datatype datatype, int root, MPI_Comm const &comm, int tag) {
    (void) tag;
    return MPI_Bcast(buffer, count, datatype, root, comm);
}

bool CollectiveOperations<MPI_Comm, MPI_Request>::implementsIscanAndBcast() {
    return false;
}

bool CollectiveOperations<MPI_Comm, MPI_Request>::implementsIbcast() {
#ifndef NO_IBCAST
    return true;
#else
    return false;
#endif
}

int CollectiveOperations<MPI_Comm, MPI_Request>::Igather(void* sendbuf, 
        int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
        MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request* request,
        int tag) {
#ifndef NO_IBCAST
    return MPI_Igather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
            root, comm, request);
#else
    (void) sendbuf;
    (void) sendcount;
    (void) sendtype;
    (void) recvbuf;
    (void) recvcount;
    (void) recvtype;
    (void) root;
    (void) comm;
    (void) request;
    (void) tag;
    assert(false);
    return -1;
#endif
}

int CollectiveOperations<MPI_Comm, MPI_Request>::Gather(void* sendbuf, 
        int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
        MPI_Datatype recvtype, int root, MPI_Comm comm, int tag) {
    (void) tag;
    return MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
            root, comm);
}

int CollectiveOperations<MPI_Comm, MPI_Request>::Igatherv(void* sendbuf, 
        int sendcount, MPI_Datatype sendtype, void* recvbuf, int* recvcounts,
        int* displs, MPI_Datatype recvtype, int root, MPI_Comm comm, 
        MPI_Request* request, int tag) {
#ifndef NO_IBCAST
    return MPI_Igatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, 
            recvtype, root, comm, request);
#else
    (void) sendbuf;
    (void) sendcount;
    (void) sendtype;
    (void) recvbuf;
    (void) recvcounts;
    (void) displs;
    (void) recvtype;
    (void) root;
    (void) request;
    (void) comm;
    (void) tag;
    assert(false);
    return -1;
#endif
}

int CollectiveOperations<MPI_Comm, MPI_Request>::Gatherv(void* sendbuf, 
        int sendcount, MPI_Datatype sendtype, void* recvbuf, int* recvcounts,
        int* displs, MPI_Datatype recvtype, int root, MPI_Comm comm, 
        int tag) {
    (void) tag;
    return MPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, 
            recvtype, root, comm);
}

int CollectiveOperations<MPI_Comm, MPI_Request>::Igatherm(void *sendbuf,
        int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount, int root,
        std::function<void (void*, void*, void*)> op, MPI_Comm comm,
        MPI_Request* request, int tag) {
    (void) sendbuf;
    (void) sendcount;
    (void) sendtype;
    (void) recvbuf;
    (void) recvcount;
    (void) root;
    (void) op;
    (void) comm;
    (void) request;
    (void) tag;
    assert(false);
    return -1;
}

int CollectiveOperations<MPI_Comm, MPI_Request>::Gatherm(void *sendbuf,
        int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount, int root, 
        std::function<void (void*, void*, void*)> op, MPI_Comm comm,
        int tag) {
    (void) sendbuf;
    (void) sendcount;
    (void) sendtype;
    (void) recvbuf;
    (void) recvcount;
    (void) root;
    (void) op;
    (void) comm;
    (void) tag;
    assert(false);
    return -1;
}

int CollectiveOperations<MPI_Comm, MPI_Request>::Ireduce(void* sendbuf, 
        void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root,
        MPI_Comm comm, MPI_Request* request, int tag) {
#ifndef NO_IBCAST
    return MPI_Ireduce(sendbuf, recvbuf, count, datatype, op, root, comm, request);
#else
    (void) sendbuf;
    (void) recvbuf;
    (void) count;
    (void) datatype;
    (void) op;
    (void) root;
    (void) comm;
    (void) request;
    (void) tag;
    assert(false);
    return -1;
#endif
}

int CollectiveOperations<MPI_Comm, MPI_Request>::Reduce(void* sendbuf, 
        void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root,
        MPI_Comm comm, int tag) {
    (void) tag;
    return MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
}

int CollectiveOperations<MPI_Comm, MPI_Request>::Iscan(void* sendbuf, 
        void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm const &comm,
        MPI_Request *request, int tag) {
#ifndef NO_IBCAST
    (void) tag;
    return MPI_Iscan(sendbuf, recvbuf, count, datatype, op, comm, request);
#else
    (void) sendbuf;
    (void) recvbuf;
    (void) count;
    (void) datatype;
    (void) op;
    (void) comm;
    (void) request;
    (void) tag;
    assert(false);
    return -1;
#endif
}

int CollectiveOperations<MPI_Comm, MPI_Request>::Scan(void* sendbuf, 
        void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm const &comm,
        int tag) {
    (void) tag;
    return MPI_Scan(sendbuf, recvbuf, count, datatype, op, comm);
}

int CollectiveOperations<MPI_Comm, MPI_Request>::IscanAndBcast(void* sendbuf,
        void* recvbuf_scan, void* recvbuf_bcast, int count, MPI_Datatype datatype,
        MPI_Op op, MPI_Comm const &comm, MPI_Request *request, int tag) {
    (void) sendbuf;
    (void) recvbuf_scan;
    (void) recvbuf_bcast;
    (void) count;
    (void) datatype;
    (void) op;
    (void) comm;
    (void) request;
    (void) tag;
    assert(false);
    return -1;
}

int CollectiveOperations<MPI_Comm, MPI_Request>::ScanAndBcast(void *sendbuf,
        void *recvbuf_scan, void *recvbuf_bcast, int count, MPI_Datatype datatype,
        MPI_Op op, MPI_Comm const &comm, int tag) {
    (void) sendbuf;
    (void) recvbuf_scan;
    (void) recvbuf_bcast;
    (void) count;
    (void) datatype;
    (void) op;
    (void) comm;
    (void) tag;
    assert(false);
    return -1;
}

int CollectiveOperations<MPI_Comm, MPI_Request>::Isend(void *sendbuf, int count, 
        MPI_Datatype datatype, int dest, int tag, MPI_Comm const &comm, MPI_Request *request) {
    return MPI_Isend(sendbuf, count, datatype, dest, tag, comm, request);
}

int CollectiveOperations<MPI_Comm, MPI_Request>::Send(void *sendbuf, int count,
        MPI_Datatype datatype, int dest, int tag, MPI_Comm const &comm) {
    return MPI_Send(sendbuf, count, datatype, dest, tag, comm);
}

int CollectiveOperations<MPI_Comm, MPI_Request>::Irecv(void *buffer, int count,
        MPI_Datatype datatype, int source, int tag, MPI_Comm const &comm, MPI_Request *request) {
    return MPI_Irecv(buffer, count, datatype, source, tag, comm, request);
}

int CollectiveOperations<MPI_Comm, MPI_Request>::Recv(void *buf, int count, 
        MPI_Datatype datatype, int source, int tag, MPI_Comm const &comm, MPI_Status *status) {
    return MPI_Recv(buf, count, datatype, source, tag, comm, status);
}

int CollectiveOperations<MPI_Comm, MPI_Request>::Sendrecv(void *sendbuf,
        int sendcount, MPI_Datatype sendtype,
        int dest, int sendtag,
        void *recvbuf, int recvcount, MPI_Datatype recvtype,
        int source, int recvtag,
        MPI_Comm const &comm, MPI_Status *status) {
    return MPI_Sendrecv(sendbuf, sendcount, sendtype,
            dest, sendtag,
            recvbuf, recvcount, recvtype,
            source, recvtag,
            comm, status);
}

int CollectiveOperations<MPI_Comm, MPI_Request>::Iprobe(int source, int tag, 
        MPI_Comm const &comm, int *flag, MPI_Status *status) {
    return MPI_Iprobe(source, tag, comm, flag, status);
}

int CollectiveOperations<MPI_Comm, MPI_Request>::Probe(int source, int tag, 
        MPI_Comm const &comm, MPI_Status *status) {
    return MPI_Probe(source, tag, comm, status);
}

int CollectiveOperations<MPI_Comm, MPI_Request>::Ibarrier(MPI_Comm comm, 
        MPI_Request *request) {
#ifndef NO_IBCAST
    return MPI_Ibarrier(comm, request);
#else
    (void) request;
    (void) comm;
    assert(false);
    return 0;
#endif
}

int CollectiveOperations<MPI_Comm, MPI_Request>::Barrier(MPI_Comm comm) {
    return MPI_Barrier(comm);
}

int CollectiveOperations<MPI_Comm, MPI_Request>::Test(MPI_Request *request, int *flag,
        MPI_Status *status) {
    return MPI_Test(request, flag, status);
}

int CollectiveOperations<MPI_Comm, MPI_Request>::Wait(MPI_Request *request, 
        MPI_Status *status) {
    return MPI_Wait(request, status);
}

int CollectiveOperations<MPI_Comm, MPI_Request>::Testall(int count, 
        MPI_Request array_of_requests[], int *flag, MPI_Status array_of_statuses[]) {
    return MPI_Testall(count, array_of_requests, flag, array_of_statuses);
}

int CollectiveOperations<MPI_Comm, MPI_Request>::Waitall(int count,
        MPI_Request array_of_requests[], MPI_Status array_of_statuses[]) {
    return MPI_Waitall(count, array_of_requests, array_of_statuses);
}

int CollectiveOperations<MPI_Comm, MPI_Request>::Comm_rank(MPI_Comm const &comm, int *rank) {
    if (comm == MPI_COMM_NULL) {
        *rank = -1;
        return 0;
    }
    else 
        return MPI_Comm_rank(comm, rank);
}

int CollectiveOperations<MPI_Comm, MPI_Request>::Comm_size(MPI_Comm const &comm, int *size) {
    if (comm == MPI_COMM_NULL) {
        *size = -1;
        return 0;
    }
    else 
        return MPI_Comm_size(comm, size);
}

void CollectiveOperations<MPI_Comm, MPI_Request>::Create_Comm_from_MPI(MPI_Comm mpi_comm,
        MPI_Comm* comm) {
    *comm = mpi_comm;
}

void CollectiveOperations<MPI_Comm, MPI_Request>::Create_Comm(const MPI_Comm& comm, 
        int first, int last, MPI_Comm* new_comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    int color;
    if (rank >= first && rank <= last)     
        color = 1;
    else
        color = MPI_UNDEFINED;
        
    MPI_Comm_split(comm, color, rank, new_comm);
}

void CollectiveOperations<MPI_Comm, MPI_Request>::createNewCommunicators(MPI_Comm const &comm, 
        int left_end, int right_start, MPI_Comm *left, MPI_Comm *right, bool) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    int color1, color2;
    if (rank <= left_end)     
        color1 = 1;
    else
        color1 = MPI_UNDEFINED;
        
    if (rank >= right_start)   
        color2 = 2;
    else
        color2 = MPI_UNDEFINED;
    
    MPI_Comm_split(comm, color1, rank, left); 
    MPI_Comm_split(comm, color2, rank, right);
}

int CollectiveOperations<MPI_Comm, MPI_Request>::getSource(MPI_Comm const &, 
        MPI_Status status) {
    return status.MPI_SOURCE;
}

int CollectiveOperations<MPI_Comm, MPI_Request>::freeComm(MPI_Comm *comm) {
    if(*comm != MPI_COMM_WORLD && *comm != MPI_COMM_NULL)
	return MPI_Comm_free(comm);
    return 0;
}
