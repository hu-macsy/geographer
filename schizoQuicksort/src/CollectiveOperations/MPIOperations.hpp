/*****************************************************************************
 * This file is part of the Project ShizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#ifndef MPIOPERATIONS_HPP
#define MPIOPERATIONS_HPP

#include <mpi.h>
#include "CollectiveOperations.hpp"

template <>
class CollectiveOperations<MPI_Comm, MPI_Request> {
public:

    static int Ibcast(void* buffer, int count, MPI_Datatype datatype, int root, 
            MPI_Comm const &comm, MPI_Request* request, int tag);
    
    static int Bcast(void* buffer, int count, MPI_Datatype datatype, int root,
            MPI_Comm const &comm, int tag);

    static bool implementsIbcast();

    static bool implementsIscanAndBcast();

    static int Igather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, int recvcount, MPI_Datatype recvtype,
            int root, MPI_Comm comm, MPI_Request* request, int tag);
    
    static int Gather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, int recvcount, MPI_Datatype recvtype,
            int root, MPI_Comm comm, int tag);
    
    static int Igatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, int *recvcounts, int *displs, MPI_Datatype recvtype,
            int root, MPI_Comm comm, MPI_Request* request, int tag);
    
    static int Gatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, int *recvcounts, int *displs, MPI_Datatype recvtype,
            int root, MPI_Comm comm, int tag);
    
    static int Igatherm(void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount, int root,
        std::function<void (void*, void*, void*)> op, MPI_Comm comm,
        MPI_Request* request, int tag);
    
    static int Gatherm(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, int root,
                std::function<void (void*, void*, void*)> op, MPI_Comm comm, int tag);
    
    static int Ireduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
            MPI_Op op, int root, MPI_Comm comm, MPI_Request *request, int tag);
    
    static int Reduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
            MPI_Op op, int root, MPI_Comm comm, int tag);
    
    static int Iscan(void *sendbuf, void *recvbuf, int count, 
            MPI_Datatype datatype, MPI_Op op, MPI_Comm const &comm, 
            MPI_Request *request, int tag); 
    
    static int Scan(void *sendbuf, void *recvbuf, int count, 
            MPI_Datatype datatype, MPI_Op op, MPI_Comm const &comm, int tag); 
    
    static int IscanAndBcast(void* sendbuf, void* recvbuf_scan, 
            void* recvbuf_bcast, int count, MPI_Datatype datatype,
            MPI_Op op, MPI_Comm const &comm, MPI_Request *requestm, int tag);
    
    static int ScanAndBcast(void *sendbuf, void *recvbuf_scan, void *recvbuf_bcast,
            int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm const &comm,
            int tag);

    static int Isend(void *sendbuf, int count, MPI_Datatype datatype,
            int dest, int tag, MPI_Comm const &comm, MPI_Request *request) ; 
    
    static int Send(void *sendbuf, int count, MPI_Datatype datatype,
            int dest, int tag, MPI_Comm const &comm);

    static int Irecv(void *buffer, int count, MPI_Datatype datatype, int source,
            int tag, MPI_Comm const &comm, MPI_Request *request);

    static int Recv(void *buf, int count, MPI_Datatype datatype, int source, 
            int tag, MPI_Comm const &comm, MPI_Status *status);

    static int Sendrecv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
            int dest, int sendtag,
            void *recvbuf, int recvcount, MPI_Datatype recvtype,
            int source, int recvtag,
            MPI_Comm const &comm, MPI_Status *status);

    static int Iprobe(int source, int tag, MPI_Comm const &comm, int *flag, 
            MPI_Status *status);
    
    static int Probe(int source, int tag, MPI_Comm const &comm, MPI_Status *status);
    
    static int Ibarrier(MPI_Comm comm, MPI_Request *request);
    
    static int Barrier(MPI_Comm comm);
    
    static int Test(MPI_Request *request, int *flag, MPI_Status *status);

    static int Wait(MPI_Request *request, MPI_Status *status);

    static int Testall(int count, MPI_Request array_of_requests[], int *flag,
            MPI_Status array_of_statuses[]);

    static int Waitall(int count, MPI_Request array_of_requests[], 
            MPI_Status array_of_statuses[]);
    
    static int Comm_rank(MPI_Comm const &comm, int *rank);

    static int Comm_size(MPI_Comm const &comm, int *size);
    
    static void Create_Comm_from_MPI(MPI_Comm mpi_comm, MPI_Comm *comm);
    
    static void Create_Comm(MPI_Comm const &comm, int first, int last,
            MPI_Comm *new_comm);
    
    static void createNewCommunicators(MPI_Comm const &comm, int left_end,
            int right_start, MPI_Comm *left, MPI_Comm *right, bool mpi_split);
    
    static int getSource(MPI_Comm const &comm, MPI_Status status);
    
    static int freeComm(MPI_Comm *comm);
};

#endif /* MPIOPERATIONS_HPP */
