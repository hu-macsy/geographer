/*****************************************************************************
 * This file is part of the Project ShizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#ifndef RANGEOPERATIONS_HPP
#define RANGEOPERATIONS_HPP

#include "../RangedComm/MPI_Ranged.hpp"
#include "CollectiveOperations.hpp"

template<>
class CollectiveOperations<Range::Comm, Range::Request> {
    
public:

    static int Ibcast(void* buffer, int count, MPI_Datatype datatype, int root, 
            Range::Comm const &comm, Range::Request* request, int tag);
    
    static int Bcast(void* buffer, int count, MPI_Datatype datatype, int root,
            Range::Comm const &comm, int tag);

    static bool implementsIbcast();

    static bool implementsIscanAndBcast();

    static int Igather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, int recvcount, MPI_Datatype recvtype,
            int root, Range::Comm comm, Range::Request* request, int tag);
    
    static int Gather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, int recvcount, MPI_Datatype recvtype,
            int root, Range::Comm comm, int tag);
    
    static int Igatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, int *recvcounts, int *displs, MPI_Datatype recvtype,
            int root, Range::Comm comm, Range::Request* request, int tag);
    
    static int Gatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, int *recvcounts, int *displs, MPI_Datatype recvtype,
            int root, Range::Comm comm, int tag);
    
    static int Igatherm(void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount, int root,
        std::function<void (void*, void*, void*)> op, Range::Comm comm,
        Range::Request* request, int tag);
    
    static int Gatherm(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, int root,
                std::function<void (void*, void*, void*)> op, Range::Comm comm, int tag);
    
    static int Ireduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
            MPI_Op op, int root, Range::Comm comm, Range::Request *request, int tag);
    
    static int Reduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
            MPI_Op op, int root, Range::Comm comm, int tag);
     
    static int Iscan(void *sendbuf, void *recvbuf, int count, 
            MPI_Datatype datatype, MPI_Op op, Range::Comm const &comm, 
            Range::Request *request, int tag); 
    
    static int Scan(void *sendbuf, void *recvbuf, int count, 
            MPI_Datatype datatype, MPI_Op op, Range::Comm const &comm, int tag);
   
    static int IscanAndBcast(void* sendbuf, void* recvbuf_scan, 
            void* recvbuf_bcast, int count, MPI_Datatype datatype,
            MPI_Op op, Range::Comm const &comm, Range::Request *requestm, int tag);
    
    static int ScanAndBcast(void *sendbuf, void *recvbuf_scan, void *recvbuf_bcast,
            int count, MPI_Datatype datatype, MPI_Op op, Range::Comm const &comm,
            int tag);

    static int Isend(void *sendbuf, int count, MPI_Datatype datatype,
            int dest, int tag, Range::Comm const &comm, Range::Request *request) ; 
    
    static int Send(void *sendbuf, int count, MPI_Datatype datatype,
            int dest, int tag, Range::Comm const &comm);

    static int Irecv(void *buffer, int count, MPI_Datatype datatype, int source,
            int tag, Range::Comm const &comm, Range::Request *request);

    static int Recv(void *buf, int count, MPI_Datatype datatype, int source, 
            int tag, Range::Comm const &comm, MPI_Status *status);

    static int Sendrecv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
            int dest, int sendtag,
            void *recvbuf, int recvcount, MPI_Datatype recvtype,
            int source, int recvtag,
            Range::Comm const &comm, MPI_Status *status);

    static int Iprobe(int source, int tag, Range::Comm const &comm, int *flag, 
            MPI_Status *status);
    
    static int Probe(int source, int tag, Range::Comm const &comm, MPI_Status *status);
    
    static int Ibarrier(Range::Comm comm, Range::Request *request);
    
    static int Barrier(Range::Comm comm);
    
    static int Test(Range::Request *request, int *flag, MPI_Status *status);

    static int Wait(Range::Request *request, MPI_Status *status);

    static int Testall(int count, Range::Request array_of_requests[], int *flag,
            MPI_Status array_of_statuses[]);

    static int Waitall(int count, Range::Request array_of_requests[], 
            MPI_Status array_of_statuses[]);
    
    static int Comm_rank(Range::Comm const &comm, int *rank);

    static int Comm_size(Range::Comm const &comm, int *size);
    
    static void Create_Comm_from_MPI(MPI_Comm mpi_comm, Range::Comm *comm);
    
    static void Create_Comm(Range::Comm const &comm, int first, int last,
            Range::Comm *new_comm);
    
    static void createNewCommunicators(Range::Comm const &comm, int left_end,
            int right_start, Range::Comm *left, Range::Comm *right, bool mpi_split);
    
    static int getSource(Range::Comm const &comm, MPI_Status status);    
    
    static int freeComm(Range::Comm *comm);
};

#endif /* RANGEOPERATIONS_HPP */
