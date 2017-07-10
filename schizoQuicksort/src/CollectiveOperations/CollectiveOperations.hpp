/*****************************************************************************
 * This file is part of the Project ShizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#ifndef COLLECTIVEOPERATIONS_HPP
#define COLLECTIVEOPERATIONS_HPP

#include <mpi.h>
#include <vector>
#include <functional>

template <typename COMM, typename REQ>
class CollectiveOperations {
public:
    
    /**
     * Non-blocking broadcast
     * @param buffer Buffer where the broadcast value will be stored
     * @param count Number of elements that will be broadcasted
     * @param datatype MPI datatype of the elements
     * @param root The rank that initially has the broadcast value
     * @param tag Tag to differentiate between multiple calls
     * @param comm The communicator on which the operation is performed
     * @param request Request that will be returned
     */
    static int Ibcast(void *buffer, int count, MPI_Datatype datatype, int root,
            COMM const &comm, REQ *request, int tag = 0);
    
    /**
     * Blocking broadcast
     * @param buffer Buffer where the broadcast value will be stored
     * @param count Number of elements that will be broadcasted
     * @param datatype MPI datatype of the elements
     * @param root The rank that initially has the broadcast value
     * @param tag Tag to differentiate between multiple calls
     * @param comm The communicator on which the operation is performed
     */
    static int Bcast(void *buffer, int count, MPI_Datatype datatype, int root,
            COMM const &comm, int tag = 0);
    
    static bool implementsIbcast();
    static bool implementsIscanAndBcast();
    
    /**
     * Non-blocking gather with equal amount of elements on each process
     * @param sendbuf Starting address of send buffer
     * @param sendcount Number of elements in send buffer
     * @param sendtype MPI datatype of the elements
     * @param recvbuf Buffer where the gathered elements will be stored (only relevant at root)
     * @param recvcount Number of elements for each receive 
     * @param recvtype MPI datatype of the receive elements
     * @param root Rank of receiving process
     * @param tag Tag to differentiate between multiple calls
     * @param comm The communicator on which the operation is performed
     * @param request Request that will be returned
     */
    static void Igather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, int recvcount, MPI_Datatype recvtype,
            int root, COMM comm, REQ* request, int tag = 0);
    
    /**
     * Blocking gather with equal amount of elements on each process
     * @param sendbuf Starting address of send buffer
     * @param sendcount Number of elements in send buffer
     * @param sendtype MPI datatype of the elements
     * @param recvbuf Buffer where the gathered elements will be stored (only relevant at root)
     * @param recvcount Number of elements for each receive 
     * @param recvtype MPI datatype of the receive elements
     * @param root Rank of receiving process
     * @param tag Tag to differentiate between multiple calls
     * @param comm The communicator on which the operation is performed
     */
    static void Gather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, int recvcount, MPI_Datatype recvtype,
            int root, COMM comm, int tag = 0);
    
    /**
     * Non-blocking gather with specified number of elements on each process
     * @param sendbuf Starting address of send buffer
     * @param sendcount Number of elements in send buffer
     * @param sendtype MPI datatype of the elements
     * @param recvbuf Buffer where the gathered elements will be stored (only relevant at root)
     * @param recvcounts Array containing the number of elements that are received from each process 
     * @param displs Array, entry i specifies the displacement relative to recvbuf at which to place the incoming data from process i
     * @param recvtype MPI datatype of the receive elements
     * @param root Rank of receiving process
     * @param tag Tag to differentiate between multiple calls
     * @param comm The communicator on which the operation is performed
     * @param request Request that will be returned
     */
    static void Igatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, int *recvcounts, int *displs, MPI_Datatype recvtype,
            int root, COMM comm, REQ* request, int tag = 0);
    
    /**
     * Blocking gather with specified number of elements on each process
     * @param sendbuf Starting address of send buffer
     * @param sendcount Number of elements in send buffer
     * @param sendtype MPI datatype of the elements
     * @param recvbuf Buffer where the gathered elements will be stored (only relevant at root)
     * @param recvcounts Array containing the number of elements that are received from each process 
     * @param displs Array, entry i specifies the displacement relative to recvbuf at which to place the incoming data from process i
     * @param recvtype MPI datatype of the receive elements
     * @param root Rank of receiving process
     * @param tag Tag to differentiate between multiple calls
     * @param comm The communicator on which the operation is performed
     */
    static void Gatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, int *recvcounts, int *displs, MPI_Datatype recvtype,
            int root, COMM comm, int tag = 0);
    
    /**
     * Non-blocking gather that merges the data via a given function
     * @param sendbuf Starting address of send buffer
     * @param sendcount Number of elements in send buffer
     * @param sendtype MPI datatype of the elements
     * @param recvbuf Buffer where the gathered elements will be stored (only relevant at root)
     * @param recvcount Number of total elements that will be received
     * @param root Rank of receiving process
     * @param tag Tag to differentiate between multiple calls
     * @param op Operation that takes (start, mid, end) as parameters and 
     *  merges the two arrays [start, mid) and [mid, end) in-place
     * @param comm The communicator on which the operation is performed
     * @param request Request that will be returned
     */
    static int Igatherm(void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, int recvcount, int root,
            std::function<void (void*, void*, void*)> op, COMM comm, REQ* request,
            int tag = 0);
    
    /**
     * Blocking gather that merges the data via a given function
     * @param sendbuf Starting address of send buffer
     * @param sendcount Number of elements in send buffer
     * @param sendtype MPI datatype of the elements
     * @param recvbuf Buffer where the gathered elements will be stored (only relevant at root)
     * @param recvcount Number of total elements that will be received
     * @param root Rank of receiving process
     * @param tag Tag to differentiate between multiple calls
     * @param op Operation that takes (start, mid, end) as parameters and 
     *  merges the two arrays [start, mid) and [mid, end) in-place
     * @param comm The communicator on which the operation is performed
     */
    static void Gatherm(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, int root,
                std::function<void (void*, void*, void*)> op, COMM comm, int tag = 0);
    
    /**
     * Non-blocking reduce
     * @param sendbuf Starting address of send buffer
     * @param recvbuf Starting address of receive buffer
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param tag Tag to differentiate between multiple calls
     * @param op Operation used to reduce two elements
     * @param root Rank of receiving process
     * @param comm The communicator on which the operation is performed
     * @param request Request that will be returned
     */
    static void Ireduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
            MPI_Op op, int root, COMM comm, REQ *request, int tag = 0);
    
    /**
     * Blocking reduce
     * @param sendbuf Starting address of send buffer
     * @param recvbuf Starting address of receive buffer
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param tag Tag to differentiate between multiple calls
     * @param op Operation used to reduce two elements
     * @param root Rank of receiving process
     * @param comm The communicator on which the operation is performed
     */
    static void Reduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
            MPI_Op op, int root, COMM comm, int tag = 0);
    
    /**
     * Non-blocking scan (partial reductions)
     * @param sendbuf Starting address of send buffer
     * @param recvbuf Starting address of receive buffer
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param tag Tag to differentiate between multiple calls
     * @param op Operation used to reduce two elements
     * @param comm The communicator on which the operation is performed
     * @param request Request that will be returned
     */
    static int Iscan(void *sendbuf, void *recvbuf, int count, 
            MPI_Datatype datatype, MPI_Op op, COMM const &comm, REQ *request, int tag = 0); 
    
    /**
     * Blocking scan (partial reductions)
     * @param sendbuf Starting address of send buffer
     * @param recvbuf Starting address of receive buffer
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param tag Tag to differentiate between multiple calls
     * @param op Operation used to reduce two elements
     * @param comm The communicator on which the operation is performed
     */    
    static int Scan(void *sendbuf, void *recvbuf, int count, 
            MPI_Datatype datatype, MPI_Op op, COMM const &comm, int tag = 0);  
    
    /**
     * Non-blocking scan (partial reductions) and broadcast of the reduction over all elements
     * @param sendbuf Starting address of send buffer
     * @param recvbuf_scan Starting address of receive buffer for the scan value
     * @param recvbuf_bcast Starting address of receive buffer for the broadcast value
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param tag Tag to differentiate between multiple calls
     * @param op Operation used to reduce two elements
     * @param comm The communicator on which the operation is performed
     * @param request Request that will be returned
     */
    static int IscanAndBcast(void *sendbuf, void *recvbuf_scan, void *recvbuf_bcast,
            int count, MPI_Datatype datatype, MPI_Op op, COMM const &comm, REQ *request,
            int tag = 0);
    
    /**
     * Non-blocking scan (partial reductions) and broadcast of the reduction over all elements
     * @param sendbuf Starting address of send buffer
     * @param recvbuf_scan Starting address of receive buffer for the scan value
     * @param recvbuf_bcast Starting address of receive buffer for the broadcast value
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param tag Tag to differentiate between multiple calls
     * @param op Operation used to reduce two elements
     * @param comm The communicator on which the operation is performed
     */
    static int ScanAndBcast(void *sendbuf, void *recvbuf_scan, void *recvbuf_bcast,
            int count, MPI_Datatype datatype, MPI_Op op, COMM const &comm,
            int tag = 0); 
    
    /**
     * Non-blocking send
     * @param sendbuf Starting address of send buffer
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param dest Destination rank
     * @param tag Tag to differentiate between multiple calls
     * @param comm The communicator on which the operation is performed
     * @param request Request that will be returned
     */
    static int Isend(void *sendbuf, int count, MPI_Datatype datatype,
            int dest, int tag, COMM const &comm, REQ *request);
    
    /**
     * Blocking send
     * @param sendbuf Starting address of send buffer
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param dest Destination rank
     * @param tag Tag to differentiate between multiple calls
     * @param comm The communicator on which the operation is performed
     */
    static int Send(void *sendbuf, int count, MPI_Datatype datatype,
            int dest, int tag, COMM const &comm);
    
    /**
     * Non-blocking receive
     * @param sendbuf Starting address of receive buffer
     * @param count Number of elements to be received
     * @param datatype MPI datatype of the elements
     * @param dest Source rank, can be MPI_ANY_SOURCE
     * @param tag Tag to differentiate between multiple calls
     * @param comm The communicator on which the operation is performed
     * @param request Request that will be returned
     */
    static int Irecv(void *buffer, int count, MPI_Datatype datatype, int source,
            int tag, COMM const &comm, REQ *request);
    
    /**
     * Blocking receive
     * @param sendbuf Starting address of receive buffer
     * @param count Number of elements to be received
     * @param datatype MPI datatype of the elements
     * @param dest Source rank, can be MPI_ANY_SOURCE
     * @param tag Tag to differentiate between multiple calls
     * @param comm The communicator on which the operation is performed
     */
    static int Recv(void *buf, int count, MPI_Datatype datatype, int source,
            int tag, COMM const &comm, MPI_Status *status);
    
    /**
     * Rend receive operation
     * @param sendbuf Starting address of send buffer
     * @param sendcount Number of elements to be send
     * @param sendtype MPI datatype of the elements
     * @param dest Target rank
     * @param sendtag Tag to differentiate between multiple calls
     * @param recvbuf Starting address of the receive buffer
     * @param recvcount Number of elements to be send
     * @param recvtype MPI datatype of the elements
     * @param source Source rank
     * @param recvtag Tag to differentiate between multiple calls
     * @param comm Starting address of the receive buffer
     */
    static int Sendrecv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
            int dest, int sendtag,
            void *recvbuf, int recvcount, MPI_Datatype recvtype,
            int source, int recvtag,
            COMM const &comm, MPI_Status *status);
    
    /**
     * Test if a message can be received
     * @param source Source rank, can be MPI_ANY_SOURCE
     * @param tag Message tag, can be MPI_ANY_TAG
     * @param comm The communicator on which the operation is performed
     * @param flag Returns 1 if message can be received, else 0
     * @param status Returns a status for the message, can be MPI_STATUS_IGNORE
     * @return 
     */
    static int Iprobe(int source, int tag, COMM const &comm, int *flag, MPI_Status *status);
    
    /**
     * Block until a message can be received
     * @param source Source rank, can be MPI_ANY_SOURCE
     * @param tag Message tag, can be MPI_ANY_TAG
     * @param comm The communicator on which the operation is performed
     * @param status Returns a status for the message, can be MPI_STATUS_IGNORE
     * @return 
     */
    static int Probe(int source, int tag, COMM const &comm, MPI_Status *status);
    
    /**
     * Non-blocking barrier
     * @param comm The communicator on which the operation is performed
     * @param request Request that will be returned
     */
    static int Ibarrier(COMM comm, REQ *request);
    
    /**
     * Blocking barrier
     * @param comm The communicator on which the operation is performed
     */
    static int Barrier(COMM comm);
    
    /**
     * Test if a operation is completed
     * @param request Request of the operation
     * @param flag Returns 1 if operation completed, else 0
     * @param status Returns a status if completed, can be MPI_STATUS_IGNORE
     * @return 
     */
    static int Test(REQ *request, int *flag, MPI_Status *status);
    
    /**
     * Wait until a operation is completed
     * @param request Request of the operation
     * @param status Returns a status if completed, can be MPI_STATUS_IGNORE
     * @return 
     */
    static int Wait(REQ *request, MPI_Status *status);
    
    /**
     * Test if multiple operations are completed
     * @param count Number of operations
     * @param array_of_requests Array of requests of the operations
     * @param flag Returns 1 if all operations completed, else 0
     * @param array_of_statuses Array of statuses for the operations, can be MPI_STATUSES_IGNORE
     */
    static int Testall(int count, REQ array_of_requests[], int *flag,
            MPI_Status array_of_statuses[]);
    
    /**
     * Wait until multiple operations are completed
     * @param count Number of operations
     * @param array_of_requests Array of requests of the operations
     * @param array_of_statuses Array of statuses for the operations, can be MPI_STATUSES_IGNORE
     */
    static int Waitall(int count, REQ array_of_requests[], MPI_Status array_of_statuses[]);
    
    /**
     * Get the rank of this process on the communicator
     * @param comm The communicator
     * @param rank Returns the rank
     */
    static int Comm_rank(COMM const &comm, int *rank);
    
    /**
     * Get the size of a communicator
     * @param comm The communicator
     * @param size Returns the size
     */
    static int Comm_size(COMM const &comm, int *size);
    
    /**
     * Create a new communicator from a MPI communicator
     * The communicatorr includes all ranks of the MPI communicator
     * @param mpi_comm the MPI communicator
     * @param comm returns the new communicator
     */
    static void Create_Comm_from_MPI(MPI_Comm mpi_comm, COMM *comm);
    
    /**
     * Create a new communicator that includes a subgroup of ranks [first, last]
     * of the ranks from the old communicator
     * @param comm old communicator
     * @param first first rank from the old communicator that is included in the new communicator
     * @param last last rank from the old communicator that is included in the new communicator
     * @param new_comm return the new communicator
     */
    static void Create_Comm(COMM const &comm, int first, int last,
            COMM *new_comm);
    
    static void createNewCommunicators(COMM const &comm, int left_end, 
            int right_start, COMM *left, COMM *right, bool mpi_split);
    
    /**
     * Get the source rank from a MPI_Status
     * @param comm the communicator the communication is based on
     * @param status a MPI_Status
     * @return the source rank of the status
     */
    static int getSource(COMM const &comm, MPI_Status status);
    
    /**
     * Free the communicator
     * @param comm 
     * @return 
     */
    static int freeComm(COMM *comm);
};

#endif	// COLLECTIVEOPERATIONS_HPP

