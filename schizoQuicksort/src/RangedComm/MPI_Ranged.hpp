/*****************************************************************************
 * This file is part of the Project ShizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#ifndef MPI_RANGED_HPP
#define MPI_RANGED_HPP

#include <mpi.h>
#include <vector>
#include <memory>

/**
 * Main class for the Range communication
 * All communication related operations are static member functions of this class
 */
class Range {
    friend class Range_Requests;

private:

    /*
     * Private constructor
     */
    Range(){};

    /**
     * Virtual superclass for the specific requests for each communication operation.
     * The header Requests.h defines the subclasses.
     */
    class R_Req {
        friend class Request;
        friend class Range;
    public:

        virtual ~R_Req() {};

    protected:

        R_Req() {};
        //test method has to be implemented by subclasses
        virtual int test(int *flag, MPI_Status *status) = 0;
    };

public:

    /**
     * Ranged based communicator
     */
    class Comm {
        friend class Range;
        friend class Range_Requests;

    public:
        
        /*
         * Create an empty communicator.
         * Use Create_Comm_from_MPI to make a usable communicator.
         */
        Comm();

    private:

        /*
         * Creates a Range comm including the MPI ranks first to last on the given MPI comm
         */
        Comm(MPI_Comm mpi_comm, int first, int last);
        MPI_Comm mpi_comm;
        int first;
        int last;

        /*
         * Returns true if the rank (on the MPI comm) is part of this Range comm
         */
        bool includesRank(int rank);
    };

    /**
     * Request class for Range communication.
     * Each non-blocking operation returns a request that is then used to call
     * the Test, Testall, Wait or Waitall operation.
     */
    class Request {
        friend class Range;

    public:
        
        Request();

    private:
        //Request acts like a pointer to a R_Req
        Request(R_Req *req);
        R_Req& operator*();
        R_Req* operator->();
        void operator=(std::unique_ptr<R_Req> req);

        std::unique_ptr<R_Req> req_ptr;
    };  

    /**
     * Non-blocking broadcast
     * @param buffer Buffer where the broadcast value will be stored
     * @param count Number of elements that will be broadcasted
     * @param datatype MPI datatype of the elements
     * @param root The rank that initially has the broadcast value
     * @param tag Tag to differentiate between multiple calls
     * @param comm The Range comm on which the operation is performed
     * @param request Request that will be returned
     */
    static void Ibcast(void *buffer, int count, MPI_Datatype datatype, int root,
            int tag, Comm comm, Request *request);
    
    /**
     * Blocking broadcast
     * @param buffer Buffer where the broadcast value will be stored
     * @param count Number of elements that will be broadcasted
     * @param datatype MPI datatype of the elements
     * @param root The rank that initially has the broadcast value
     * @param tag Tag to differentiate between multiple calls
     * @param comm The Range comm on which the operation is performed
     */
    static void Bcast(void *buffer, int count, MPI_Datatype datatype, int root,
            int tag, Comm comm);
    
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
     * @param comm The Range comm on which the operation is performed
     * @param request Request that will be returned
     */
    static void Igather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, int recvcount, MPI_Datatype recvtype,
            int root, int tag, Comm comm, Range::Request* request);
    
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
     * @param comm The Range comm on which the operation is performed
     */
    static void Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, int recvcount, MPI_Datatype recvtype,
            int root, int tag, Comm comm);
    
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
     * @param comm The Range comm on which the operation is performed
     * @param request Request that will be returned
     */
    static void Igatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, const int *recvcounts, const int *displs, MPI_Datatype recvtype,
            int root, int tag, Comm comm, Range::Request* request);
    
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
     * @param comm The Range comm on which the operation is performed
     */
    static void Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, const int *recvcounts, const int *displs, MPI_Datatype recvtype,
            int root, int tag, Comm comm);
    
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
     * @param comm The Range comm on which the operation is performed
     * @param request Request that will be returned
     */
    static void Igatherm(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, int root, int tag,
                std::function<void (void*, void*, void*)> op, Comm comm,
                Range::Request* request);
    
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
     * @param comm The Range comm on which the operation is performed
     */
    static void Gatherm(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, int root, int tag,
                std::function<void (void*, void*, void*)> op, Comm comm);
    
    /**
     * Non-blocking reduce
     * @param sendbuf Starting address of send buffer
     * @param recvbuf Starting address of receive buffer
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param tag Tag to differentiate between multiple calls
     * @param op Operation used to reduce two elements
     * @param root Rank of receiving process
     * @param comm The Range comm on which the operation is performed
     * @param request Request that will be returned
     */
    static void Ireduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
            int tag, MPI_Op op, int root, Comm comm, Request *request);
    
    /**
     * Blocking reduce
     * @param sendbuf Starting address of send buffer
     * @param recvbuf Starting address of receive buffer
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param tag Tag to differentiate between multiple calls
     * @param op Operation used to reduce two elements
     * @param root Rank of receiving process
     * @param comm The Range comm on which the operation is performed
     */
    static void Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
            int tag, MPI_Op op, int root, Comm comm);
    
    /**
     * Non-blocking scan (partial reductions)
     * @param sendbuf Starting address of send buffer
     * @param recvbuf Starting address of receive buffer
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param tag Tag to differentiate between multiple calls
     * @param op Operation used to reduce two elements
     * @param comm The Range comm on which the operation is performed
     * @param request Request that will be returned
     */
    static void Iscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
            int tag, MPI_Op op, Comm comm, Request *request);
    
    /**
     * Blocking scan (partial reductions)
     * @param sendbuf Starting address of send buffer
     * @param recvbuf Starting address of receive buffer
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param tag Tag to differentiate between multiple calls
     * @param op Operation used to reduce two elements
     * @param comm The Range comm on which the operation is performed
     */
    static void Scan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
            int tag, MPI_Op op, Comm comm);
    
    /**
     * Non-blocking scan (partial reductions) and broadcast of the reduction over all elements
     * @param sendbuf Starting address of send buffer
     * @param recvbuf_scan Starting address of receive buffer for the scan value
     * @param recvbuf_bcast Starting address of receive buffer for the broadcast value
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param tag Tag to differentiate between multiple calls
     * @param op Operation used to reduce two elements
     * @param comm The Range comm on which the operation is performed
     * @param request Request that will be returned
     */
    static void IscanAndBcast(const void *sendbuf, void *recvbuf_scan, void *recvbuf_bcast,
            int count, MPI_Datatype datatype, int tag, MPI_Op op, Comm comm,
            Request *request);
    
    /**
     * Non-blocking scan (partial reductions) and broadcast of the reduction over all elements
     * @param sendbuf Starting address of send buffer
     * @param recvbuf_scan Starting address of receive buffer for the scan value
     * @param recvbuf_bcast Starting address of receive buffer for the broadcast value
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param tag Tag to differentiate between multiple calls
     * @param op Operation used to reduce two elements
     * @param comm The Range comm on which the operation is performed
     */
    static void ScanAndBcast(const void *sendbuf, void *recvbuf_scan, void *recvbuf_bcast,
            int count, MPI_Datatype datatype, int tag, MPI_Op op, Comm comm);
    
    /**
     * Non-blocking send
     * @param sendbuf Starting address of send buffer
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param dest Destination rank
     * @param tag Tag to differentiate between multiple calls
     * @param comm The Range comm on which the operation is performed
     * @param request Request that will be returned
     */
    static void Isend(const void *sendbuf, int count, MPI_Datatype datatype,
            int dest, int tag, Comm comm, Request *request);
    
    /**
     * Blocking send
     * @param sendbuf Starting address of send buffer
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param dest Destination rank
     * @param tag Tag to differentiate between multiple calls
     * @param comm The Range comm on which the operation is performed
     */
    static int Send(const void *sendbuf, int count, MPI_Datatype datatype,
            int dest, int tag, Comm comm);
    
    /**
     * Non-blocking receive
     * @param sendbuf Starting address of receive buffer
     * @param count Number of elements to be received
     * @param datatype MPI datatype of the elements
     * @param dest Source rank, can be MPI_ANY_SOURCE
     * @param tag Tag to differentiate between multiple calls
     * @param comm The Range comm on which the operation is performed
     * @param request Request that will be returned
     */
    static void Irecv(void *buffer, int count, MPI_Datatype datatype, int source,
            int tag, Comm comm, Request *request);
    
    /**
     * Blocking receive
     * @param sendbuf Starting address of receive buffer
     * @param count Number of elements to be received
     * @param datatype MPI datatype of the elements
     * @param dest Source rank, can be MPI_ANY_SOURCE
     * @param tag Tag to differentiate between multiple calls
     * @param comm The Range comm on which the operation is performed
     */
    static int Recv(void *buf, int count, MPI_Datatype datatype, int source,
            int tag, Comm comm, MPI_Status *status);

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
    static int Sendrecv(void *sendbuf,
            int sendcount, MPI_Datatype sendtype,
            int dest, int sendtag,
            void *recvbuf, int recvcount, MPI_Datatype recvtype,
            int source, int recvtag,
            Comm const &comm, MPI_Status *status);
    
    /**
     * Test if a message can be received
     * @param source Source rank, can be MPI_ANY_SOURCE
     * @param tag Message tag, can be MPI_ANY_TAG
     * @param comm The Range comm on which the operation is performed
     * @param flag Returns 1 if message can be received, else 0
     * @param status Returns a status for the message, can be MPI_STATUS_IGNORE
     * @return 
     */
    static int Iprobe(int source, int tag, Comm comm, int *flag, MPI_Status *status);
    
    /**
     * Block until a message can be received
     * @param source Source rank, can be MPI_ANY_SOURCE
     * @param tag Message tag, can be MPI_ANY_TAG
     * @param comm The Range comm on which the operation is performed
     * @param status Returns a status for the message, can be MPI_STATUS_IGNORE
     * @return 
     */
    static int Probe(int source, int tag, Comm comm, MPI_Status *status);
    
    /**
     * Non-blocking barrier
     * @param comm The Range comm on which the operation is performed
     * @param request Request that will be returned
     */
    static void Ibarrier(Comm comm, Request *request);
    
    /**
     * Blocking barrier
     * @param comm The Range comm on which the operation is performed
     */
    static void Barrier(Comm comm);
    
    /**
     * Test if a operation is completed
     * @param request Request of the operation
     * @param flag Returns 1 if operation completed, else 0
     * @param status Returns a status if completed, can be MPI_STATUS_IGNORE
     * @return 
     */
    static int Test(Request *request, int *flag, MPI_Status *status);
    
    /**
     * Wait until a operation is completed
     * @param request Request of the operation
     * @param status Returns a status if completed, can be MPI_STATUS_IGNORE
     * @return 
     */
    static int Wait(Request *request, MPI_Status *status);
    
    /**
     * Test if multiple operations are completed
     * @param count Number of operations
     * @param array_of_requests Array of requests of the operations
     * @param flag Returns 1 if all operations completed, else 0
     * @param array_of_statuses Array of statuses for the operations, can be MPI_STATUSES_IGNORE
     */
    static void Testall(int count, Request array_of_requests[], int *flag,
            MPI_Status array_of_statuses[]);
    
    /**
     * Wait until multiple operations are completed
     * @param count Number of operations
     * @param array_of_requests Array of requests of the operations
     * @param array_of_statuses Array of statuses for the operations, can be MPI_STATUSES_IGNORE
     */
    static void Waitall(int count, Request array_of_requests[],
            MPI_Status array_of_statuses[]);
    
    /**
     * Get the rank of this process on the communicator
     * @param comm The Range communicator
     * @param rank Returns the rank
     */
    static void Comm_rank(Comm comm, int *rank);
    
    /**
     * Get the size of a Range communicator
     * @param comm The Range communicator
     * @param size Returns the size
     */
    static void Comm_size(Comm comm, int *size);

    /**
     * Create a new communicator from a MPI communicator
     * The communicatorr includes all ranks of the MPI communicator
     * @param mpi_comm the MPI communicator
     * @param comm returns the new communicator
     */
    static void Create_Comm_from_MPI(MPI_Comm mpi_comm, Comm *rcomm);
    
    /**
     * Create a new communicator that includes a subgroup of ranks [first, last]
     * of the ranks from the old communicator
     * @param comm old communicator
     * @param first first rank from the old communicator that is included in the new communicator
     * @param last last rank from the old communicator that is included in the new communicator
     * @param new_comm return the new communicator
     */
    static void Create_Comm(Comm comm, int first, int last,
        Comm *new_comm);

    /**
     * Returns the rank in the communicator of the source from a MPI_Status
     * @param comm the communicator
     * @param status the status
     */
    static int get_Rank_from_Status(Range::Comm const &comm, MPI_Status status); 
};

#endif /* MPI_RANGED_HPP */
