/*****************************************************************************
 * This file is part of the Project ShizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#include <mpi.h>
#include <cmath>
#include "MPI_Ranged.hpp"
#include "Requests.hpp"
#include "Functions.hpp"
#include <iostream>
#include <cassert>

void Range::Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount, MPI_Datatype recvtype,
        int root, int tag, Range::Comm comm) {
    Range::Request request;
    Igather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root,
            tag, comm, &request);
    Wait(&request, MPI_STATUS_IGNORE);
}

/*
 * Request for the gather
 */
class Range_Requests::Igather : public Range::R_Req {
public:
    Igather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount, MPI_Datatype recvtype,
        int root, int tag, std::function<void (void*, void*, void*)> op,
        const int *recvcounts, const int *displs, Range::Comm comm);
    ~Igather();
    int test(int *flag, MPI_Status *status);
private:
    const void *sendbuf;
    int sendcount, recvcount, root, tag;
    MPI_Datatype sendtype, recvtype;
    void *recvbuf;
    std::function<void (void*, void*, void*)> op;
    const int *recvcounts, *displs;
    int own_height, size, rank, height, received, count, new_rank, 
        sendtype_size, recvtype_size, recv_size;
    Range::Comm comm;
    bool receive, send, completed;
    char *recv_buf;
    Range::Request recv_req, send_req;
};


void Range::Igather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount, MPI_Datatype recvtype,
        int root, int tag, Range::Comm comm, Range::Request* request) {
    int size;
    Range::Comm_size(comm, &size);
    std::function<void (void*, void*, void*)> op = 
        [](void* a, void* b, void* c) { return; };
    *request = std::unique_ptr<R_Req>(new Range_Requests::Igather(sendbuf, sendcount,
            sendtype, recvbuf, sendcount * size, recvtype, root, tag, 
            op, nullptr, nullptr, comm));
};

void Range::Igatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, const int *recvcounts, const int *displs, 
        MPI_Datatype recvtype, int root, int tag, Range::Comm comm,
        Range::Request* request) {
    int size;
    Range::Comm_size(comm, &size);
    std::function<void (void*, void*, void*)> op = 
        [](void* a, void* b, void* c) { return; };
    int recvcount;
    for (int i = 0; i < size; i++)
        recvcount += recvcounts[i];
    *request = std::unique_ptr<R_Req>(new Range_Requests::Igather(sendbuf, sendcount,
            sendtype, recvbuf, recvcount, recvtype, root, tag, 
            op, recvcounts, displs, comm));
};

void Range::Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, const int *recvcounts, const int *displs, 
        MPI_Datatype recvtype, int root, int tag, Range::Comm comm) {
    Range::Request request;
    Igatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype,
            root, tag, comm, &request);
    Wait(&request, MPI_STATUS_IGNORE);
}

void Range::Igatherm(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount, int root, int tag, 
        std::function<void (void*, void*, void*)> op, Range::Comm comm,
        Range::Request* request) {
    *request = std::unique_ptr<R_Req>(new Range_Requests::Igather(sendbuf, sendcount,
            sendtype, recvbuf, recvcount, sendtype, root, tag, 
            op, nullptr, nullptr, comm));
};

void Range::Gatherm(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount, int root, int tag,
        std::function<void (void*, void*, void*)> op, Range::Comm comm) {
    Range::Request request;
    Igatherm(sendbuf, sendcount, sendtype, recvbuf, recvcount, root,
            tag, op, comm, &request);
    Wait(&request, MPI_STATUS_IGNORE);
}

Range_Requests::Igather::Igather(const void *sendbuf, int sendcount, 
        MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
        int root, int tag, std::function<void (void*, void*, void*)> op,
        const int *recvcounts, const int *displs, Range::Comm comm) 
        : sendbuf(sendbuf), sendcount(sendcount), recvcount(recvcount), root(root), tag(tag), 
        sendtype(sendtype), recvtype(recvtype), recvbuf(recvbuf),
         op(op), recvcounts(recvcounts), displs(displs),
        own_height(0), size(0), rank(0), height(1), received(0), count(0), 
        comm(comm), receive(false), send(false), completed(false) {
    Range::Comm_rank(comm, &rank);
    Range::Comm_size(comm, &size);
    new_rank = (rank - root) % size;
    int max_height = log2(size);
    if (pow(2, max_height) < size)
        max_height++;
    for (int i = 0; ((new_rank >> i) % 2 == 0) && (i < max_height); i++)
        own_height++;
    MPI_Type_size(recvtype, &recvtype_size);
    MPI_Type_size(sendtype, &sendtype_size);
    recv_size = recvcount * sendtype_size;
    if (rank == root && recvcounts == nullptr) {        
        assert(recvbuf != nullptr);
        recv_buf = static_cast<char*>(recvbuf);
    }
    else
        recv_buf = new char[recv_size];
    //Copy send data into receive buffer
    copyArray(recv_buf, static_cast<const char*>(sendbuf), sendcount * sendtype_size);  
    received = sendcount;
};

Range_Requests::Igather::~Igather() {
    if (rank != root || recvcounts != nullptr)
        delete[] recv_buf;
}

int Range_Requests::Igather::test(int *flag, MPI_Status *status) {     
    if (completed) {
        *flag = 1;
        return 0;
    }
    
    //If messages have to be received
    if (height <= own_height) {        
        if (!receive) {
            int tmp_src = new_rank + pow(2, height - 1);
            if (tmp_src < size) {
                int src = (tmp_src + root) % size;
                //Range::Test if message can be received
                MPI_Status status;
                int ready;
                Range::Iprobe(src, tag, comm, &ready, &status);
                if (ready) {
                    //Receive message with non-blocking receive
                    MPI_Get_count(&status, sendtype, &count);
                     Range::Irecv(recv_buf + received * sendtype_size, count, sendtype, src, tag, comm, &recv_req);
                    receive = true;
                }
            } else {
                //Source rank larger than comm size
                height++;
            }
        } else {
            //Range::Test if receive finished
            int finished;
            Range::Test(&recv_req, &finished, MPI_STATUS_IGNORE);
            if (finished) {       
                //Merge the received data
                op(recv_buf, recv_buf + received * sendtype_size,
                        recv_buf + (received + count) * sendtype_size);
                received += count;
                height++;
                receive = false;
            }
        }
    }

    //If all messages have been received
    if (height > own_height) {
        if(rank == root) {
            //root doesn't send to anyone
            completed = true;
            assert(recvcount == received);
            if (recvcounts != nullptr) {
                char *buf = static_cast<char*>(recvbuf);
                char *recv_ptr = recv_buf;
                for (int i = 0; i < size; i++) {
                    copyArray(buf + displs[i] * sendtype_size, recv_ptr,
                            recvcounts[i] * sendtype_size);
                    recv_ptr += recvcounts[i] * sendtype_size;
                }
            }                
        } else if (!send) {      
            //Start non-blocking send to parent node
            int tmp_dest = new_rank - pow(2, height - 1);
            int dest = (tmp_dest + root) % size;
             Range::Isend(recv_buf, received, sendtype, dest, tag, comm, &send_req);
            send = true;
        } else {
            //Gather is completed when the send is finished
            int finished;
            Range::Test(&send_req, &finished, MPI_STATUS_IGNORE);
            if (finished) {
                completed = true;
            }
        }
    }
    
    if (completed)
        *flag = 1;
    return 0;
    
}
