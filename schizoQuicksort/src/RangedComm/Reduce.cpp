/*****************************************************************************
 * This file is part of the Project ShizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#include <mpi.h>
#include "MPI_Ranged.hpp"
#include "Functions.hpp"
#include "Requests.hpp"
#include <cmath>

void Range::Reduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
        int tag, MPI_Op op, int root, Range::Comm comm) {
    Range::Request request;
    Ireduce(sendbuf, recvbuf, count, datatype, tag, op, root, comm, &request);
    Wait(&request, MPI_STATUS_IGNORE);
}


/*
 * Request for the reduce
 */
class Range_Requests::Ireduce : public Range::R_Req {
public:
    Ireduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
            int tag, MPI_Op op, int root, Range::Comm comm);
    ~Ireduce();
    int test(int *flag, MPI_Status *status);
private:
    const void *sendbuf;
    void *recvbuf;
    int count, tag, root;
    MPI_Datatype datatype;
    MPI_Op op;
    Range::Comm comm;
    bool send, completed;
    int rank, size, new_rank, height, own_height, datatype_size, recv_size,
        receives;
    char *recvbuf_arr, *reduce_buf;
    Range::Request send_req;
    std::vector<Range::Request> recv_requests;
};

void Range::Ireduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, 
        int tag, MPI_Op op, int root, Range::Comm comm, Range::Request* request) {
    *request = std::unique_ptr<R_Req>(new Range_Requests::Ireduce(sendbuf, recvbuf,
            count, datatype, tag, op, root, comm));
}

Range_Requests::Ireduce::Ireduce(const void* sendbuf, void* recvbuf, int count,
        MPI_Datatype datatype, int tag, MPI_Op op, int root, Range::Comm comm) : 
        sendbuf(sendbuf), recvbuf(recvbuf), count(count), tag(tag), root(root), 
        datatype(datatype), op(op), comm(comm), send(false), completed(false) {
    
    Range::Comm_rank(comm, &rank);
    Range::Comm_size(comm, &size);
    MPI_Type_size(datatype, &datatype_size);
    recv_size = count * datatype_size;
    new_rank = rank - root - 1;
    if (new_rank < 0)
        new_rank += size;
    
    height = log2(size);
    if (pow(2, height) < size)
        height++;
    
    own_height = 0;
    if (new_rank == (size - 1)) {
        own_height = height;
    } else {
        for (int i = 0; ((new_rank >> i) % 2 == 1) && (i < height); i++)
            own_height++;
    }
    
    recvbuf_arr = new char[recv_size * own_height];
    reduce_buf = new char[recv_size];
    copyArray(reduce_buf, static_cast<const char*>(sendbuf), recv_size);     
}

Range_Requests::Ireduce::~Ireduce() {
    delete[] recvbuf_arr;
    delete[] reduce_buf;
}

int Range_Requests::Ireduce::test(int* flag, MPI_Status* status) {
    if (completed) {
        *flag = 1;
        return 0;
    }
    
    if (height > 0 && recv_requests.size() == 0) {
        //Receive data
        int tmp_rank = new_rank;
        if (new_rank == size - 1)
            tmp_rank = pow(2, height) - 1;

        for (int i = own_height - 1; i >= 0; i--) {
            int tmp_src = tmp_rank - pow(2, i);
            if (tmp_src < new_rank) {
                recv_requests.push_back(Range::Request());
                int src = (tmp_src + root + 1) % size;
                 Range::Irecv(recvbuf_arr + (recv_requests.size() - 1) * recv_size, count,
                        datatype, src,
                        tag, comm, &recv_requests.back());
            } else {
                tmp_rank = tmp_src;
            }
        }
        receives = recv_requests.size();
    }

    if (!send) {
        int recv_finished;
        Range::Testall(recv_requests.size(), &recv_requests.front(), &recv_finished, 
                MPI_STATUSES_IGNORE);
        if (recv_finished && receives > 0) {
            //Reduce received data and local data
            for (int i = 0; i < (receives - 1); i++) {
                MPI_Reduce_local(recvbuf_arr + i * recv_size,
                        recvbuf_arr + (i + 1) * recv_size, count, datatype, op);
            }
            MPI_Reduce_local(recvbuf_arr + (receives - 1) * recv_size,
                    reduce_buf, count, datatype, op);
        }

        //Send data
        if (recv_finished) {
            if (new_rank < size - 1) {
                int tmp_dest = new_rank + pow(2, own_height);
                if (tmp_dest > size - 1)
                    tmp_dest = size - 1;
                int dest = (tmp_dest + root + 1) % size;
                 Range::Isend(reduce_buf, count, datatype, dest, tag, comm, &send_req);
            }
            send = true;
        }
    }
    if (send) {
        if (new_rank == size - 1) {
            copyArray(static_cast<char*>(recvbuf), reduce_buf, recv_size);
            *flag = 1;
        }
        else
            Range::Test(&send_req, flag, MPI_STATUS_IGNORE);  
        if (*flag)
            completed = true;      
    }
    return 0;
}

