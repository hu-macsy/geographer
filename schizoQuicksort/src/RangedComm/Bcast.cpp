/*****************************************************************************
 * This file is part of the Project ShizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (c) 2016-2017, Michael Axtmann <michael.axtmann@kit.edu>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#include <mpi.h>
#include "MPI_Ranged.hpp"
#include "Requests.hpp"
#include <cmath>

void Range::Bcast(void* buffer, int count, MPI_Datatype datatype, int root, 
        int tag, Range::Comm comm) {
    Range::Request request;
    MPI_Status status;
    int rank = 0;
    int global_rank = 0;
    int size = 0;
    int own_height = 0;
    Range::Comm_rank(comm, &rank);
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    Range::Comm_size(comm, &size);
    int temp_rank = rank - root;
    if (temp_rank < 0)
        temp_rank += size;
    int height = log2(size);
    if (pow(2, height) < size)
        height++;
    for (int i = 0; ((temp_rank >> i) % 2 == 0) && (i < height); i++)
        own_height++;
    if (rank != root){
        int temp_rank = rank - root;
        if (temp_rank < 0)
            temp_rank += size;
        int mask = 0x1;
        while ((temp_rank ^ mask) > temp_rank) {
            mask = mask << 1;
        }
        int temp_src = temp_rank ^ mask;
        int src = (temp_src + root) % size;
        Range::Recv(buffer, count, datatype, src, tag, comm, &status);
    }

    while (height > 0) {
        if (own_height >= height) {
            int temp_rank = rank - root;
            if (temp_rank < 0)
                temp_rank += size;
            int temp_dest = temp_rank + pow(2, height - 1);
            if (temp_dest < size) {
                int dest = (temp_dest + root) % size;
                Range::Send(buffer, count, datatype, dest, tag, comm);
            }
        }
        height--;
    }
    return;
}
    
/*
 * Request for the broadcast
 */
class Range_Requests::Ibcast : public Range::R_Req {
public:
    Ibcast(void *buffer, int count, MPI_Datatype datatype, int root,
            int tag, Range::Comm comm);
    int test(int *flag, MPI_Status *status);

private:
    void *buffer;
    MPI_Datatype datatype;
    int count, root, tag, own_height, size, rank, height, received, sends;
    Range::Comm comm;
    bool send, completed;
    Range::Request recv_req;
    std::vector<Range::Request> req_vector;
};

void Range::Ibcast(void *buffer, int count, MPI_Datatype datatype,
        int root, int tag, Range::Comm comm, Range::Request *request) {
    *request = std::unique_ptr<R_Req>(new Range_Requests::Ibcast(buffer, count, 
            datatype, root, tag, comm));
};

Range_Requests::Ibcast::Ibcast(void *buffer, int count, MPI_Datatype datatype,
        int root, int tag, Range::Comm comm) : buffer(buffer), datatype(datatype),
        count(count), root(root), tag(tag), own_height(0), size(0), rank(0), 
        height(0), received(0), comm(comm), send(false), completed(false) {
    Range::Comm_rank(comm, &rank);
    Range::Comm_size(comm, &size);
    sends = 0;
    int temp_rank = rank - root;
    if (temp_rank < 0)
        temp_rank += size;
    height = log2(size);
    if (pow(2, height) < size)
        height++;
    for (int i = 0; ((temp_rank >> i) % 2 == 0) && (i < height); i++)
        own_height++;
    if (rank == root)
        received = 1;
    else
         Range::Irecv(buffer, count, datatype, MPI_ANY_SOURCE, tag, comm, &recv_req);
};

int Range_Requests::Ibcast::test(int *flag, MPI_Status *status) {
    if (completed) {
        *flag = 1;
        return 0;
    }
    if (!received) {  
            Range::Test(&recv_req, &received, MPI_STATUS_IGNORE);
        }
    if (received && !send) {
        while (height > 0) {
            if (own_height >= height) {
                int temp_rank = rank - root;
                if (temp_rank < 0)
                    temp_rank += size;
                int temp_dest = temp_rank + pow(2, height - 1);
                if (temp_dest < size) {
                    int dest = (temp_dest + root) % size;
                    req_vector.push_back(Range::Request());
                     Range::Isend(buffer, count, datatype, dest, tag, comm, &req_vector.back());
                }
            }
            height--;
        }
        send = true;
    }
    if (send) {
        Range::Testall(req_vector.size(), &req_vector.front(), flag, MPI_STATUSES_IGNORE);        
        if (*flag == 1)
            completed = true;
    }
    return 0;
};

