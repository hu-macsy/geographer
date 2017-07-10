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

void Range::ScanAndBcast(const void* sendbuf, void* recvbuf_scan, 
        void* recvbuf_bcast, int count, MPI_Datatype datatype, int tag,
        MPI_Op op, Range::Comm comm) {
    Range::Request request;
    IscanAndBcast(sendbuf, recvbuf_scan, recvbuf_bcast, count, datatype, tag,
            op, comm, &request);
    Wait(&request, MPI_STATUS_IGNORE);    
}

/*
 * Request for the scan and broadcast
 */
class Range_Requests::IscanAndBcast : public Range::R_Req {
public:
    IscanAndBcast(const void *sendbuf, void *recvbuf_scan, void *recvbuf_bcast,
            int count, MPI_Datatype datatype, int tag, MPI_Op op, Range::Comm comm);
    ~IscanAndBcast();
    int test(int *flag, MPI_Status *status);

private:
    const void *sendbuf;
    void *recvbuf_scan, *recvbuf_bcast;
    int count;
    MPI_Datatype datatype;
    int tag;
    MPI_Op op;
    Range::Comm comm;
    int rank, size, height, up_height, down_height, receives, sends, recv_size,
    datatype_size;
    std::vector<int> target_ranks;
    char *recvbuf_arr, *tmp_buf, *tmp_buf2, *scan_buf, *bcast_buf;
    bool upsweep, downsweep, send, completed;
    Range::Request send_req, recv_req;
    std::vector<Range::Request> recv_requests;
};


void Range::IscanAndBcast(const void* sendbuf, void* recvbuf_scan, void* recvbuf_bcast,
        int count, MPI_Datatype datatype, int tag, MPI_Op op, Range::Comm comm, 
        Range::Request* request) {
    *request = std::unique_ptr<R_Req>(new Range_Requests::IscanAndBcast(sendbuf, 
            recvbuf_scan, recvbuf_bcast, count, datatype, tag, op, comm));
}

Range_Requests::IscanAndBcast::IscanAndBcast(const void* sendbuf, void* recvbuf_scan,
        void* recvbuf_bcast, int count, MPI_Datatype datatype, int tag, MPI_Op op, 
        Range::Comm comm) : sendbuf(sendbuf), recvbuf_scan(recvbuf_scan), 
        recvbuf_bcast(recvbuf_bcast), count(count), datatype(datatype), tag(tag),
        op(op), comm(comm), receives(0), sends(0), upsweep(true), downsweep(false),
        send(false), completed(false) {
    Range::Comm_rank(comm, &rank);
    Range::Comm_size(comm, &size);
    MPI_Type_size(datatype, &datatype_size);
    recv_size = count * datatype_size;
    height = log2(size);
    if (pow(2, height) < size)
        height++;

    up_height = 0;
    if (rank == (size - 1)) {
        up_height = height;
    } else {
        for (int i = 0; ((rank >> i) % 2 == 1) && (i < height); i++)
            up_height++;
    }
    
    recvbuf_arr = new char[recv_size * (1 + up_height)];
    tmp_buf = new char[recv_size * 2];
    tmp_buf2 = tmp_buf + recv_size;
    scan_buf = new char[recv_size * 2];
    bcast_buf = scan_buf + recv_size;
    down_height = 0;
    if (rank < size - 1)
        down_height = up_height + 1;

    copyArray(scan_buf, static_cast<const char*>(sendbuf), recv_size);
}

Range_Requests::IscanAndBcast::~IscanAndBcast() {
    delete[] recvbuf_arr;
    delete[] tmp_buf;
    delete[] scan_buf;
}

int Range_Requests::IscanAndBcast::test(int* flag, MPI_Status * status) {
    if (completed) {
        *flag = 1;
        return 0;
    }
    if (comm.last - comm.first == 0 && upsweep) {
        copyArray(static_cast<char*>(recvbuf_scan), static_cast<const char*>(sendbuf),
                recv_size);
        copyArray(static_cast<char*>(recvbuf_bcast), static_cast<const char*>(sendbuf), 
                recv_size);
        upsweep = false;
        downsweep = false;
    }
    //upsweep phase
    if (upsweep) {
        //Receive data
        if (up_height > 0 && recv_requests.size() == 0) {
            int tmp_rank = rank;
            if (rank == size - 1)
                tmp_rank = pow(2, height) - 1;

            for (int i = up_height - 1; i >= 0; i--) {
                int source = tmp_rank - pow(2, i);
                if (source < rank) {
                    recv_requests.push_back(Range::Request());
                     Range::Irecv(recvbuf_arr + (recv_requests.size() - 1) * recv_size, 
                            count, datatype, source,
                            tag, comm, &recv_requests.back());
                    //Save communication partner rank in vector
                    target_ranks.push_back(source);
                } else {
                    tmp_rank = source;
                }
            }
            receives = recv_requests.size();
        }

        //Reduce received data
        if (!send) {
            int finished;
            Range::Testall(recv_requests.size(), &recv_requests.front(), &finished, 
                    MPI_STATUSES_IGNORE);
            if (finished && receives > 0) {
                //Reduce received data and local data
                for (int i = 0; i < (receives - 1); i++) {
                    MPI_Reduce_local(recvbuf_arr + i * recv_size,
                            recvbuf_arr + (i + 1) * recv_size, count, datatype, op);
                }
                MPI_Reduce_local(recvbuf_arr + (receives - 1) * recv_size, scan_buf,
                        count, datatype, op);
            }

            //Send data
            if (finished) {
                if (rank < size - 1) {
                    int dest = rank + pow(2, up_height);
                    if (dest > size - 1)
                        dest = size - 1;
                     Range::Isend(scan_buf, count, datatype, dest, tag, comm, &send_req);
                    target_ranks.push_back(dest);
                }
                send = true;
            }
        }
        
        //End upsweep phase when data send
        if (send) {
            int finished = 1;
            if (rank < size - 1)
                Range::Test(&send_req, &finished, MPI_STATUS_IGNORE);
            
            if (finished) {
                upsweep = false;
                downsweep = true;
                send = false;
                
                if (rank == size - 1) {
                    for (int i = 0; i < recv_size; i++) {
                        bcast_buf[i] = scan_buf[i];
                        tmp_buf2[i] = bcast_buf[i];
                        scan_buf[i] = 0;
                    }
                }
            }
        }
    }

    //downsweep phase
    if (downsweep) {
        int finished1 = 0, finished2 = 0;
        if (down_height == height) {
            //Communicate with higher ranks
            if (!send) {
                copyArray(tmp_buf, scan_buf, recv_size);
                int dest = target_ranks.back();
                 Range::Isend(tmp_buf, count, datatype, dest, tag, comm, &send_req);
                 Range::Irecv(scan_buf, count * 2, datatype, dest, tag, comm, &recv_req);
                send = true;
            }
            Range::Test(&send_req, &finished1, MPI_STATUS_IGNORE);
            Range::Test(&recv_req, &finished2, MPI_STATUS_IGNORE);
        } else if (receives >= height) {
            //Communicate with lower ranks
            if (!send) {
                copyArray(tmp_buf, scan_buf, recv_size);
                copyArray(tmp_buf2, bcast_buf, recv_size);
                int dest = target_ranks[sends];
                 Range::Isend(tmp_buf, count * 2, datatype, dest, tag, comm, &send_req);
                 Range::Irecv(scan_buf, count, datatype, dest, tag, comm, &recv_req);
                sends++;
                send = true;            
            }
            Range::Test(&send_req, &finished1, MPI_STATUS_IGNORE);
            Range::Test(&recv_req, &finished2, MPI_STATUS_IGNORE);
            if (finished1 && finished2)
                MPI_Reduce_local(tmp_buf, scan_buf, count, datatype, op);    
        } else
            height--;
        //Send and receive completed
        if (finished1 && finished2) {
            height--;
            send = false;
        }
        //End downsweep phase
        if (height == 0) {
            downsweep = false;
            char *buf = const_cast<char*>(static_cast<const char*>(sendbuf)); 
            MPI_Reduce_local(buf, scan_buf, count, datatype, op);
            copyArray(static_cast<char*>(recvbuf_scan), scan_buf, recv_size);
            copyArray(static_cast<char*>(recvbuf_bcast), bcast_buf, recv_size);
        }
    }    
 
    if(!upsweep && !downsweep) {
        *flag = 1;
        completed = true;
    }
    return 0;
}
