/*****************************************************************************
 * This file is part of the Project ShizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#include "MPI_Ranged.hpp"
#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <cmath>

int world_size = 0, world_rank = 0;
Range::Comm worldcomm;
std::vector<MPI_Comm> global_comms;

void barrier(double &start, double &end) {
    Range::Request request;
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    Range::Ibarrier(worldcomm, &request);
    Range::Wait(&request, MPI_STATUS_IGNORE);
    end = MPI_Wtime();
}

void bcast(double &start, double &end, int N) {
    double *arr = new double[N];
    if (world_rank == 0) {
        for (int i = 0; i < N; i++)
            arr[i] = i;
    }
    Range::Request request;
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    Range::Ibcast(arr, N, MPI_DOUBLE, 0, 12, worldcomm, &request);
    Range::Wait(&request, MPI_STATUS_IGNORE);    
    end = MPI_Wtime();
    assert(abs(arr[N - 1] - (N - 1)) < 0.01);
    delete[] arr;
}

void bcastMPI(double &start, double &end, int N) {
    double *arr = new double[N];
    if (world_rank == 0) {
        for (int i = 0; i < N; i++)
            arr[i] = i;
    }
    MPI_Request request;

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
#ifndef NO_IBCAST
    MPI_Ibcast(arr, N, MPI_DOUBLE, 0, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
#else    
    MPI_Bcast(arr, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    end = MPI_Wtime();
    assert(abs(arr[N - 1] - (N - 1)) < 0.01);
    delete[] arr;
}

void bcast100(double &start, double &end, int N) {
    double *arr = new double[N];
    if (world_rank == 0) {
        for (int i = 0; i < N; i++)
            arr[i] = i;
    }
    Range::Request request;
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    for (int i = 0; i < 50; i++) {
	if (world_rank == 0)
	    arr[N - 1] = 1;
	Range::Bcast(arr, N, MPI_DOUBLE, 0, 12, worldcomm);
        if ((abs(arr[N - 1] - 1) > 0.01))
	    exit(10);
	if (world_rank == 0)
	    arr[N - 1] = 2;
	Range::Bcast(arr, N, MPI_DOUBLE, 0, 12, worldcomm);
        if ((abs(arr[N - 1] - 2) > 0.01))
	    exit(11);
    }
    end = MPI_Wtime();
    delete[] arr;
}

void bcast100_MPI(double &start, double &end, int N) {
    double *arr = new double[N];
    if (world_rank == 0) {
        for (int i = 0; i < N; i++)
            arr[i] = i;
    }
    MPI_Request request;

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    for (int i = 0; i < 50; i++) {
	if (world_rank == 0)
	    arr[N - 1] = 1;
	MPI_Bcast(arr, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if ((abs(arr[N - 1] - 1) > 0.01))
	    exit(13);
	if (world_rank == 0)
	    arr[N - 1] = 2;
	MPI_Bcast(arr, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if ((abs(arr[N - 1] - 2) > 0.01))
	    exit(14);
    }
    end = MPI_Wtime();
    delete[] arr;
}
void bcast100r(double &start, double &end, int N) {
    double *arr = new double[N];
    if (world_rank == 0) {
        for (int i = 0; i < N; i++)
            arr[i] = i;
    }
    Range::Request request;
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    for (int i = 0; i < 50; i++) {
	if (world_rank == 0)
	    arr[N - 1] = 1;
	Range::Bcast(arr, N, MPI_DOUBLE, 0, 12, worldcomm);
	if (world_rank == world_size - 1)
	    arr[N - 1] = 2;
	Range::Bcast(arr, N, MPI_DOUBLE, world_size - 1, 12, worldcomm);
    }
    end = MPI_Wtime();
    delete[] arr;
}

void bcast100r_MPI(double &start, double &end, int N) {
    double *arr = new double[N];
    if (world_rank == 0) {
        for (int i = 0; i < N; i++)
            arr[i] = i;
    }
    MPI_Request request;

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    for (int i = 0; i < 50; i++) {
	if (world_rank == 0)
	    arr[N - 1] = 1;
	MPI_Bcast(arr, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if ((abs(arr[N - 1] - 1) > 0.01))
	    exit(17);
	if (world_rank == world_size - 1)
	    arr[N - 1] = 2;
	MPI_Bcast(arr, N, MPI_DOUBLE, world_size - 1, MPI_COMM_WORLD);
        if ((abs(arr[N - 1] - 2) > 0.01))
	    exit(18);
    }
    end = MPI_Wtime();
    delete[] arr;
}

void reduce(double &start, double &end, int N) {
    double *arr = new double[N];
    for (int i = 0; i < N; i++)
        arr[i] = world_rank;

    double *reduce = new double[N];
    Range::Request request;

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    Range::Ireduce(arr, reduce, N, MPI_DOUBLE, 0, MPI_SUM, world_size - 1, worldcomm, &request);
    Range::Wait(&request, MPI_STATUS_IGNORE);
    end = MPI_Wtime();
    delete[] arr; delete[] reduce;
}

void reduceMPI(double &start, double &end, int N) {
    double *arr = new double[N];
    for (int i = 0; i < N; i++)
        arr[i] = world_rank;

    double *reduce = new double[N];
    MPI_Request request;
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
#ifndef NO_IBCAST   
    MPI_Ireduce(arr, reduce, N, MPI_DOUBLE, MPI_SUM, world_size - 1, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
#else
    MPI_Reduce(arr, reduce, N, MPI_DOUBLE, MPI_SUM, world_size - 1, MPI_COMM_WORLD);
#endif
    end = MPI_Wtime();
    delete[] arr; delete[] reduce;
}

void scan(double &start, double &end, int N) {
    double *arr = new double[N];
    for (int i = 0; i < N; i++)
        arr[i] = world_rank;

    double *scan = new double[N];
    Range::Request request;

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    Range::Iscan(arr, scan, N, MPI_DOUBLE, 0, MPI_SUM, worldcomm, &request);
    Range::Wait(&request, MPI_STATUS_IGNORE);
    end = MPI_Wtime();
    delete[] arr; delete[] scan;
}

void scanMPI(double &start, double &end, int N) {
    double *arr = new double[N];
    for (int i = 0; i < N; i++)
        arr[i] = world_rank;

    double *scan = new double[N];
    MPI_Request request;
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
#ifndef NO_IBCAST   
    MPI_Iscan(arr, scan, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
#else
    MPI_Scan(arr, scan, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    end = MPI_Wtime();
    delete[] arr; delete[] scan;
}

void scanAndBcast(double &start, double &end, int N) {
    double *arr = new double[N];
    for (int i = 0; i < N; i++)
        arr[i] = world_rank;

    double *scan = new double[N];
    double *bcast = new double[N];
    Range::Request request;

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    Range::IscanAndBcast(arr, scan, bcast, N, MPI_DOUBLE, 0, MPI_SUM, worldcomm, &request);
    Range::Wait(&request, MPI_STATUS_IGNORE);
    end = MPI_Wtime();
    delete[] arr; delete[] scan; delete[] bcast;
}

void scanAndBcastMPI(double &start, double &end, int N) {
    double *arr = new double[N];
    for (int i = 0; i < N; i++)
        arr[i] = world_rank;

    double *scan = new double[N];
    double *bcast = new double[N];
    MPI_Request request;
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
#ifndef NO_IBCAST
    MPI_Iscan(arr, scan, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
#else
    MPI_Scan(arr, scan, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    if (world_rank == world_size - 1) {
        for (int i = 0; i < N; i++)
            bcast[i] = scan[i];
    }
#ifndef NO_IBCAST
    MPI_Ibcast(bcast, N, MPI_DOUBLE, world_size - 1, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
#else
    MPI_Bcast(bcast, N, MPI_DOUBLE, world_size - 1, MPI_COMM_WORLD);
#endif
    end = MPI_Wtime();
    delete[] arr; delete[] scan; delete[] bcast;
}

void split_bcast(double &start, double &end, int N, int bcast_iterations) {
/*    
    MPI_Comm comm = MPI_COMM_WORLD;
    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    
    int split = size / 2;
    
    double *arr = new double[N];
    if (rank == 0 || rank == split) {
        for (int i = 0; i < N; i++)
            arr[i] = i + rank;
    }
    Range::Comm rcomm;
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    
    if (rank < split)
        rcomm = Range::Comm(comm, 0, split - 1);
    else
        rcomm = Range::Comm(comm, split, size - 1);
    
    for (int i = 0; i < bcast_iterations; i++) {
        Range::Bcast(arr, N, MPI_DOUBLE, 0, 55, rcomm);
	if (rank != 0 && rank != split) {
	    if (rank < split) {
                if (abs(arr[N-1] - (N-1)) > 0.1)
	            exit(12);
	    } else {
		if (abs(arr[N-1] - (N-1+split)) > 0.1)
		    exit(12);
	    }
	    arr[N-1] = -1;
	}
    } 
    
    end = MPI_Wtime();
    delete[] arr;
*/
}

void split_bcastMPI(double &start, double &end, int N, int bcast_iterations) {
    MPI_Comm comm = MPI_COMM_WORLD;
    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    
    MPI_Comm new_comm;
    int color;
    if (rank < size / 2)
        color = 1;
    else
        color = 2;
    
    double *arr = new double[N];
    if (rank == 0 || rank == size / 2) {
        for (int i = 0; i < N; i++)
            arr[i] = i + rank;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    
    MPI_Comm_split(comm, color, rank, &new_comm);    
    int split = size / 2;
    for (int i = 0; i < bcast_iterations; i++) {
        MPI_Bcast(arr, N, MPI_DOUBLE, 0, new_comm);
	if (rank != 0 && rank != split) {
	    if (rank < split) {
                if (abs(arr[N-1] - (N-1)) > 0.1)
	            exit(12);
	    } else {
		if (abs(arr[N-1] - (N-1+split)) > 0.1)
		    exit(12);
	    }
	    arr[N-1] = -1;
	}
    } 
    end = MPI_Wtime();
    delete[] arr;
}

void splitCommRange(Range::Comm comm, bool recursive) {
    int size, rank;
    Range::Comm_size(comm, &size);
    Range::Comm_rank(comm, &rank);
    if (size < 2)
        return;

    Range::Comm new_comm;
    if (rank < size / 2)
	Range::Create_Comm(comm, 0, size / 2 - 1, &new_comm);
    else
	Range::Create_Comm(comm, size / 2, size - 1, &new_comm);
    
    if (recursive)
        splitCommRange(new_comm, true);
}

void splitComm_split(MPI_Comm comm, bool recursive) {
    if (MPI_COMM_NULL == comm)
        return;

    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    if (size < 2) {
        if (comm != MPI_COMM_WORLD)
            MPI_Comm_free(&comm);
        return;
    }

    MPI_Comm new_comm;
    int color;
    if (rank < size / 2)
        color = 1;
    else
        color = 2;
    
    MPI_Comm_split(comm, color, rank, &new_comm);  

    if (comm != MPI_COMM_WORLD)
        MPI_Comm_free(&comm);

    if (recursive) {
        splitComm_split(new_comm, true);
    } else {
        MPI_Comm_free(&new_comm);
    }
}

void splitComm_group(MPI_Comm &comm, bool recursive) {
    if (MPI_COMM_NULL == comm)
        return;

    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    if (size <= 1) {
        if (comm != MPI_COMM_WORLD)
            MPI_Comm_free(&comm);
        return;
    }

    MPI_Group group, new_group;
    MPI_Comm_group(comm, &group);
    int ranges[2][3] = {{0, size / 2 - 1, 1},{size / 2, size - 1, 1}};
    
    if (rank < size / 2)
        MPI_Group_range_incl(group, 1, &ranges[0], &new_group);
    else
        MPI_Group_range_incl(group, 1, &ranges[1], &new_group);
    MPI_Comm new_comm;
    MPI_Comm_create(comm, new_group, &new_comm);

    if (comm != MPI_COMM_WORLD)
        MPI_Comm_free(&comm);

    if (recursive) {
        splitComm_group(new_comm, true);
    } else {
        MPI_Comm_free(&new_comm);
    }
}

void splitCommOnceRange(double &start, double &end, MPI_Comm comm) {
    Range::Comm rcomm;
    Range::Create_Comm_from_MPI(comm, &rcomm);
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();    
    splitCommRange(rcomm, false);
    end = MPI_Wtime();
}

void splitCommRecursiveRange(double &start, double &end, MPI_Comm comm) {
    Range::Comm rcomm;
    Range::Create_Comm_from_MPI(comm, &rcomm);
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    splitCommRange(rcomm, true);
    end = MPI_Wtime();
}

void splitCommRecursive_group(double &start, double &end, MPI_Comm comm) {
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    splitComm_group(comm, true);
    end = MPI_Wtime();
}

void splitCommRecursive_split(double &start, double &end, MPI_Comm comm) {
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    splitComm_split(comm, true);
    end = MPI_Wtime();
}

void splitCommOnce_group(double &start, double &end, MPI_Comm comm) {
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    splitComm_group(comm, false);
    end = MPI_Wtime();
}

void splitCommOnce_split(double &start, double &end, MPI_Comm comm) {
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    splitComm_split(comm, false);
    end = MPI_Wtime();
}

void Gather(double &start, double &end, int N) {
    int local = N;
    N *= world_size;
    assert(N >= 0);
    double* arr = new double[local];
    for (int i = 0; i < local; i++)
        arr[i] = i + (world_size - world_rank) * 6.77;
    
    double *recv = new double[N];
    
    Range::Request request;            
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    Range::Igather(arr, local, MPI_DOUBLE, recv, local, MPI_DOUBLE, 0, 12, worldcomm, &request);    
    Range::Wait(&request, MPI_STATUS_IGNORE);
    end = MPI_Wtime();
    delete[] arr; delete[] recv;
}

void GatherMPI(double &start, double &end, int N) {
    int local = N;
    N *= world_size;
    assert(N >= 0);
    double* arr = new double[local];
    for (int i = 0; i < local; i++)
        arr[i] = i + (world_size - world_rank) * 6.77;
    
    double *recv = new double[N];
           
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
#ifndef NO_IBCAST    
    MPI_Request request; 
    MPI_Igather(arr, local, MPI_DOUBLE, recv, local, MPI_DOUBLE, 0, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
#else
    MPI_Gather(arr, local, MPI_DOUBLE, recv, local, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    
    end = MPI_Wtime();
    delete[] arr; delete[] recv;
}

//void Gather_sparse(double &start, double &end, int N) {
//    int N;    
//    if (world_rank % 2 == 1)
//        N = 1;
//    else
//        N = 0;
//    
//    double arr[N];
//    for (int i = 0; i < N; i++)
//        arr[i] = i + (world_size - world_rank) * 6.77;
//    
//    int recv_size = world_size / 2;
//    double recv[recv_size];
//    std::function<void (void*, void*, void*) > op =
//            [](void* start, void* mid, void* end) {
//                std::inplace_merge<double*>(static_cast<double*> (start),
//                        static_cast<double*> (mid), static_cast<double*> (end));
//            };
//         
//    Range::Comm rcomm = Range::Comm(MPI_COMM_WORLD, 0, world_size - 1);
//    MPI_Barrier(MPI_COMM_WORLD);
//    start = MPI_Wtime();
//    Range::Request request;   
//    Range::Igatherm(arr, N, MPI_DOUBLE, recv, recv_size, 0, 12, op, rcomm, &request);    
//    Range::Wait(&request, MPI_STATUS_IGNORE);
//    end = MPI_Wtime();
//
//    if (world_rank == 0) {
//        for (int i = 1; i < recv_size; i++) {
//            assert(recv[i - 1] <= recv[i]);
//        }
//    }           
//}
//
//void Gather_sparseMPI(double &start, double &end, int N) {
//    int N;    
//    if (world_rank % 2 == 1)
//        N = 1;
//    else
//        N = 0;
//    
//    double arr[N];
//    for (int i = 0; i < N; i++)
//        arr[i] = world_rank;
//    int elements[world_size], displs[world_size];
//    for (int i = 0; i < world_size; i++) {
//        elements[i] = i % 2;
//        displs[i] = (i-1) / 2;
//    }
//    
//    
//    int recv_size = world_size / 2;
//    double recv[recv_size];
//           
//    MPI_Barrier(MPI_COMM_WORLD);
//    start = MPI_Wtime();
//#ifndef NO_IBCAST
//    MPI_Request request;        
//    MPI_Igatherv(arr, N, MPI_DOUBLE, recv, elements, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD, &request); 
//    MPI_Wait(&request, MPI_STATUS_IGNORE);
//#else
//    MPI_Gatherv(arr, N, MPI_DOUBLE, recv, elements, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//#endif
//    end = MPI_Wtime();
//
//    if (world_rank == 0) {
//        for (int i = 0; i < recv_size; i++)
//            assert(recv[i] == (i+1) * 2 - 1);
//    }           
//}



void executeBenchmark(std::string benchmark, std::ofstream &file, int N, int iteration) {
    double start, end;
    /* Benchmark */
    if (benchmark == "barrier")
        barrier(start, end);
    else if (benchmark == "bcast")
        bcast(start, end, N);
    else if (benchmark == "bcastMPI")
        bcastMPI(start, end, N);
    else if (benchmark == "bcast100")
        bcast100(start, end, N);
    else if (benchmark == "bcast100_MPI")
        bcast100_MPI(start, end, N);
    else if (benchmark == "bcast100r")
        bcast100r(start, end, N);
    else if (benchmark == "bcast100r_MPI")
        bcast100r_MPI(start, end, N);
    else if (benchmark == "reduce")
        reduce(start, end, N);
    else if (benchmark == "reduceMPI")
        reduceMPI(start, end, N);
    else if (benchmark == "scan")
        scan(start, end, N);
    else if (benchmark == "scanMPI")
        scanMPI(start, end, N);
    else if (benchmark == "scanAndBcast")
        scanAndBcast(start, end, N);
    else if (benchmark == "scanAndBcastMPI")
        scanAndBcastMPI(start, end, N);
    else if (benchmark == "splitCommOnceRange") {
        if (N > 1)
            return;
        splitCommOnceRange(start, end, MPI_COMM_WORLD);
    } else if (benchmark == "splitCommRecursiveRange") {
        if (N > 1)
            return;
        splitCommRecursiveRange(start, end, MPI_COMM_WORLD);
    } else if (benchmark == "splitCommRecursive_group") {
        if (N > 1)
            return;
        splitCommRecursive_group(start, end, MPI_COMM_WORLD);
    } else if (benchmark == "splitCommRecursive_split") {
        if (N > 1)
            return;
        splitCommRecursive_split(start, end, MPI_COMM_WORLD);
    } else if (benchmark == "splitCommOnce_group") {
        if (N > 1)
            return;
        splitCommOnce_group(start, end, MPI_COMM_WORLD);
    } else if (benchmark == "splitCommOnce_split") {
        if (N > 1)
            return;
        splitCommOnce_split(start, end, MPI_COMM_WORLD);
    } else if (benchmark == "gather") {
        if (N > std::pow(2,25) / world_size)
	    return;
	Gather(start, end, N);
    }
    else if (benchmark == "gatherMPI") {
        if (N > std::pow(2,25) / world_size)
	    return;
        GatherMPI(start, end, N);
    }
//    else if (benchmark == "gather_sparse")
//        Gather_sparse(start, end, N);
//    else if (benchmark == "gather_sparseMPI")
//        Gather_sparseMPI(start, end, N);
    else if (benchmark == "split_bcast_0")
        split_bcast(start, end, N, 0);
    else if (benchmark == "split_bcast_1")
        split_bcast(start, end, N, 1);
    else if (benchmark == "split_bcast_10")
        split_bcast(start, end, N, 10);
    else if (benchmark == "split_bcast_50")
        split_bcast(start, end, N, 50);
    else if (benchmark == "split_bcast_MPI_0")
        split_bcastMPI(start, end, N, 0);
    else if (benchmark == "split_bcast_MPI_1")
        split_bcastMPI(start, end, N, 1);
    else if (benchmark == "split_bcast_MPI_10")
        split_bcastMPI(start, end, N, 10);
    else if (benchmark == "split_bcast_MPI_50")
        split_bcastMPI(start, end, N, 50);
    else {
        if (world_rank == 0) {
            std::cout << "Error: not recognized" << std::endl;
        }
        return;
    }

    MPI_Comm comm = MPI_COMM_WORLD;
    double runningTimes[world_size];
    double runningTime = (end - start);
    MPI_Gather(&runningTime, 1, MPI_DOUBLE, runningTimes, 1, MPI_DOUBLE, 0, comm);
    if (world_rank == 0) {
        double max_time = runningTime;
        for (int i = 0; i < world_size; i++) {
            if (runningTimes[i] > max_time)
                max_time = runningTimes[i];
        }
        std::cout << "time: " << max_time * 1000 << "ms" << std::endl;
        std::string result = "RESULT benchmark=" + benchmark 
                + " size=" + std::to_string(world_size)
                + " elements=" + std::to_string(N) 
                + " time=" + std::to_string(max_time)
                + " iteration=" + std::to_string(iteration)
                + "\n";
        file << result;
    }
}

/*
 * arguments: number_of_repeats, benchmark1, benchmark2, ..
 * arguments optional
 */
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    Range::Create_Comm_from_MPI(MPI_COMM_WORLD,  &worldcomm);
    if (world_rank == 0)
        std::cout << "Starting Benchmarks" << std::endl;
    std::vector<std::string> arguments;
    std::vector<std::string> benchmarks{"bcast", "bcastMPI", "reduce", "reduceMPI",
        "scan", "scanMPI", "scanAndBcast", "scanAndBcastMPI",
	"gather", "gatherMPI",	
        "splitCommOnceRange", "splitCommRecursiveRange",
//        "splitCommOnce_group",  "splitCommRecursive_group",
        "splitCommOnce_split","splitCommRecursive_split",
        "split_bcast_0", "split_bcast_MPI_0",
        "split_bcast_1", "split_bcast_MPI_1", "split_bcast_10", 
        "split_bcast_MPI_10", "split_bcast_50", "split_bcast_MPI_50"};
    int repeats = 12;
#ifdef N_ELEMENTS
    int min_elements = 0;
    int max_elements = 18;
#else
    int min_elements = 0;
    int max_elements = 0;
#endif    
    if (argc >= 2)
        repeats = atoi(argv[1]);
    if (argc >= 3)
        min_elements = atoi(argv[2]);
    if (argc >= 4)
        max_elements = atoi(argv[3]);
    if (argc < 5) {
        for (unsigned int i = 0; i < benchmarks.size(); i++)
            arguments.push_back(benchmarks[i]);
    } else {
        for (int i = 4; i < argc; i++)
            arguments.push_back(argv[i]);
    }
    std::ofstream file;
    file.open("Benchmark.txt", std::ios::app);
    

    double start, end;
    for (int i = 0; i < 5; i++) {
        split_bcastMPI(start, end, 1, 10);
	Gather(start, end, 8);
    }

    /*for (int N = world_size; N <= pow(2, max_elements); N *= 2) {
        for (unsigned int i = 0; i < gather.size(); i++) {
            std::string benchmark = gather[i];
            if (world_rank == 0) {
                std::cout << "Benchmark: " << benchmark << std::endl;
                std::cout << "Elements: " << N << std::endl;
            }
            for (int k = 0; k < repeats; k++)
                executeBenchmark(benchmark, file, N, k);
            if (world_rank == 0)
                std::cout << std::endl;
        }
    }*/

    for (int j = min_elements; j <= max_elements; j += 2) {
        int N = std::pow(2, j);
        for (unsigned int i = 0; i < arguments.size(); i++) {
            std::string benchmark = arguments[i];
            if (world_rank == 0) {
                std::cout << "Benchmark: " << benchmark << std::endl;
                std::cout << "Elements: " << N << std::endl;
            }
            for (int k = 0; k < repeats; k++)
                executeBenchmark(benchmark, file, N, k);
            if (world_rank == 0)
                std::cout << std::endl;
        }
    }
    double end_time = MPI_Wtime();
    if (world_rank == 0) 
	std::cout << "Total time: " << end_time - start_time << std::endl; 
   
    file.close();

    MPI_Finalize();
};
