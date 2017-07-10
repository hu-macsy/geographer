/*****************************************************************************
 * This file is part of the Project ShizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Tobias Heuer <tobias.heuer@gmx.net>
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (C) 2016-2017 Michael Axtmann <michael.axtmann@kit.edu>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#ifndef QUICKSORT_HPP
#define QUICKSORT_HPP

#include <mpi.h>
#include "../io/SortingDatatype.hpp"
#include "../CollectiveOperations/CollectiveOperations.hpp"
#include "tb_splitter.hpp"
#include <vector>
#include <cassert>
#include <random>
#include <algorithm>
#include <iostream>
#include <cstring>

//Macros
#define V(X) std::cout << #X << "=" << X << endl
#define W(X) #X << "=" << X << ", "

/*
 * This struct represents a interval of data used in a Quicksort call
 */
template<typename T, typename COMM, typename REQ>
struct QSInterval {
    using coll = CollectiveOperations<COMM, REQ>;

    QSInterval(long long global_start, long long global_end, long long split_size,
            COMM comm, int seed, bool left_first, T *last_pivot_ptr = nullptr) :
    global_start(global_start), global_end(global_end), split_size(split_size),
    rank(0), comm(comm), communicate_first(left_first), has_last_pivot(false){
        if (last_pivot_ptr != nullptr) {
            has_last_pivot = true;
            last_pivot = *last_pivot_ptr;
        }
        coll::Comm_size(comm, &number_of_PEs);
        coll::Comm_rank(comm, &rank);

        this->seed = seed * 48271 % 2147483647;
        global_elements = global_end - global_start + 1;
        start_PE = global_start / split_size;
        end_PE = global_end / split_size;
        global_rank = start_PE + rank;

        //calculate local data        
        if (rank == 0)
            local_start = global_start % split_size;
        else
            local_start = 0;
        int end_PE = global_end / split_size - start_PE;
        if (rank == end_PE)
            local_end = global_end % split_size;
        else
            local_end = split_size - 1;
        local_elements = local_end - local_start + 1;
    }

    long long global_start, global_end, split_size, global_elements, local_start, local_end,
            local_elements, presum_small, presum_large, local_small_elements, 
            local_large_elements, global_small_elements, global_large_elements,
            bound1, split, bound2;
    int seed, number_of_PEs, rank, global_rank, start_PE, end_PE;
    T last_pivot; 
    COMM comm;
    bool communicate_first, has_last_pivot;
};

/*
 * This class represents the Quicksort algorithm
 */
template<typename T, typename COMM, typename REQ>
class QuickSort {
public:
    using coll = CollectiveOperations<COMM, REQ>;

    QuickSort(bool blocking, bool mpi_split, bool barriers, std::string pivot_samples) : blocking(blocking),
            mpi_split(mpi_split), barriers(barriers), pivot_samples(pivot_samples) {
        mpi_type = SortingDatatype<T>::getMPIDatatype();
    }

    ~QuickSort() {
    }

    /*
     * Sorts the data
     */
    void sort(T *first, long long split_size, MPI_Comm mpi_comm, int seed) {
        sort(first, split_size, mpi_comm, seed, std::greater<T>());
    }
    template<class Compare>
    void sort(T *first, long long split_size, MPI_Comm mpi_comm, int seed, Compare comp) {
        MPI_Barrier(mpi_comm);
        double total_start = getTime();
        
	data = first;
        this->split_size = split_size;
	buffer = new T[split_size];
        COMM comm;
        coll::Create_Comm_from_MPI(mpi_comm, &comm);
        int size, rank;
        coll::Comm_size(comm, &size);
        coll::Comm_rank(comm, &rank);
        QSInterval<T, COMM, REQ> ival(0, split_size * size - 1, split_size, comm,
                seed, true);

	generator.seed(seed);
        sample_generator.seed(seed + rank);

        quickSort(ival, comp);

        double start, end;
        if (barriers)
            coll::Barrier(comm);
        start = getTime();
        sortTwoPEIntervals(comp);
        end = getTime();
        t_sort_two = end - start;

        if (barriers)
            coll::Barrier(comm);
        start = getTime();
        sortLocalIntervals(comp);
        end = getTime();
        t_sort_local = end - start;

        double total_end = getTime();
        delete[] buffer;
        t_runtime = (total_end - total_start);
    }
/*
     * Get timers and their names
     */
    void getTimers(std::vector<std::string> &timer_names, std::vector<double> &timers,
	    std::vector<double> &max_timers, std::vector<double> &avg_timers,
	    std::vector<double> &min_timers, MPI_Comm comm) {
        int size;
        MPI_Comm_size(comm, &size);
        /*        for (int i = 0; i < 35; i++) {
                    exchange_timer_min.push_back(0.0);
                    exchange_timer_avg.push_back(0.0);
                    exchange_timer_max.push_back(0.0);
                    double time = 0.0;
                    if (t_vec_exchange.size() > i)
                        time = t_vec_exchange[i];
                    MPI_Reduce(&time, &exchange_timer_min[i], 1, MPI_DOUBLE, MPI_MIN, 0, comm);
                    MPI_Reduce(&time, &exchange_timer_avg[i], 1, MPI_DOUBLE, MPI_SUM, 0, comm);
                    exchange_timer_avg[i] /= size;
                    MPI_Reduce(&time, &exchange_timer_max[i], 1, MPI_DOUBLE, MPI_MAX, 0, comm);
                }
                exchange_split_times = exchange_times;
         */
        if (barriers) {
            timers.push_back(t_pivot);
            timer_names.push_back("pivot");
            timers.push_back(t_partition);
            timer_names.push_back("partition");
            timers.push_back(t_calculate);
            timer_names.push_back("calculate");
            timers.push_back(t_exchange);
            timer_names.push_back("exchange");
            timers.push_back(t_create_comms);
            timer_names.push_back("create_comms");
            timers.push_back(t_sort_two);
            timer_names.push_back("sort_two");
            timers.push_back(t_sort_local);
            timer_names.push_back("sort_local");
            double sum = 0.0;
            for (int i = 0; i < timers.size(); i++)
                sum += timers[i];
            timers.push_back(sum);
            timer_names.push_back("sum");
        }
        timers.push_back(t_runtime);
        timer_names.push_back("runtime");
        timers.push_back(depth);
        timer_names.push_back("depth");

        for (int i = 0; i < timers.size(); i++) {
            double time = 0.0;
            MPI_Reduce(&timers[i], &time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            max_timers.push_back(time);
            MPI_Reduce(&timers[i], &time, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
            min_timers.push_back(time);
            MPI_Reduce(&timers[i], &time, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
            avg_timers.push_back(time / size);
        }
    }

private:

    /*
     * Execute the Quicksort algorithm on the given QSInterval
     */
    template<class Compare>
    void quickSort(QSInterval<T, COMM, REQ> &ival, Compare comp) {
        //Check if recursion should be ended        
        if (ival.number_of_PEs == 2) {
            addTwoPEInterval(ival);
            return;
        }
        if (ival.number_of_PEs == 1) {
            addLocalInterval(ival);
            return;
        }
        depth++;

        T pivot;
        int split_idx;
        double t_start, t_end;
        t_start = startTime(ival.comm);
        getPivot(ival, pivot, split_idx, comp);
        t_end = getTime();
        t_pivot += (t_end - t_start);
        
        //check if pivot is the same as the last one
        t_start = startTime(ival.comm);
        int bound1, bound2;        
        partitionData(ival, pivot, split_idx, &bound1, &bound2, comp);
        t_end = getTime();
        t_partition += (t_end - t_start);

        t_start = startTime(ival.comm);
        calculateExchangeData(ival, bound1, split_idx, bound2);
        t_end = getTime();
        t_calculate += (t_end - t_start);
	
        t_start = startTime(ival.comm);
        exchangeData(ival);
        t_end = getTime();
        t_exchange += (t_end - t_start);
        t_vec_exchange.push_back(t_end - t_start);

        long long start = ival.global_start;
        long long mid = ival.global_start + ival.global_small_elements;
        long long end = ival.global_end;

        COMM left, right;
        t_start = startTime(ival.comm);
        createNewCommunicators(ival, mid, &left, &right);
        t_end = getTime();
        t_create_comms += (t_end - t_start);

        QSInterval<T, COMM, REQ> ival_left(start, mid - 1, ival.split_size, 
                left, ival.seed, true, &pivot);
        QSInterval<T, COMM, REQ> ival_right(mid, end, ival.split_size, 
                right, ival.seed + 1, false, &pivot);

        bool sort_left = false, sort_right = false;
        if (ival.global_rank <= ival_left.end_PE)
            sort_left = true;
        if (ival.global_rank >= ival_right.start_PE)
            sort_right = true;

        if (sort_left && sort_right) {
            shizophrenicQuickSort(ival_left, ival_right, comp);
        } else if (sort_left) {
            quickSort(ival_left, comp);
        } else if (sort_right) {
            quickSort(ival_right, comp);
        }
    }

    /*
     * Execute the Quicksort algorithm as schizophrenic PE
     */
    template<class Compare>
    void shizophrenicQuickSort(QSInterval<T, COMM, REQ> &ival_left, QSInterval<T,
                               COMM, REQ> &ival_right, Compare comp) {
        //Check if recursion should be ended        
        if ((ival_left.number_of_PEs == 1) && (ival_right.number_of_PEs == 1)) {
            addLocalInterval(ival_right);
            return;
        } else if (ival_left.number_of_PEs == 1) {
            addLocalInterval(ival_left);
            quickSort(ival_right, comp);
            return;
        } else if (ival_right.number_of_PEs == 1) {
            addLocalInterval(ival_right);
            quickSort(ival_left, comp);
            return;
        }
        if (ival_left.number_of_PEs == 2) {
            addTwoPEInterval(ival_left);
            quickSort(ival_right, comp);
            return;
        }
        if (ival_right.number_of_PEs == 2) {
            addTwoPEInterval(ival_right);
            quickSort(ival_left, comp);
            return;
        }

        depth++;

        T pivot_left, pivot_right;
        int split_idx_left, split_idx_right;
        double t_start, t_end;
        
        t_start = startTimeSchizo(ival_left.comm, ival_right.comm);
        getPivotSchizo(ival_left, ival_right, pivot_left, pivot_right,
                       split_idx_left, split_idx_right, comp);
        t_end = getTime();
        t_pivot += (t_end - t_start);


        t_start = startTimeSchizo(ival_left.comm, ival_right.comm);
        
        int bound1_left, bound2_left, bound1_right, bound2_right;        
                
        partitionData(ival_left, pivot_left, split_idx_left, &bound1_left, 
                      &bound2_left, comp);          
        partitionData(ival_right, pivot_right, split_idx_right, &bound1_right, 
                      &bound2_right, comp);  
        t_end = getTime();
        t_partition += (t_end - t_start);

        t_start = startTimeSchizo(ival_left.comm, ival_right.comm);
        calculateExchangeDataSchizo(ival_left, ival_right, bound1_left, split_idx_left,
                bound2_left, bound1_right, split_idx_right, bound2_right);
        t_end = getTime();
        t_calculate += (t_end - t_start);

        t_start = startTimeSchizo(ival_left.comm, ival_right.comm);
        exchangeDataSchizo(ival_left, ival_right);
        t_end = getTime();
        t_exchange += (t_end - t_start);
        t_vec_exchange.push_back(t_end - t_start);


        //indices for interval starts and ends
        long long start_left = ival_left.global_start;
        long long mid_left = ival_left.global_start + ival_left.global_small_elements;
        long long end_left = ival_left.global_end;
        long long start_right = ival_right.global_start;
        long long mid_right = ival_right.global_start + ival_right.global_small_elements;
        long long end_right = ival_right.global_end;

        COMM left1, right1, left2, right2;

        t_start = startTimeSchizo(ival_left.comm, ival_right.comm);
        createNewCommunicatorsSchizo(ival_left, ival_right, mid_left, mid_right,
                &left1, &right1, &left2, &right2);
        t_end = getTime();
        t_create_comms += (t_end - t_start);

        QSInterval<T, COMM, REQ> ival_left_left(start_left, mid_left - 1, split_size,
                left1, ival_left.seed, true, &pivot_left);
        QSInterval<T, COMM, REQ> ival_left_right(mid_left, end_left, split_size,
                right1, ival_left.seed + 1, false, &pivot_left);
        QSInterval<T, COMM, REQ> ival_right_left(start_right, mid_right - 1, split_size,
                left2, ival_right.seed, true, &pivot_right);
        QSInterval<T, COMM, REQ> ival_right_right(mid_right, end_right, split_size,
                right2, ival_right.seed + 1, false, &pivot_right);

        bool sort_left = false, sort_right = false;
        QSInterval<T, COMM, REQ> *left_i, *right_i;
        //Calculate new left interval and if it need to be sorted
        if (ival_left.global_rank == ival_left_right.start_PE) {
            addLocalInterval(ival_left_right);
            left_i = &ival_left_left;
            if (ival_left_left.end_PE == ival_left.global_rank)
                sort_left = true;
        } else {
            left_i = &ival_left_right;
            sort_left = true;
        }
        //Calculate new right interval and if it need to be sorted        
        if (ival_right_left.end_PE == ival_right.global_rank) {
            addLocalInterval(ival_right_left);
            right_i = &ival_right_right;
            if (ival_right_right.start_PE == ival_right.global_rank)
                sort_right = true;
        } else {
            right_i = &ival_right_left;
            sort_right = true;
        }

        //Recursive call to quicksort/shizophrenicQuicksort
        if (sort_left && sort_right) {
            shizophrenicQuickSort(*left_i, *right_i, comp);
        } else if (sort_left) {
            quickSort(*left_i, comp);
        } else if (sort_right) {
            quickSort(*right_i, comp);
        }
    }

    double startTime(COMM &comm) {
        if (!barriers)
            return getTime();
        if (coll::implementsIbcast()) {
            REQ req;
            coll::Ibarrier(comm, &req);
            coll::Wait(&req, MPI_STATUS_IGNORE);
        } else
            coll::Barrier(comm);
        return getTime();
    }

    double startTimeSchizo(COMM &left_comm, COMM &right_comm) {
        if (!barriers)
            return getTime();
        if (coll::implementsIbcast()) {
            REQ req[2];
            coll::Ibarrier(left_comm, &req[0]);
            coll::Ibarrier(right_comm, &req[1]);
            coll::Waitall(2, req, MPI_STATUS_IGNORE);
        } else {
            coll::Barrier(left_comm);
            coll::Barrier(right_comm);
        }
        return getTime();
    }
    
    /*
     * Selects a random element from the interval as the pivot
     */
    template<class Compare>
    void getPivot(QSInterval<T, COMM, REQ> const &ival, T &pivot, int &split_idx, Compare comp) {
        int sample_count = getSampleCount(ival);

        //randomly pick samples from local data
        std::vector<TbSplitter<T>> samples;
        getLocalSamples(ival, sample_count, samples, comp);        

        auto tb_splitter_comp = [comp](const TbSplitter<T>& first,
                                     const TbSplitter<T>& second) {
            return first.compare(second, comp);
        };
        
        //Merge function used in the gather
        std::function<void (void*, void*, void*) > merge = [tb_splitter_comp](
            void* start, void* mid, void* end) {
            std::inplace_merge(static_cast<TbSplitter<T>*> (start), 
                               static_cast<TbSplitter<T>*> (mid),
                               static_cast<TbSplitter<T>*> (end),
                               tb_splitter_comp);
        };

        //Gather the samples to rank 0
        TbSplitter<T>* all_samples;
        if (ival.rank == 0)
            all_samples = new TbSplitter<T>[sample_count];

        MPI_Datatype splitter_type = TbSplitter<T>::MpiType(mpi_type);  
                
        REQ req_gather;
        coll::Igatherm(&samples[0], samples.size(), splitter_type, all_samples, sample_count,
                0, merge, ival.comm, &req_gather, PIVOT_GATHER);
        coll::Wait(&req_gather, MPI_STATUS_IGNORE);
        
        TbSplitter<T> splitter;
        if (ival.rank == 0)
	    splitter = all_samples[sample_count / 2];
	 
        //Broadcast the pivot from rank 0
        REQ req_bcast;
        coll::Ibcast(&splitter, 1, splitter_type, 0, ival.comm, &req_bcast, PIVOT_BCAST);
        coll::Wait(&req_bcast, MPI_STATUS_IGNORE);
        
        pivot = splitter.Splitter();
        int splitter_PE = splitter.GID() / ival.split_size;
        if (ival.global_rank < splitter_PE)
            split_idx = ival.local_end + 1;
        else if (ival.global_rank > splitter_PE)
            split_idx = ival.local_start;
        else {
            split_idx = splitter.GID() % ival.split_size + 1;
        }
        
        if (ival.rank == 0)
            delete[] all_samples;
    }
    
    /*
     * Select a random element as the pivot from both intervals
     */
    template<class Compare>
    void getPivotSchizo(QSInterval<T, COMM, REQ> const &ival_left,
                        QSInterval<T, COMM, REQ> const &ival_right, T &pivot_left,
                        T &pivot_right, int &split_idx_left, int &split_idx_right,
                        Compare comp) {
        //blocking version
        if (blocking) {
            if (ival_left.communicate_first) {
                getPivot(ival_left, pivot_left, split_idx_left, comp);
                getPivot(ival_right, pivot_right, split_idx_right, comp);
            } else {                
                getPivot(ival_right, pivot_right, split_idx_right, comp);
                getPivot(ival_left, pivot_left, split_idx_left, comp);
            }
            return;
        }
        
        int sample_count_left = getSampleCount(ival_left);
        int sample_count_right = getSampleCount(ival_right);

        //Randomly pick samples from local data
        std::vector<TbSplitter<T>> samples_left, samples_right;
        getLocalSamples(ival_left, sample_count_left, samples_left, comp);
        getLocalSamples(ival_right, sample_count_right, samples_right, comp);

        //Merge function used in the gather

        auto tb_splitter_comp = [comp](const TbSplitter<T>& first,
                                     const TbSplitter<T>& second) {
            return first.compare(second, comp);
        };
        std::function<void (void*, void*, void*) > merge = [tb_splitter_comp](
            void* start, void* mid, void* end) {
            std::inplace_merge(static_cast<TbSplitter<T>*> (start), 
                               static_cast<TbSplitter<T>*> (mid),
                               static_cast<TbSplitter<T>*> (end), tb_splitter_comp);
        };
        
        TbSplitter<T> splitter_left, splitter_right;
        MPI_Datatype splitter_type = TbSplitter<T>::MpiType(mpi_type); 
        
        //Gather the samples
        TbSplitter<T>* all_samples = new TbSplitter<T>[sample_count_right];
        REQ req_gather[2];
        coll::Igatherm(&samples_left[0], samples_left.size(), splitter_type, nullptr, 
                sample_count_left, 0, merge, ival_left.comm, &req_gather[0], PIVOT_GATHER);
        coll::Igatherm(&samples_right[0], samples_right.size(), splitter_type, all_samples, 
                sample_count_right, 0, merge, ival_right.comm, &req_gather[1], PIVOT_GATHER);
        coll::Waitall(2, req_gather, MPI_STATUSES_IGNORE);

        //Determine pivot on right interval 
        splitter_right = all_samples[sample_count_right / 2];

        //Broadcast the pivots
        REQ req_bcast[2];
        coll::Ibcast(&splitter_left, 1, splitter_type, 0, ival_left.comm, &req_bcast[0], PIVOT_BCAST);
        coll::Ibcast(&splitter_right, 1, splitter_type, 0, ival_right.comm, &req_bcast[1], PIVOT_BCAST);
        coll::Waitall(2, req_bcast, MPI_STATUSES_IGNORE);
        
        pivot_left = splitter_left.Splitter();
        pivot_right = splitter_right.Splitter();
        
        int splitter_PE_left = splitter_left.GID() / ival_left.split_size;
        int splitter_PE_right = splitter_right.GID() / ival_right.split_size;
        
        if (ival_left.global_rank < splitter_PE_left)
            split_idx_left = ival_left.local_end + 1;
        else if (ival_right.global_rank > splitter_PE_left)
            split_idx_left = ival_left.local_start;
        else {
            split_idx_left = splitter_left.GID() % ival_left.split_size + 1;
        }
        if (ival_right.global_rank < splitter_PE_right)
            split_idx_right = ival_right.local_end + 1;
        else if (ival_right.global_rank > splitter_PE_right)
            split_idx_right = ival_right.local_start;
        else {
            split_idx_right = splitter_right.GID() % ival_right.split_size + 1;
        }
        
        delete[] all_samples;
    }
    
    /*
     * Get the number of samples
     */
    int getSampleCount(QSInterval<T, COMM, REQ> const &ival) {
        int sample_count;
        if (pivot_samples == "9")
            sample_count = 9;
        else if (pivot_samples == "16_log_p")
            sample_count = 16 * log2(static_cast<double>(ival.number_of_PEs));
        else if (pivot_samples == "32_log_p")
            sample_count = 32 * log2(static_cast<double>(ival.number_of_PEs));
        else if (pivot_samples == "64_log_p")
            sample_count = 64 * log2(static_cast<double>(ival.number_of_PEs));
        else if (pivot_samples == "128_log_p")
            sample_count = 128 * log2(static_cast<double>(ival.number_of_PEs));
        else if (pivot_samples == "1/400_s")
            sample_count = ival.split_size / 400;
        else if (pivot_samples == "1/200_s")
            sample_count = ival.split_size / 200;
        else if (pivot_samples == "1/100_s")
            sample_count = ival.split_size / 100;
        else if (pivot_samples == "1/50_s")
            sample_count = ival.split_size / 50;
        else if (pivot_samples == "1/25_s")
            sample_count = ival.split_size / 25;
        else if (pivot_samples == "8_50") {
	    int k_1 = ival.split_size / 50;
	    int k_2 = 8 * log2(static_cast<double>(ival.number_of_PEs));
            sample_count = std::max(k_1, k_2);
        } else if (pivot_samples == "16_50") {
	    int k_1 = ival.split_size / 50;
	    int k_2 = 16 * log2(static_cast<double>(ival.number_of_PEs));
            sample_count = std::max(k_1, k_2);
        } else if (pivot_samples == "32_50") {
	    int k_1 = ival.split_size / 50;
	    int k_2 = 32 * log2(static_cast<double>(ival.number_of_PEs));
            sample_count = std::max(k_1, k_2);
	} else {
            std::cout << pivot_samples << std::endl;
            exit(55);
        }
            
        if (sample_count % 2 == 0)
            sample_count++;
        sample_count = std::max(sample_count, 9);
        
        assert(sample_count >= 9);
        assert(sample_count % 2 == 1);
        
        return sample_count;
    }
    
    /*
     * Determine how much samples need to be send and pick them randomly
     */
    template<class Compare>
    void getLocalSamples(QSInterval<T, COMM, REQ> const &ival, int total_samples,
                         std::vector<TbSplitter<T>> &sample_vec, Compare comp) {
        int max_height = ceil(log2(static_cast<double>(ival.number_of_PEs)));
        int own_height = 0;
        for (int i = 0; ((ival.rank >> i) % 2 == 0) && (i < max_height); i++)
            own_height++;

        int missing_on_first = ival.global_start - ival.start_PE * split_size;
        int missing_on_last = (ival.end_PE + 1) * split_size - 1 - ival.global_end;
        int first_PE = 0;
        int last_PE = ival.number_of_PEs - 1;
        int samples = total_samples;
        generator.seed(ival.seed);
        
        for (int height = max_height; height > 0; height--) {
            if (first_PE + pow(2, height - 1) > last_PE) {
                //right subtree is empty
            } else {
                int left_size = pow(2, height - 1);
                int right_size = last_PE - first_PE + 1 - left_size;
		assert(left_size > 0);
		assert(right_size > 0);
		assert(left_size+right_size == last_PE - first_PE + 1);
                long long left_elements = left_size * ival.split_size;
                long long right_elements = right_size * ival.split_size;
                assert(left_elements >= 0);
                assert(right_elements >= 0);
                if (first_PE == 0)
                    left_elements -= missing_on_first;
                if (last_PE == ival.number_of_PEs - 1)
                    right_elements -= missing_on_last;
                
                double percentage_left = static_cast<double>(left_elements) / static_cast<double>(left_elements + right_elements);
                assert(percentage_left > 0);
		std::binomial_distribution<int> binom_distr(samples, percentage_left);
                int samples_left = binom_distr(generator);
                int samples_right = samples - samples_left;

                int mid_PE = first_PE + pow(2, height - 1);

                if (ival.rank < mid_PE) {
                    //left side
                    last_PE = mid_PE - 1;
                    samples = samples_left;
                } else {
                    //right side
                    first_PE = mid_PE;
                    samples = samples_right;
                }
            }
        }
        std::uniform_int_distribution<long long> distr(ival.local_start, ival.local_end);
        for (int i = 0; i < samples; i++) {
            long long index = distr(sample_generator);
            long long global_index = ival.global_rank * ival.split_size + index;
            sample_vec.push_back(TbSplitter<T>(data[index], global_index));
        }

        auto tb_splitter_comp = [comp](const TbSplitter<T>& first,
                                     const TbSplitter<T>& second) {
            return first.compare(second, comp);
        };
        std::sort(sample_vec.begin(), sample_vec.end(), tb_splitter_comp);
        return;
    }
    
    /*
     * Partitions the data and returns the index of the first element of the right partition
     */
    template<class Compare>
    void partitionData(QSInterval<T, COMM, REQ> const &ival, T pivot, int less_idx,
                       int *index1, int  *index2, Compare comp) {
        int start1 = ival.local_start, end1 = less_idx,
                start2 = less_idx, end2 = ival.local_end + 1;
        *index1 = partitionData_(pivot, start1, end1, true, comp);
        *index2 = partitionData_(pivot, start2, end2, false, comp);	
    }

    template<class Compare>
    int partitionData_(T pivot, int start, int end, bool less_equal, Compare comp) {
        T* bound;
	if (less_equal) {
            bound = std::partition(data + start, data + end,
                                   [pivot, comp](T x){return !comp(pivot, x)/*x <= pivot*/;});
	} else {
            bound = std::partition(data + start, data + end,
                                   [pivot, comp](T x){return comp(x, pivot);});
	}
	return bound - data;
    }

    /*
     * Prefix sum of small/large elements and broadcast of global small/large elements
     */
    void calculateExchangeData(QSInterval<T, COMM, REQ> &ival, int bound1,
            int split, int bound2) {        
        ival.bound1 = bound1;
        ival.bound2 = bound2;
        ival.split = split;
        int small_elements = (bound1 - ival.local_start) + (bound2 - split);
        ival.local_small_elements = small_elements;
        ival.local_large_elements = ival.local_elements - ival.local_small_elements;
        REQ request;
        if (coll::implementsIscanAndBcast()) {
            coll::IscanAndBcast(&(ival.local_small_elements), &(ival.presum_small),
                    &(ival.global_small_elements), 1, MPI_LONG_LONG, MPI_SUM, ival.comm,
                    &request, CALC_EXCH);
            coll::Wait(&request, MPI_STATUS_IGNORE);
        } else if (coll::implementsIbcast()) {
            coll::Iscan(&(ival.local_small_elements), &(ival.presum_small), 1,
                    MPI_LONG_LONG, MPI_SUM, ival.comm, &request);
            coll::Wait(&request, MPI_STATUS_IGNORE);
            if (ival.rank == ival.number_of_PEs - 1) {
                ival.global_small_elements = ival.presum_small;
            }
            coll::Ibcast(&(ival.global_small_elements), 1, MPI_LONG_LONG, ival.number_of_PEs - 1,
                    ival.comm, &request);
            coll::Wait(&request, MPI_STATUS_IGNORE);
        } else {
            coll::Scan(&(ival.local_small_elements), &(ival.presum_small), 1,
                    MPI_LONG_LONG, MPI_SUM, ival.comm);
            if (ival.rank == ival.number_of_PEs - 1) {
                ival.global_small_elements = ival.presum_small;
            }
            coll::Bcast(&(ival.global_small_elements), 1, MPI_LONG_LONG,
                    ival.number_of_PEs - 1, ival.comm);
        }

        ival.presum_small -= ival.local_small_elements;
        long long local_start;
        if (ival.rank == 0)
            local_start = ival.local_start + ival.start_PE * split_size;
        else
            local_start = (ival.rank + ival.start_PE) * split_size;
        ival.presum_large = local_start - ival.global_start - ival.presum_small;
        ival.global_large_elements = ival.global_elements - ival.global_small_elements;
    }

    void calculateExchangeDataSchizo(QSInterval<T, COMM, REQ> &ival_left,
            QSInterval<T, COMM, REQ> &ival_right, int bound1_left, int split_left, 
            int bound2_left, int bound1_right, int split_right, int bound2_right) {
        // version without Ibcast
        if (!coll::implementsIbcast() || blocking) {
            if (ival_left.communicate_first) {
                calculateExchangeData(ival_left, bound1_left, split_left, bound2_left);
                calculateExchangeData(ival_right, bound1_right, split_right, bound2_right);
            } else {
                calculateExchangeData(ival_right, bound1_right, split_right, bound2_right);  
                calculateExchangeData(ival_left, bound1_left, split_left, bound2_left);              
            }
            return;
        }
           
        ival_left.bound1 = bound1_left;
        ival_left.bound2 = bound2_left;
        ival_left.split = split_left;
        ival_left.local_small_elements = (bound1_left - ival_left.local_start) 
            + (bound2_left - split_left);
        ival_left.local_large_elements = ival_left.local_elements - ival_left.local_small_elements;
        
        ival_right.bound1 = bound1_right;
        ival_right.bound2 = bound2_right;
        ival_right.split = split_right;
        ival_right.local_small_elements = (bound1_right - ival_right.local_start) 
            + (bound2_right - split_right);
        ival_right.local_large_elements = ival_right.local_elements - ival_right.local_small_elements;
        
        REQ requests[2];
//        if (coll::implementsIscanAndBcast()) {
            coll::IscanAndBcast(&(ival_left.local_small_elements), &(ival_left.presum_small),
                    &(ival_left.global_small_elements), 1, MPI_LONG_LONG, MPI_SUM, ival_left.comm,
                    &requests[0], CALC_EXCH);
            coll::IscanAndBcast(&(ival_right.local_small_elements), &(ival_right.presum_small),
                    &(ival_right.global_small_elements), 1, MPI_LONG_LONG, MPI_SUM, ival_right.comm,
                    &requests[1], CALC_EXCH);
            coll::Waitall(2, requests, MPI_STATUSES_IGNORE);
//        } else {
//            coll::Iscan(&(ival_left.local_small_elements), &(ival_left.presum_small), 1,
//                    MPI_LONG_LONG, MPI_SUM, ival_left.comm, &requests[0]);
//            coll::Iscan(&(ival_right.local_small_elements), &(ival_right.presum_small), 1,
//                    MPI_LONG_LONG, MPI_SUM, ival_right.comm, &requests[1]);
//            coll::Waitall(2, requests, MPI_STATUSES_IGNORE);
//
//            ival_left.global_small_elements = ival_left.presum_small;
//
//            coll::Ibcast(&(ival_left.global_small_elements), 1, MPI_LONG_LONG,
//                    ival_left.number_of_PEs - 1, ival_left.comm, &requests[0]);
//            coll::Ibcast(&(ival_right.global_small_elements), 1, MPI_LONG_LONG,
//                    ival_right.number_of_PEs - 1, ival_right.comm, &requests[1]);
//            coll::Waitall(2, &requests[0], MPI_STATUSES_IGNORE);
//        }

        ival_left.presum_small -= ival_left.local_small_elements;
        long long local_start_left;
        if (ival_left.rank == 0)
            local_start_left = ival_left.local_start + ival_left.start_PE * split_size;
        else
            local_start_left = (ival_left.rank + ival_left.start_PE) * split_size;
        ival_left.presum_large = local_start_left - ival_left.global_start - ival_left.presum_small;
        ival_left.global_large_elements = ival_left.global_elements - ival_left.global_small_elements;

        ival_right.presum_small -= ival_right.local_small_elements;
        long long local_start_right;
        if (ival_right.rank == 0)
            local_start_right = ival_right.local_start + ival_right.start_PE * split_size;
        else
            local_start_right = (ival_right.rank + ival_right.start_PE) * split_size;
        ival_right.presum_large = local_start_right - ival_right.global_start - ival_right.presum_small;
        ival_right.global_large_elements = ival_right.global_elements - ival_right.global_small_elements;
    }

    /*
     * Exchange the data with other PEs
     */
    void exchangeData(QSInterval<T, COMM, REQ> &ival) {
        long long recv_small = 0, recv_large = 0, recv_count_small = 0, recv_count_large = 0;
        std::vector<std::unique_ptr<REQ>> requests;
        
        //copy current data (that will be send) into buffer
        copyDataToBuffer(ival);

        //calculate how much data need to be send and received and start non-blocking sends
        exchangeData_calculateAndSend(ival, requests, recv_small, recv_large,
                recv_count_small, recv_count_large);
        
        //receive data
        while ((recv_count_small < recv_small) || (recv_count_large < recv_large)) {
            receiveData(ival.comm, requests,
                    data + ival.local_start + recv_count_small,
                    recv_count_small, recv_small, EXCHANGE_SMALL);
            receiveData(ival.comm, requests,
                    data + ival.local_start + recv_small + recv_count_large,
                    recv_count_large, recv_large, EXCHANGE_LARGE);
        }

        //wait until all sends and receives have finished
        Waitall_vector(requests);
    }

    void copyDataToBuffer(QSInterval<T, COMM, REQ> &ival) {        
        int copy = ival.bound1 - ival.local_start;
        std::memcpy(buffer + ival.local_start, data + ival.local_start, copy * sizeof(T));
        
        int small_right = ival.bound2 - ival.split;
        copy = ival.split - ival.bound1;
        std::memcpy(buffer + ival.bound1 + small_right, data + ival.bound1, copy * sizeof(T));
        
        copy = ival.bound2 - ival.split;
        std::memcpy(buffer + ival.bound1, data + ival.split, copy * sizeof(T));
        
        copy = ival.local_end + 1 - ival.bound2;
        std::memcpy(buffer + ival.bound2, data + ival.bound2, copy * sizeof(T));
    }
    
    /*
     * Exchange the data with other PEs on both intervals simultaneously
     */
    void exchangeDataSchizo(QSInterval<T, COMM, REQ> &left, QSInterval<T, COMM, REQ> &right) {
        long long recv_small_l = 0, recv_large_l = 0, recv_count_small_l = 0, recv_count_large_l = 0;
        long long recv_small_r = 0, recv_large_r = 0, recv_count_small_r = 0, recv_count_large_r = 0;
        std::vector<std::unique_ptr<REQ>> requests;
        
        //copy current data (that will be send) into buffer        
        copyDataToBuffer(left);
        copyDataToBuffer(right);
        
        //calculate how much data need to be send and received and start non-blocking sends
        exchangeData_calculateAndSend(left, requests, recv_small_l, recv_large_l,
                recv_count_small_l, recv_count_large_l);
        exchangeData_calculateAndSend(right, requests, recv_small_r, recv_large_r,
                recv_count_small_r, recv_count_large_r);

        //receive data
        while ((recv_count_small_l < recv_small_l) || (recv_count_large_l < recv_large_l)
                || (recv_count_small_r < recv_small_r) || (recv_count_large_r < recv_large_r)) {
            receiveData(left.comm, requests,
                    data + left.local_start + recv_count_small_l,
                    recv_count_small_l, recv_small_l, EXCHANGE_SMALL);
            receiveData(left.comm, requests,
                    data + left.local_start + recv_small_l + recv_count_large_l,
                    recv_count_large_l, recv_large_l, EXCHANGE_LARGE);
            receiveData(right.comm, requests,
                    data + right.local_start + recv_count_small_r,
                    recv_count_small_r, recv_small_r, EXCHANGE_SMALL);
            receiveData(right.comm, requests,
                    data + right.local_start + recv_small_r + recv_count_large_r,
                    recv_count_large_r, recv_large_r, EXCHANGE_LARGE);
        }

        //wait until all sends and receives have finished
        Waitall_vector(requests);
    }

    void Waitall_vector(std::vector<std::unique_ptr<REQ>> &requests) {
        int flag = 0;
        while(flag == 0) {
            flag = 1;
            for (size_t i = 0; i < requests.size(); i++) {
                int tmp_flag;
                coll::Test(requests[i].get(), &tmp_flag, MPI_STATUS_IGNORE);
                if (tmp_flag == 0)
                    flag = 0;
            }
        }
    }
    
    /*
     * calculate how much data need to be send and received and start non-blocking sends
     */
    void exchangeData_calculateAndSend(QSInterval<T, COMM, REQ> &ival,
            std::vector<std::unique_ptr<REQ>> &requests, long long &recv_small, 
            long long &recv_large, long long &recv_count_small, long long &recv_count_large) {
        int small_PE_1 = 0, small_PE_2 = 0, large_PE_1 = 0, large_PE_2 = 0;
        long long send_small_1, send_small_2, send_large_1, send_large_2;

        calculateSendCount(ival.global_start + ival.presum_small, ival.start_PE,
                ival.local_small_elements,
                small_PE_1, small_PE_2, send_small_1, send_small_2);

        calculateSendCount(ival.global_start + ival.global_small_elements + ival.presum_large,
                ival.start_PE, ival.local_large_elements,
                large_PE_1, large_PE_2, send_large_1, send_large_2);

        //calculate how much small and large data need to be received
        int small_end_PE = (ival.global_start + ival.global_small_elements - 1)
                / split_size - ival.start_PE;
        int large_start_PE = (ival.global_start + ival.global_small_elements)
                / split_size - ival.start_PE;
        if (large_start_PE > ival.rank) {
            recv_small = ival.local_elements;
            recv_large = 0;
        } else if (small_end_PE < ival.rank) {
            recv_small = 0;
            recv_large = ival.local_elements;
        } else {
            recv_small = (ival.global_start + ival.global_small_elements) % ival.split_size
                    - ival.local_start;
            recv_large = ival.local_elements - recv_small;
        }
    
        long long index_small_1 = ival.local_start;
        long long index_small_2 = index_small_1 + send_small_1;
        long long index_large_1 = ival.local_start + ival.local_small_elements;
        long long index_large_2 = index_large_1 + send_large_1;
        
        //send small data        
        sendData(send_small_1, recv_count_small, index_small_1, index_small_1,
                small_PE_1, EXCHANGE_SMALL, ival, requests);
        sendData(send_small_2, recv_count_small, ival.local_start, index_small_2,
                small_PE_2, EXCHANGE_SMALL, ival, requests);

        //send large data
        sendData(send_large_1, recv_count_large, ival.local_start + recv_small,
                index_large_1, large_PE_1, EXCHANGE_LARGE, ival, requests);
        sendData(send_large_2, recv_count_large, ival.local_start + recv_small,
                index_large_2, large_PE_2, EXCHANGE_LARGE, ival, requests);

    }

    /*
     * Calculate where and how much data need to be send
     */
    void calculateSendCount(long long start_index, int start_PE, long long local_elements,
            int &PE_1, int &PE_2, long long &send_1, long long &send_2) {        
        long long start = start_index;
        long long end = start + local_elements - 1;
        
        PE_1 = static_cast<int>(start / split_size) - start_PE;
        PE_2 = static_cast<int>(end / split_size) - start_PE;
        
        if (PE_1 >= PE_2) {
            send_1 = local_elements;
            send_2 = 0;
        } else {
            send_1 = split_size - (start % split_size);
            send_2 = local_elements - send_1;
        }

    }

    /*
     * Starts a non-blocking receive if data can be received
     */
    void receiveData(COMM const &comm, std::vector<std::unique_ptr<REQ>> &requests,
            void *recvbuf, long long &recv_count, long long recv_total, int tag) {
        if (recv_count < recv_total) {
            int ready;
            MPI_Status status;
            coll::Iprobe(MPI_ANY_SOURCE, tag, comm, &ready, &status);
            if (ready) {
                int count;
                MPI_Get_count(&status, mpi_type, &count);
                assert(recv_count + count <= recv_total);
                int source = coll::getSource(comm, status);
                requests.push_back(std::unique_ptr<REQ>(new REQ));
                coll::Irecv(recvbuf, count, mpi_type, source,
                        tag, comm, requests.back().get());
                recv_count += count;
            }
        }
    }

    /*
     * Sends the data to the target or writes in local buffer
     */
    void sendData(long long send_count, long long &recv_count, long long recv_index,
            long long send_index, int target, int tag, QSInterval<T, COMM, REQ> const &ival,
            std::vector<std::unique_ptr<REQ>> &requests) {
        if (send_count > 0) {
            if (target == ival.rank) {
                recv_count += send_count;
                std::memcpy(data + recv_index, buffer + send_index, send_count * sizeof(T));
            } else {
                requests.push_back(std::unique_ptr<REQ>(new REQ));
                coll::Isend(buffer + send_index, send_count, mpi_type, target,
                        tag, ival.comm, requests.back().get());
            }
        }
    }

    /*
     * Splits the communicator into two new, left and right
     */
    void createNewCommunicators(QSInterval<T, COMM, REQ> &ival, long long middle, 
            COMM *left, COMM *right) {
        int end_left = (middle - 1) / split_size - ival.start_PE;
        int start_right = middle / split_size - ival.start_PE;

        coll::createNewCommunicators(ival.comm, end_left, start_right, 
                left, right, mpi_split);

        coll::freeComm(&ival.comm);
    }
    
    void createNewCommunicatorsSchizo(QSInterval<T, COMM, REQ> &ival_left,
            QSInterval<T, COMM, REQ> &ival_right, long long middle_left,
            long long middle_right, COMM *left_left, COMM *right_left,
            COMM *left_right, COMM *right_right) {
        if (ival_left.communicate_first) {
            createNewCommunicators(ival_left, middle_left, left_left, right_left);
            createNewCommunicators(ival_right, middle_right, left_right, right_right);
        } else {
            createNewCommunicators(ival_right, middle_right, left_right, right_right);  
            createNewCommunicators(ival_left, middle_left, left_left, right_left);
        }
    }
    

    /*
     * sort interval with only two PEs sequential to terminate recursion
     */
    template<class Compare>
    void sortOnTwoPEs(QSInterval<T, COMM, REQ> &ival, Compare comp) {
        long long recv_elements = ival.global_elements - ival.local_elements;
        T* tmp_buffer = new T[ival.global_elements];
        REQ requests[2];
        if (ival.rank == 0) {
            coll::Isend(data + ival.local_start, ival.local_elements, mpi_type, 
                    1, TWO_PE, ival.comm, &requests[0]);
            coll::Irecv(tmp_buffer, recv_elements, mpi_type, 1, TWO_PE, ival.comm, 
                    &requests[1]);
        } else {
            assert(ival.rank == 1);
            coll::Isend(data + ival.local_start, ival.local_elements, mpi_type, 
                    0, TWO_PE, ival.comm, &requests[0]);
            coll::Irecv(tmp_buffer, recv_elements, mpi_type, 0, TWO_PE, ival.comm, 
                    &requests[1]);
        }        

        std::memcpy(tmp_buffer + recv_elements, data + ival.local_start,
                ival.local_elements * sizeof(T));
        
        coll::Waitall(2, requests, MPI_STATUSES_IGNORE);
        
        
        
        if (ival.rank == 0) {
            T* nth_element = tmp_buffer + ival.local_elements - 1;
            std::nth_element(tmp_buffer, nth_element, tmp_buffer + ival.global_elements, comp); 
            std::sort(tmp_buffer, nth_element + 1, comp);
            std::memcpy(data + ival.local_start, tmp_buffer, 
                    ival.local_elements * sizeof(T));         
        } else {
            T* nth_element = tmp_buffer + recv_elements;
            std::nth_element(tmp_buffer, nth_element, tmp_buffer + ival.global_elements, comp); 
            std::sort(nth_element, tmp_buffer + ival.global_elements, comp);          
            std::memcpy(data + ival.local_start, tmp_buffer + recv_elements, 
                    ival.local_elements * sizeof(T));          
        }
        
        //free communicator
        coll::freeComm(&ival.comm);
        delete[] tmp_buffer;
    }

    /*
     * sort two intervals with two PEs simultaneously
     */
    template<class Compare>
    void sortOnTwoPEsSchizo(QSInterval<T, COMM, REQ> &ival_1,
                            QSInterval<T, COMM, REQ> &ival_2,
                            Compare comp) {
        long long recv_elements_1 = ival_1.global_elements - ival_1.local_elements;
        long long recv_elements_2 = ival_2.global_elements - ival_2.local_elements;
        T* buffer_1 = new T[ival_1.global_elements];
        T* buffer_2 = new T[ival_2.global_elements];
        REQ requests_1[2];
        REQ requests_2[2];

        if (ival_1.rank == 0) {
            coll::Isend(data + ival_1.local_start, ival_1.local_elements, 
                    mpi_type, 1, TWO_PE, ival_1.comm, &requests_1[0]);
            coll::Irecv(buffer_1, recv_elements_1, mpi_type, 1, TWO_PE, 
                    ival_1.comm, &requests_1[1]);
        } else {
            assert(ival_1.rank == 1);
            coll::Isend(data + ival_1.local_start, ival_1.local_elements, 
                    mpi_type, 0, TWO_PE, ival_1.comm, &requests_1[0]);
            coll::Irecv(buffer_1, recv_elements_1, mpi_type, 0, TWO_PE, 
                    ival_1.comm, &requests_1[1]);
        }
        if (ival_2.rank == 0) {
            coll::Isend(data + ival_2.local_start, ival_2.local_elements, 
                    mpi_type, 1, TWO_PE, ival_2.comm, &requests_2[0]);
            coll::Irecv(buffer_2, recv_elements_2, mpi_type, 1, TWO_PE, 
                    ival_2.comm, &requests_2[1]);
        } else {
            assert(ival_2.rank == 1);
            coll::Isend(data + ival_2.local_start, ival_2.local_elements, 
                    mpi_type, 0, TWO_PE, ival_2.comm, &requests_2[0]);
            coll::Irecv(buffer_2, recv_elements_2, mpi_type, 0, TWO_PE, 
                    ival_2.comm, &requests_2[1]);
        }
        
        std::memcpy(buffer_1 + recv_elements_1, data + ival_1.local_start,
                ival_1.local_elements * sizeof(T));
        
        std::memcpy(buffer_2 + recv_elements_2, data + ival_2.local_start,
                ival_2.local_elements * sizeof(T));
        
        coll::Waitall(2, requests_1, MPI_STATUSES_IGNORE);
        coll::Waitall(2, requests_2, MPI_STATUSES_IGNORE);

        if (ival_1.rank == 0) {
            T* nth_element = buffer_1 + ival_1.local_elements - 1;
            std::nth_element(buffer_1, nth_element, buffer_1 + ival_1.global_elements, comp); 
            std::sort(buffer_1, nth_element + 1, comp);             
            std::memcpy(data + ival_1.local_start, buffer_1,
                    ival_1.local_elements * sizeof(T));        
        } else {
            T* nth_element = buffer_1 + recv_elements_1;
            std::nth_element(buffer_1, nth_element, buffer_1 + ival_1.global_elements, comp); 
            std::sort(nth_element, buffer_1 + ival_1.global_elements, comp); 
            std::memcpy(data + ival_1.local_start, buffer_1 + recv_elements_1,
                    ival_1.local_elements * sizeof(T));            
        }
        
        if (ival_2.rank == 0) {
            T* nth_element = buffer_2 + ival_2.local_elements - 1;
            std::nth_element(buffer_2, nth_element, buffer_2 + ival_2.global_elements, comp); 
            std::sort(buffer_2, nth_element + 2, comp);             
            std::memcpy(data + ival_2.local_start, buffer_2,
                    ival_2.local_elements * sizeof(T));            
        } else {
            T* nth_element = buffer_2 + recv_elements_2;
            std::nth_element(buffer_2, nth_element, buffer_2 + ival_2.global_elements, comp); 
            std::sort(nth_element, buffer_2 + ival_2.global_elements, comp);             
            std::memcpy(data + ival_2.local_start, buffer_2 + recv_elements_2,
                    ival_2.local_elements * sizeof(T));         
        }

        //free communicators
        coll::freeComm(&ival_1.comm);
        coll::freeComm(&ival_2.comm);
        
        delete[] buffer_1;
        delete[] buffer_2;
    }

    void printArray(QSInterval<T, COMM, REQ> const &ival, T *array) {
        std::cout << ival.rank << ": ";
        for (long long i = 0; i < split_size; i++)
            std::cout << array[i] << " ";
        std::cout << std::endl;
    }

    /* 
     * Add a interval that can be sorted locally
     */
    void addLocalInterval(QSInterval<T, COMM, REQ> &ival) {
        local_intervals.push_back(ival);
	coll::freeComm(&ival.comm);
    }

    /* 
     * Sort all local intervals 
     */
    template<class Compare>
    void sortLocalIntervals(Compare comp) {
        for (size_t i = 0; i < local_intervals.size(); i++) {
            std::sort(data + local_intervals[i].local_start, 
                      data + local_intervals[i].local_end + 1, comp);
        }
    }

    /* 
     * Add a interval with two PEs 
     */
    void addTwoPEInterval(QSInterval<T, COMM, REQ> const &ival) {
        two_PE_intervals.push_back(ival);
    }

    /* 
     * Sort the saved two PE-intervals 
     */
    template<class Compare>
    void sortTwoPEIntervals(Compare comp) {
        if (two_PE_intervals.size() == 2)
            sortOnTwoPEsSchizo(two_PE_intervals[0], two_PE_intervals[1], comp);
        else if (two_PE_intervals.size() == 1)
            sortOnTwoPEs(two_PE_intervals[0], comp);
        else
            assert(two_PE_intervals.size() == 0);
    }

    /*
     * Returns the current time
     */
    double getTime() {
        return MPI_Wtime();
    }

    int const SEND_INDEX = 50, RECV_INDEX = 51, BCAST_PIVOT = 52, PIVOT_GATHER = 60,
            PIVOT_BCAST = 64, CALC_EXCH = 70, EXCHANGE_SMALL = 80, EXCHANGE_LARGE = 81,
            TWO_PE = 90;
    long long split_size;
    int depth = 0;
    double t_pivot = 0.0, t_calculate = 0.0, t_exchange = 0.0, t_partition = 0.0,
            t_sort_two = 0.0, t_sort_local = 0.0, t_create_comms = 0.0, t_runtime;
    std::vector<double> t_vec_exchange, exchange_times{0.0, 0.0, 0.0, 0.0};
    T *data, *buffer;
    MPI_Datatype mpi_type;
    std::mt19937 generator, sample_generator;
    std::vector<QSInterval<T, COMM, REQ>> local_intervals, two_PE_intervals;
    bool blocking, mpi_split, barriers;
    std::string pivot_samples;
};

#endif // QUICKSORT_HPP
