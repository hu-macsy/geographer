/*****************************************************************************
 * This file is part of the Project ShizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#ifndef SCHIZOQS_HPP
#define SCHIZOQS_HPP

#include <mpi.h>
#include "QuickSort.hpp"
#include "../RangedComm/MPI_Ranged.hpp"

namespace SchizoQS {
    template<typename T>
    void sort(T *first, int size) {
        MPI_Comm comm = MPI_COMM_WORLD;
        QuickSort<T, Range::Comm, Range::Request> qs(false, false, false, "16_50");
        int seed = 324719;
        qs.sort(first, size, comm, seed);
    }
}

#endif /* SCHIZOQS_HPP */

