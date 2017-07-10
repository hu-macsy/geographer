/*****************************************************************************
 * This file is part of the Project ShizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#include "MPI_Ranged.hpp"
#include "Requests.hpp"

void Range::Barrier(Range::Comm comm) {
    Range::Request request;
    Ibarrier(comm, &request);
    Wait(&request, MPI_STATUS_IGNORE);
}


/*
 * Request for the barrier
 */
class Range_Requests::Ibarrier : public Range::R_Req {
public:
    Ibarrier(Range::Comm comm);
    int test(int *flag, MPI_Status *status);

private:
    Range::Comm comm;
    int rank, buf;
    Range::Request request;
};

void Range::Ibarrier(Range::Comm comm, Range::Request* request) {
    *request = std::unique_ptr<R_Req>(new Range_Requests::Ibarrier(comm));
}

Range_Requests::Ibarrier::Ibarrier(Range::Comm comm) : comm(comm), buf(0) {
    Range::IscanAndBcast(&buf, &buf, &buf, 1, MPI_INT, 3000, MPI_SUM, comm, &request);
}

int Range_Requests::Ibarrier::test(int* flag, MPI_Status* status) {
    return Range::Test(&request, flag, MPI_STATUS_IGNORE);
}
