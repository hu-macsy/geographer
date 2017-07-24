/*****************************************************************************
 * This file is part of the Project ShizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#ifndef REQUESTS_HPP
#define REQUESTS_HPP
#include "MPI_Ranged.hpp"

/*
 * This class contains one request class for each non-blocking operation.
 */
class Range_Requests {
    friend class Range;
private:

    /*
     * The request classes for the diffent communication operations
     */    
    class Ibarrier;
    class Ibcast;
    class Igather;
    class Ireduce;
    class Iscan;
    class IscanAndBcast;
    class Isend;
    class Irecv;
};

#endif /* REQUESTS_HPP */


