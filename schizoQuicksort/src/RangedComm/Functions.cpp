/*****************************************************************************
 * This file is part of the Project ShizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#include "Functions.hpp"
#include <iostream>

void copyArray(char *array1, char const *array2, int size) {
    for (int i = 0; i < size; i++) {
        array1[i] = array2[i];
    }
}


void printArray(char *array, int size) {
    for (int i = 0; i < size; i++)
        std::cout << (int) array[i] << " ";
    std::cout << std::endl;
}

