###
 # @file make.inc
 #
 # @license
 # Copyright (c) 2009-2016
 # Fraunhofer Institute for Algorithms and Scientific Computing SCAI
 # for Fraunhofer-Gesellschaft
 #
 # This file is part of the SCAI framework LAMA.
 #
 # LAMA is free software: you can redistribute it and/or modify it under the
 # terms of the GNU Affero General Public License as published by the Free
 # Software Foundation, either version 3 of the License, or (at your option)
 # any later version.
 #
 # LAMA is distributed in the hope that it will be useful, but WITHOUT ANY
 # WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 # FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for
 # more details.
 #
 # You should have received a copy of the GNU Affero General Public License
 # along with LAMA. If not, see <http://www.gnu.org/licenses/>.
 #
 # Other Usage
 # Alternatively, this file may be used in accordance with the terms and
 # conditions contained in a signed written agreement between you and
 # Fraunhofer SCAI. Please contact our distributor via info[at]scapos.com.
 # @endlicense
 #
 # @brief Definition of variables that will be include in each Makefile of examples
 # @author Lauretta Schubert
 # @date 02.03.2016
###

SCAI_ROOT              = /home/harry/Lama
SCAI_BOOST_INCLUDE_DIR = /usr/include
SCAI_CUDA_INCLUDE_DIR  = 

CXX      = /usr/bin/c++
CXX_CUDA = 

CXX_FLAGS      = -std=c++11 -fopenmp -O3 -DNDEBUG -ffast-math -msse4a
CXX_CUDA_FLAGS = 

SCAI_DEFINES   =  -DSCAI_ASSERT_LEVEL_ERROR -DSCAI_LOG_LEVEL_INFO -DSCAI_TRACE_OFF

ADDITIONAL_EXAMPLE_LINK_LIBRARIES = 
SCAI_LD_FLAGS                     =  -L/home/harry/Lama/lib -lscai_solver -lscai_lama -lscai_dmemo -lscai_sparsekernel -lscai_utilskernel -lscai_blaskernel -lscai_kregistry -lscai_hmemo -lscai_tasking -lscai_tracing -lscai_logging -lscai_common 

SCAI_CUDA_LIBRARY_PATH = 

SCAI_EXECUTABLES =  ParcoRepartTest.exe
SCAI_MODULES     = 
SCAI_LIBS        = 
