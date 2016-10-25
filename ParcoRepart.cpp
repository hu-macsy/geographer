/*
 * ParcoReport.cpp
 *
 *  Created on: 25.10.2016
 *      Author: moritzl
 */

#include "ParcoRepart.h"

namespace ITI {

template<typename IndexType>
scai::lama::DenseVector<IndexType> ParcoRepart<IndexType>::partitionGraph(scai::lama::Matrix &input, scai::lama::Vector &coordinates,
					IndexType dimensions,	IndexType k,  double epsilon) {
	DenseVector<IndexType> result(input.getNumRows(),0);
	return result;
}

template scai::lama::DenseVector<double> ParcoRepart<double>::partitionGraph(scai::lama::Matrix &input, scai::lama::Vector &coordinates,
					double dimensions,	double k,  double epsilon);

}
