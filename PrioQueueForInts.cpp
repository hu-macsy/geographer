/*
 * BucketPQ.cpp
 *
 *  Created on: 26.06.2015
 *      Author: Henning
 */

#include <assert.h>

#include "PrioQueueForInts.h"

namespace ITI {


PrioQueueForInts::PrioQueueForInts(index size, index maxPrio):
		buckets(maxPrio+1), nodePtr(size), myBucket(size, none),
		minNotEmpty(maxPrio+1), maxNotEmpty(-1), maxPrio(maxPrio), numElems(0)
{

}


PrioQueueForInts::PrioQueueForInts(std::vector<index>& prios, index maxPrio):
		buckets(maxPrio+1), nodePtr(prios.size()), myBucket(prios.size(), none),
		minNotEmpty(maxPrio+1), maxNotEmpty(-1), maxPrio(maxPrio), numElems(0)
{
	for (index i = 0; i < prios.size(); ++i) {
		if (prios[i] != none) {
			insert(i, prios[i]);
		}
	}
}


void PrioQueueForInts::updateKey(index prio, index elem) {
	remove(elem);
	insert(elem, prio);
}


void PrioQueueForInts::insert(index prio, index elem) {
	assert(0 <= prio && prio <= maxPrio);
	assert(0 <= elem && elem < nodePtr.size());

	nodePtr[elem] = buckets[prio].end();
	buckets[prio].push_back(elem);
	myBucket[elem] = prio;
	++numElems;

	// bookkeeping
	if (prio < minNotEmpty || minNotEmpty > maxPrio) {
		minNotEmpty = prio;
	}
	if (maxNotEmpty < 0 || prio > (unsigned int) maxNotEmpty) {
		maxNotEmpty = prio;
	}
}


index PrioQueueForInts::extractMin() {
	if (minNotEmpty > maxPrio) {
		assert(empty());
		return none;
	}
	else {
		assert(! buckets[minNotEmpty].empty());
		index result = buckets[minNotEmpty].front();
		remove(result);
		return result;
	}
}


index PrioQueueForInts::extractMax() {
	if (maxNotEmpty < 0) {
		assert(empty());
		return none;
	}
	else {
		assert(! buckets[maxNotEmpty].empty());
		index result = buckets[maxNotEmpty].front();
		remove(result);
		return result;
	}
}


void PrioQueueForInts::remove(index elem) {
	assert(0 <= elem && elem < nodePtr.size());

	if (myBucket[elem] != none) {
		// remove from appropriate bucket
		index prio = myBucket[elem];
		buckets[prio].erase(nodePtr[elem]);
		myBucket[elem] = none;
		--numElems;

		// adjust max pointer if necessary
		while (buckets[maxNotEmpty].empty() && maxNotEmpty >= 0) {
			--maxNotEmpty;
		}

		// adjust min pointer if necessary
		while (buckets[minNotEmpty].empty() && minNotEmpty <= maxPrio) {
			++minNotEmpty;
		}
	}
}


index PrioQueueForInts::extractAt(index prio) {
	assert(0 <= prio && prio <= maxPrio);
	if (buckets[prio].empty()) {
		return none;
	}
	else {
		index result = buckets[prio].front();
		myBucket[result] = none;
		buckets[prio].pop_front();
		return result;
	}
}


index PrioQueueForInts::priority(index elem) const {
	return myBucket[elem];
}


bool PrioQueueForInts::empty() const {
	return (numElems == 0);
}


index PrioQueueForInts::size() const {
	return numElems;
}

} /* namespace ITI */
