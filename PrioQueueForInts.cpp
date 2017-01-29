/*
 * BucketPQ.cpp
 *
 *  Created on: 26.06.2015
 *      Author: Henning
 */

#include <assert.h>

#include "PrioQueueForInts.h"

namespace ITI {

template<typename Key, typename Val>
PrioQueueForInts<Key, Val>::PrioQueueForInts(std::vector<Key>& prios, Key maxPrio):
		buckets(maxPrio+1), nodePtr(prios.size()), myBucket(prios.size(), NetworKit::none),
		minNotEmpty(maxPrio+1), maxNotEmpty(-1), maxPrio(maxPrio), numElems(0)
{
	for (Key i = 0; i < prios.size(); ++i) {
		if (prios[i] != PrioQueueForInts::none) {
			insert(i, prios[i]);
		}
	}
}

template<typename Key, typename Val>
void PrioQueueForInts<Key, Val>::changePrio(Val elem, Key prio) {
	remove(elem);
	insert(elem, prio);
}

template<typename Key, typename Val>
void PrioQueueForInts<Key, Val>::insert(Val elem, Key prio) {
	assert(0 <= prio && prio <= maxPrio);
	assert(0 <= elem && elem < nodePtr.size());

	buckets[prio].push_front(elem);
	nodePtr[elem] = buckets[prio].begin();
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

template<typename Key, typename Val>
Val PrioQueueForInts<Key, Val>::extractMin() {
	if (minNotEmpty > maxPrio) {
		return PrioQueueForInts::none;
	}
	else {
		assert(! buckets[minNotEmpty].empty());
		Val result = buckets[minNotEmpty].front();
		remove(result);
		return result;
	}
}

template<typename Key, typename Val>
Val PrioQueueForInts<Key, Val>::extractMax() {
	if (maxNotEmpty < 0) {
		return NetworKit::none;
	}
	else {
		assert(! buckets[maxNotEmpty].empty());
		Val result = buckets[maxNotEmpty].front();
		remove(result);
		return result;
	}
}

template<typename Key, typename Val>
void PrioQueueForInts<Key, Val>::remove(Val elem) {
	assert(0 <= elem && elem < nodePtr.size());

	if (myBucket[elem] != NetworKit::none) {
		// remove from appropriate bucket
		Key prio = myBucket[elem];
		buckets[prio].erase(nodePtr[elem]);
		myBucket[elem] = NetworKit::none;
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

template<typename Key, typename Val>
Val PrioQueueForInts<Key, Val>::extractAt(Key prio) {
	assert(0 <= prio && prio <= maxPrio);
	if (buckets[prio].empty()) {
		return NetworKit::none;
	}
	else {
		Key result = buckets[prio].front();
		myBucket[result] = NetworKit::none;
		buckets[prio].pop_front();
		return result;
	}
}

template<typename Key, typename Val>
Key PrioQueueForInts<Key, Val>::priority(Val elem) {
	return myBucket[elem];
}

template<typename Key, typename Val>
bool PrioQueueForInts<Key, Val>::empty() const {
	return (numElems == 0);
}

template<typename Key, typename Val>
uint64_t PrioQueueForInts<Key, Val>::size() const {
	return numElems;
}

} /* namespace Aux */
