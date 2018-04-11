/*
 * BucketPQ.h
 *
 *  Created on: 26.06.2015
 *      Author: Henning
 */

#pragma once

#include <list>
#include <limits>
#include <vector>

namespace ITI {

/**
 * Addressable priority queue for elements in the range [0,n) and
 * integer priorities in the range [0, maxPrio].
 * Amortized constant running time for each operation.
 */

typedef int index;
typedef std::list<index> Bucket;

constexpr static index none = std::numeric_limits<index>::max();

class PrioQueueForInts {
private:
	std::vector<Bucket> buckets;			// the actual buckets
	std::vector<Bucket::iterator> nodePtr;	// keeps track of node positions
	std::vector<index> myBucket;			// keeps track of current bucket = priority
	unsigned int minNotEmpty;				// current min priority
	int maxNotEmpty;						// current max priority
	index maxPrio;							// maximum admissible priority
	index numElems;							// number of elements stored

public:
	/**
	 * Constructor that initializes the PQ with the full batch of entries.
	 * @param[in] prios Contains the batch of n entries, where prios[i]
	 *   represents the key-value pair (i, prios[i]). Priorities must be in
	 *   range [0, maxPrio] or none (the latter means that the element does
	 *   not exist).
	 * @param[in] maxPrio Maximum priority value.
	 */
	PrioQueueForInts(std::vector<index>& prios, index maxPrio);

	PrioQueueForInts(index size, index maxPrio);

	/**
	 * Insert element @a elem with priority @a prio.
	 * @param[in] elem Element to be inserted, must be in range [0, n).
	 * @param[in] prio Priority of element to be inserted, must be in range
	 *   [0, maxPrio].
	 */
	void insert(index prio, index elem);

	/**
	 * Destructor.
	 */
	~PrioQueueForInts() = default;

	/**
	 * Remove element with key @a key from PQ.
	 * @param[in] elem Element to be removed.
	 */
	void remove(index elem);

	/**
	 * Changes priority of element @a elem to priority @a prio.
	 * @param[in] prio New priority, must be in range [0, maxPrio].
	 * @param[in] elem Element whose priority is changed.
	 */
	void updateKey(index prio, index elem);

	index getMinNotEmpty() const;

	index getMaxNotEmpty() const;

	/**
	 * @return Element with minimum priority.
	 */
	index extractMin();

	/**
	 * @return Element with maximum priority.
	 */
	index extractMax();

	/**
	 * @return Arbitrary element with priority @a prio. Returns none
	 *   if no such element exists.
	 * @param[in] Priority for which a corresponding element shall be returned,
	 *   must be in range [0, maxPrio].
	 */
	index extractAt(index prio);

	/**
	 * @return Priority of elem @a elem.
	 * @param[in] Element whose priority shall be returned.
	 */
	index priority(index elem) const;

	/**
	 * @return True if priority queue does not contain any elements, otherwise false.
	 */
	bool empty() const;

	/**
	 * @return Number of elements contained in priority queue.
	 */
	index size() const;
};

} /* namespace Aux */
