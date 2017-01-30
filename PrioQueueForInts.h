/*
 * BucketPQ.h
 *
 *  Created on: 26.06.2015
 *      Author: Henning
 */

#ifndef BUCKETPQ_H_
#define BUCKETPQ_H_

#include <list>
#include <limits>
#include <vector>

namespace ITI {

/**
 * Addressable priority queue for elements in the range [0,n) and
 * integer priorities in the range [0, maxPrio].
 * Amortized constant running time for each operation.
 */

typedef std::list<int> Bucket;

template<class Key, class Val>
class PrioQueueForInts {
private:
	std::vector<std::list<Key>> buckets;			// the actual buckets
	std::vector<Bucket::iterator> nodePtr;	// keeps track of node positions
	std::vector<Key> myBucket;			// keeps track of current bucket = priority
	unsigned int minNotEmpty;				// current min priority
	int maxNotEmpty;						// current max priority
	Key maxPrio;							// maximum admissible priority
	uint64_t numElems;							// number of elements stored

	/**
	 * Insert element @a elem with priority @a prio.
	 * @param[in] elem Element to be inserted, must be in range [0, n).
	 * @param[in] prio Priority of element to be inserted, must be in range
	 *   [0, maxPrio].
	 */
	void insert(Key prio, Val elem);

public:
	/**
	 * Constructor that initializes the PQ with the full batch of entries.
	 * @param[in] prios Contains the batch of n entries, where prios[i]
	 *   represents the key-value pair (i, prios[i]). Priorities must be in
	 *   range [0, maxPrio] or none (the latter means that the element does
	 *   not exist).
	 * @param[in] maxPrio Maximum priority value.
	 */
	PrioQueueForInts(std::vector<Key>& prios, Key maxPrio);

	/**
	 * Destructor.
	 */
	~PrioQueueForInts() = default;

	/**
	 * Remove element with key @a key from PQ.
	 * @param[in] elem Element to be removed.
	 */
	void remove(Val elem);

	/**
	 * Changes priority of element @a elem to priority @a prio.
	 * @param[in] elem Element whose priority is changed.
	 * @param[in] prio New priority, must be in range [0, maxPrio].
	 */
	void changePrio(Key prio, Val elem);

	/**
	 * @return Element with minimum priority.
	 */
	Val extractMin();

	/**
	 * @return Element with maximum priority.
	 */
	Val extractMax();

	/**
	 * @return Arbitrary element with priority @a prio. Returns none
	 *   if no such element exists.
	 * @param[in] Priority for which a corresponding element shall be returned,
	 *   must be in range [0, maxPrio].
	 */
	Val extractAt(Key prio);

	/**
	 * @return Priority of elem @a elem.
	 * @param[in] Element whose priority shall be returned.
	 */
	Key priority(Val elem);

	/**
	 * @return True if priority queue does not contain any elements, otherwise false.
	 */
	bool empty() const;

	/**
	 * @return Number of elements contained in priority queue.
	 */
	uint64_t size() const;

	const static Key none = std::numeric_limits<Key>::max();
};

} /* namespace Aux */
#endif /* BUCKETPQ_H_ */
