/*
 * PrioQueue.h
 *
 *  Created on: 21.02.2014
 *      Author: Henning Meyerhenke
 */

#ifndef PRIOQUEUE_H_
#define PRIOQUEUE_H_

#include <cassert>
#include <set>
#include <vector>
#include <limits>
#include <iostream>

namespace ITI {

/**
 * Priority queue with extract-min and decrease-key.
 * The type Val takes on integer values between 0 and n-1.
 * O(n log n) for construction, O(log n) for typical operations.
 */
template<class Key, class Val>
class PrioQueue {
	typedef std::pair<Key, Val> ElemType;

private:
	std::set<ElemType> pqset;
	std::vector<Key> mapValToKey;

	const Key undefined = std::numeric_limits<Key>::max(); // TODO: make static

public:
	/**
	 * Builds priority queue from the vector @a elems.
	 */
	PrioQueue(const std::vector<ElemType>& elems);

	/**
	 * Builds priority queue from the vector @a keys, values are indices
	 * in @a keys.
	 */
	PrioQueue(std::vector<Key>& keys);

	/**
	* Builds priority queue of the specified size @a len.
	*/
	PrioQueue(uint64_t len);


	virtual ~PrioQueue() = default;

	/**
	 * Inserts key-value pair stored in @a elem.
	 */
	virtual void insert(Key key, Val value);

	/**
	 * Removes the element with minimum key and returns it.
	 */
	virtual ElemType extractMin();

	/**
	 * Returns the element with minimum key without removing it.
	 */
	virtual ElemType inspectMin();

	/**
	* Returns True iff value val is present.
	*/
	virtual bool contains(const Val& val);

	/**
	* Return key of value val if present, undefined otherwise
	*/
	virtual Key getKey(const Val& val);

	/**
	 * Modifies entry with value @a value.
	 * The entry is then set to @a newKey with the same value.
	 * If the corresponding key is not present, the element will be inserted.
	 */
	virtual void updateKey(Key newKey, Val value);

	/**
	 * slightly optimized version of updateKey when the old key is known
	 */
	virtual void updateKey(Key oldKey, Key newKey, Val value);


	/**
	 * Removes key-value pair given by @a elem.
	 */
	virtual void remove(const ElemType& elem);

	/**
	 * Removes key-value pair given by value @a val.
	 */
	virtual void remove(const Val& val);

	/**
	 * @return Number of elements in PQ.
	 */
	virtual uint64_t size() const;


	/**
	 * @return current content of queue
	 */
	virtual std::set<std::pair<Key, Val>> content() const;

	/**
	 * Removes all elements from the PQ.
	 */
	virtual void clear();
};

} /* namespace ITI */

template<class Key, class Val>
ITI::PrioQueue<Key, Val>::PrioQueue(const std::vector<ElemType>& elems) {
	mapValToKey.resize(elems.size());
	for (auto elem: elems) {
		insert(elem.first, elem.second);
	}
}

template<class Key, class Val>
ITI::PrioQueue<Key, Val>::PrioQueue(std::vector<Key>& keys) {
	mapValToKey.resize(keys.size());
	uint64_t index = 0;
	for (auto key: keys) {
		insert(key, index);
		++index;
	}
}

template<class Key, class Val>
ITI::PrioQueue<Key, Val>::PrioQueue(uint64_t len) {
	mapValToKey.resize(len, undefined);
}

template<class Key, class Val>
inline void ITI::PrioQueue<Key, Val>::insert(Key key, Val value) {
	SCAI_REGION( "PrioQueue.insert" )
	if (value >= mapValToKey.size()) {
		uint64_t doubledSize = 2 * mapValToKey.size();
		assert(value < doubledSize);
		mapValToKey.resize(doubledSize);
	}
	pqset.insert(std::make_pair(key, value));
	mapValToKey.at(value) = key;
}

template<class Key, class Val>
inline bool ITI::PrioQueue<Key, Val>::contains(const Val& val) {
	return mapValToKey.at(val) != undefined;
}

template<class Key, class Val>
inline Key ITI::PrioQueue<Key, Val>::getKey(const Val& val) {
	return mapValToKey.at(val);
}
	
template<class Key, class Val>
inline void ITI::PrioQueue<Key, Val>::remove(const ElemType& elem) {
	remove(elem.second);
}

template<class Key, class Val>
inline void ITI::PrioQueue<Key, Val>::remove(const Val& val) {
	SCAI_REGION( "PrioQueue.remove" )
	Key key = mapValToKey.at(val);
	pqset.erase(std::make_pair(key, val));
	mapValToKey.at(val) = undefined;
}

template<class Key, class Val>
std::pair<Key, Val> ITI::PrioQueue<Key, Val>::inspectMin() {
	assert(pqset.size() > 0);
	ElemType elem = (* pqset.begin());
	return elem;
}


template<class Key, class Val>
std::pair<Key, Val> ITI::PrioQueue<Key, Val>::extractMin() {
	SCAI_REGION( "PrioQueue.extractMin" )
	assert(pqset.size() > 0);
	ElemType elem = (* pqset.begin());
	remove(elem);
	return elem;
}

template<class Key, class Val>
inline void ITI::PrioQueue<Key, Val>::updateKey(Key oldKey, Key newKey, Val value) {
	SCAI_REGION( "PrioQueue.updateKey" )
	//slightly optimized version when old key is known, saves one hashmap access
	pqset.erase(std::make_pair(oldKey, value));
	pqset.insert(std::make_pair(newKey, value));

	mapValToKey.at(value) = newKey;
}

template<class Key, class Val>
inline void ITI::PrioQueue<Key, Val>::updateKey(Key newKey, Val value) {
	SCAI_REGION( "PrioQueue.updateKey" )
	// find and remove element with given key
	remove(value);

	// insert element with new value
	insert(newKey, value);
}

template<class Key, class Val>
inline uint64_t ITI::PrioQueue<Key, Val>::size() const {
	return pqset.size();
}

template<class Key, class Val>
inline std::set<std::pair<Key, Val>> ITI::PrioQueue<Key, Val>::content() const {
	return pqset;
}

template<class Key, class Val>
inline void ITI::PrioQueue<Key, Val>::clear() {
	pqset.clear();
	mapValToKey.clear();
}


#endif /* PRIOQUEUE_H_ */
