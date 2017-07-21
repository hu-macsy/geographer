/*
 * GraphUtils.cpp
 *
 *  Created on: 29.06.2017
 *      Author: moritzl
 */

#include <assert.h>
#include <queue>

#include <scai/hmemo/ReadAccess.hpp>
#include <scai/hmemo/WriteAccess.hpp>

#include "GraphUtils.h"

using std::vector;
using std::queue;

namespace ITI {

namespace GraphUtils {

using scai::hmemo::ReadAccess;
using scai::dmemo::Distribution;

template<typename IndexType, typename ValueType>
IndexType getFarthestLocalNode(const scai::lama::CSRSparseMatrix<ValueType> graph, std::vector<IndexType> seedNodes) {
	/**
	 * Yet another BFS. This currently has problems with unconnected graphs.
	 */
	const IndexType localN = graph.getLocalNumRows();
	const Distribution& dist = graph.getRowDistribution();

	if (seedNodes.size() == 0) return rand() % localN;

	vector<bool> visited(localN, false);
	queue<IndexType> bfsQueue;

	for (IndexType seed : seedNodes) {
		bfsQueue.push(seed);
		assert(seed >= 0 || seed < localN);
		visited[seed] = true;
	}

	const scai::lama::CSRStorage<ValueType>& storage = graph.getLocalStorage();
	ReadAccess<IndexType> ia(storage.getIA());
	ReadAccess<IndexType> ja(storage.getJA());

	IndexType nextNode = 0;
	while (bfsQueue.size() > 0) {
		nextNode = bfsQueue.front();
		bfsQueue.pop();
		visited[nextNode] = true;

		for (IndexType j = ia[nextNode]; j < ia[nextNode+1]; j++) {
			IndexType localNeighbour = dist.global2local(ja[j]);
			if (localNeighbour != nIndex && !visited[localNeighbour]) {
				bfsQueue.push(localNeighbour);
				visited[localNeighbour] = true;
			}
		}
	}

	//if nodes are unvisited, the graph is unconnected and the unvisited nodes are in fact the farthest
	for (IndexType v = 0; v < localN; v++) {
		if (!visited[v]) nextNode = v;
		break;
	}

	return nextNode;
}

template int getFarthestLocalNode(const scai::lama::CSRSparseMatrix<double> graph, std::vector<int> seedNodes);

}

} /* namespace ITI */
