#include <iostream>
#include <string>
#include <networkit/io/METISGraphReader.h>
#include <networkit/numerics/LAMG/Lamg.h>
#include <networkit/graph/Graph.h>
#include <networkit/auxiliary/Timer.h>
#include <networkit/algebraic/CSRMatrix.h>
#include <networkit/algebraic/Vector.h>


using namespace std;
using namespace NetworKit;

Vector createZeroSumVec(const int n, size_t seed);

int main() {

    vector<pair<string,string>> instances {
        make_pair("delaunay","../meshes/delaunayTest.graph"),
        make_pair("bubbles","../meshes/bubbles-00010.graph"),
        make_pair("trace","../meshes/trace-00008.graph")
    };

    METISGraphReader reader;
    Aux::Timer timer;

    for(auto const& input: instances) {

        string graph = input.first;
        Graph G = reader.read(input.second);
        Lamg<CSRMatrix> sl;
        CSRMatrix matrix = CSRMatrix::laplacianMatrix(G);
        sl.setupConnected(matrix);
        SolverStatus status;
        Vector b(G.numberOfNodes());
        Vector x(G.numberOfNodes(), 0.);
        b = createZeroSumVec(G.numberOfNodes(), 12345);
        Vector result = x;
        timer.start();
        status = sl.solve(b, result);
        timer.stop();
        cout << " solve time\t " << timer.elapsedMilliseconds() << " mlsecs\n";
    }

}

Vector createZeroSumVec(const int n, size_t seed) {
    mt19937 rand(seed);
    auto rand_value = uniform_int_distribution<int> (0,n-1);
    int v1 = rand_value(rand);
    int v2 = rand_value(rand);
    while( v1 == v2 )
        v2 = rand_value(rand);
    Vector b(n, 0.0);
    b[v1] = 1.0;
    b[v2] = -1.0;
    return b;
}
