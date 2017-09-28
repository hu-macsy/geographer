// File:   generate_dt.cpp
// Author: Manuel Holtgrewe <holtgrewe@ira.uka.de>
//
// Creates Delaunay Triangulations of random points in the unit square.
// The Delaunay Triangulations are created using CGAL and written as METIS
// graphs and .xyz files.

#include <cstdio>
#include <cstdlib>
#include <tr1/random>
#include <tr1/unordered_map>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <chrono>

#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

// The Kernel we will use in our datastructures.
struct K : CGAL::Exact_predicates_inexact_constructions_kernel {};

typedef CGAL::Delaunay_triangulation_2<K> Dt;
typedef Dt::Point Point;
typedef Dt::Finite_vertices_iterator Fvi;
typedef Dt::Vertex Vertex;
typedef Dt::Vertex Vertex;
typedef Dt::Vertex_circulator Vc;
typedef Dt::Vertex_handle Vh;

const char *kUsage = "<n> <filename.graph> [seed]";
typedef unsigned long int ULONG;
const int fileTypeVersionNumber= 3;

struct HashPoint {
  std::size_t operator()(const Point &p) const
  {
    std::tr1::hash<double> h;
    return h(p.x()) ^ h(p.y());
  }
};


int main(int argc, char **argv){
    
    if (argc < 3 or argc > 4) {
        fprintf(stderr, "ERROR: Must give two to three parameters.\n");
        fprintf(stderr, "%s %s\n", argv[0], kUsage);
        return 1;
    }
    
    int seed = 0;
    if (argc == 4)
        seed = atoi(argv[3]);
    
    ULONG n = atoi(argv[1]);
    
    std::chrono::time_point<std::chrono::system_clock> start =  std::chrono::system_clock::now();
    
    // Build Delaunay Triangulation.
    printf("Building Delaunay Triangulation..\n");
    Dt t;
    
    { // Generate points.
        std::tr1::mt19937 mt(seed);
        std::cout << "Generating " << n << " points...\n";
        printf("seed = %d\n", seed);
        std::vector<Point> points;
        points.reserve(n);
        for (int i = 0; i < n; ++i)
            points.push_back(Point(1.0 * mt() / mt.max(), 1.0 * mt() / mt.max()));
        
        t.insert(points.begin(), points.end());
    }
    std::chrono::duration<double> dtTime = std::chrono::system_clock::now() - start;
    
    // Getting point ids.
    std::cout<< "time elapsed: " << dtTime.count() << std::endl << std::endl <<"Getting point ids." << std::endl;
    
    std::tr1::unordered_map<Point, int, HashPoint> map;
    int i = 1;
    for (Fvi it = t.finite_vertices_begin(), itend = t.finite_vertices_end(); it != itend; ++it, ++i) {
        map[it->point()] = i;
    }
    
    std::chrono::duration<double> pointsTime = std::chrono::system_clock::now() - start;

    // Getting m and the ia/offsets arrays
    std::cout<< "time elapsed: " << pointsTime.count() << std::endl<< std::endl << "Getting ia/offsets."<< std::endl;
    
    ULONG m = 0;
    
    //TODO: memory costly, can use smaller arrays and multiple writes?
    // maybe doesn't make sense since the structure Dt t is never destroyed.
    ULONG* ia = new ULONG[n+1];
    
    int headerSize = 3; // version, n and m
    
    // the first offset is after the header and th ia array that we will write it first
    ULONG offset = (headerSize + n +1)*sizeof(ULONG) ;
    ia[0] = offset;
    int index =0;
    
    for (Fvi it = t.finite_vertices_begin(), itend = t.finite_vertices_end(); it != itend; ++it) {
        Vc vc = it->incident_vertices(), done(vc);
        ia[index] = offset + m*sizeof(ULONG);
        index++;
        // TODO: probably there is a way to avoid the while but must look in the CGAL data structures
        if ( ! vc.is_empty()) {
            do {
                if (not t.is_infinite(Vh(vc))) {
                    m += 1;
                }
            } while (++vc != done);
        }
    }
    
    assert(index==n);
    ia[index]=offset + m*sizeof(ULONG);
  
    if (m % 2 != 0) { 
        std::cout << m << " = m % 2 != 0\n"; 
        abort(); 
    }

    const char *filename = argv[2];
    char coords_filename[100];
    sprintf(coords_filename, "%s.xyz", filename);

    
    std::ofstream graph_fileBin( filename,  std::ios::binary | std::ios::out );
    std::ofstream coords_fileBin( coords_filename,  std::ios::binary | std::ios::out );
    
    //
    // Writing graph file.
    //
    
    std::chrono::duration<double> iaTime = std::chrono::system_clock::now() - start;
    
    std::cout << "time elapsed: " << iaTime.count() << std::endl<< std::endl << "Writting graph file" << std::endl;
    
    // write header: version, n and m
    graph_fileBin.write((char*)(&fileTypeVersionNumber), sizeof( ULONG ));
    graph_fileBin.write((char*)(&n), sizeof( ULONG ));
    graph_fileBin.write((char*)(&m), sizeof( ULONG ));
        
    //write the offsets/ia array
    graph_fileBin.write((char*)(ia), (n+1)*sizeof(ULONG));
 
    delete[] ia;
      
    // this is memory costly
    //TODO: allocate smaller arrays and write more than one times ?
    ULONG *edges = new ULONG[m];
    
    ULONG pos = 0;
    
    int old = -1;
    Vh first = t.finite_vertices_begin()->handle();
    
    for (Fvi it = t.finite_vertices_begin(), itend = t.finite_vertices_end(); it != itend; ++it) {
        
        int nodeDegree =0 ;
        
        Point p = it->point();
        if (map.find(p) == map.end()) abort();
        int current = map[p];
        if (old > -1 and current <= old) { printf("%d == current <= old == %d\n", current, old); abort(); }
        
        Vc start = it->incident_vertices();
        Vc c = start;
        if (c != 0) {
            do {
                if (t.is_infinite(Vh(c))) continue;
                if (map.find(c->point()) == map.end()) { printf("line %d\n", __LINE__); abort(); }
                edges[pos++] = map[c->point()]-1;         // -1 because of metis format????
                ++nodeDegree;
            } while (++c != start);
        }
        old = current;
    }
    
    assert(pos==m);
    
    graph_fileBin.write((char*)(edges), (m)*sizeof(ULONG));
    graph_fileBin.close();
    delete[] edges;
    
    //
    // Writing coordinate file.
    //
    
    std::chrono::duration<double> writeGraphTime = std::chrono::system_clock::now() - start;
    
    std::cout << "time elapsed: " << writeGraphTime.count() << std::endl<< std::endl << "Writing coordinate file." << std::endl;
    
    // 3 coords per point/node, 3rd coordinate is 0
    double* coords = new double[3*n]; 
    pos = 0;
  
    for (Fvi it = t.finite_vertices_begin(), itend = t.finite_vertices_end(); it != itend; ++it) {
            Point p = it->point();
            coords[pos++] = (double)(p.x());
            coords[pos++] = (double)(p.y());
            coords[pos++] = (double)(0.0);
    }

    assert( pos== 3*n );
    coords_fileBin.write( (char*)(coords), 3*n*sizeof(double) );
    delete[] coords;
    
    coords_fileBin.close();

    std::chrono::duration<double> totalTime = std::chrono::system_clock::now() - start;
    std::cout<< "\033[1;34mDone, total time elapsed: " << totalTime.count() << "\033[0m"<< std::endl  << std::endl;
    return 0;
}


