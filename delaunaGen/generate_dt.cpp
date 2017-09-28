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

struct HashPoint {
  std::size_t operator()(const Point &p) const
  {
    std::tr1::hash<double> h;
    return h(p.x()) ^ h(p.y());
  }
};

int main(int argc, char **argv)
{
  if (argc < 3 or argc > 4) {
    fprintf(stderr, "ERROR: Must give two to three parameters.\n");
    fprintf(stderr, "%s %s\n", argv[0], kUsage);
    return 1;
  }
  int seed = 0;
  if (argc == 4)
    seed = atoi(argv[3]);
  unsigned long long int n = atoi(argv[1]);

  std::chrono::time_point<std::chrono::system_clock> start =  std::chrono::system_clock::now();
    
  // Generate points.
  std::tr1::mt19937 mt(seed);
  std::cout << "Generating " << n << " points...\n";
  printf("seed = %d\n", seed);
  std::vector<Point> points;
  points.reserve(n);            //this has a size limit TODO: break to multiple arrays?
  for (int i = 0; i < n; ++i)
    points.push_back(Point(1.0 * mt() / mt.max(), 1.0 * mt() / mt.max()));
  
  // Build Delaunay Triangulation.
  printf("Building Delaunay Triangulation..\n");
  Dt t;
  t.insert(points.begin(), points.end());
  
  std::chrono::duration<double> dtTime = std::chrono::system_clock::now() - start;
  
  // Getting point ids.
  std::cout<< "time elapsed: " << dtTime.count() << std::endl << std::endl <<"Getting point ids." << std::endl;

  std::tr1::unordered_map<Point, int, HashPoint> map;
  int i = 1;
  for (Fvi it = t.finite_vertices_begin(), itend = t.finite_vertices_end();
       it != itend; ++it, ++i) {
      map[it->point()] = i;
       }

  // Getting n and m.
  printf("Getting m.\n");
  unsigned long long int m = 0;
  for (Fvi it = t.finite_vertices_begin(), itend = t.finite_vertices_end();
       it != itend; ++it) {
    Vc vc = it->incident_vertices(), done(vc);
    if ( ! vc.is_empty()) {
      do {
        if (not t.is_infinite(Vh(vc))) {
          m += 1;
        }
      } while (++vc != done);
    }
  }
  if (m % 2 != 0) { 
      std::cout << m << " = m % 2 != 0\n"; 
      abort(); 
  }

  const char *filename = argv[2];
  char coords_filename[100];
  sprintf(coords_filename, "%s.xyz", filename);
  
  FILE *graph_file = fopen(filename, "w");
  FILE *coords_file = fopen(coords_filename, "w");
  
  
  //
  // Writing graph file.
  //
  
  std::chrono::duration<double> iaTime = std::chrono::system_clock::now() - start;
  
  std::cout << "time elapsed: " << iaTime.count() << std::endl<< std::endl << "Writting graph file" << std::endl;
  
  fprintf(graph_file, "%llu %llu\n", n, m / 2);
  int old = -1;
  Vh first = t.finite_vertices_begin()->handle();
  for (Fvi it = t.finite_vertices_begin(), itend = t.finite_vertices_end();
       it != itend; ++it) {
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
        fprintf(graph_file, "%d ", map[c->point()]);
      } while (++c != start);
    }
    old = current;
    fprintf(graph_file, "\n");
    }
    
    //
    // Writing coordinate file.
    //
    
    std::chrono::duration<double> writeGraphTime = std::chrono::system_clock::now() - start;
    
    std::cout << "time elapsed: " << writeGraphTime.count() << std::endl<< std::endl << "Writing coordinate file." << std::endl;
    
    for (Fvi it = t.finite_vertices_begin(), itend = t.finite_vertices_end();
       it != itend; ++it) {
    Point p = it->point();
    fprintf(coords_file, "%f %f %f\n", p.x(), p.y(), 0.0);
  }

  fclose(coords_file);
  fclose(graph_file);

  std::chrono::duration<double> totalTime = std::chrono::system_clock::now() - start;
  std::cout<< "\033[1;32mDone, total time elapsed: " << totalTime.count() << "\033[0m"<< std::endl  << std::endl;
    
  return 0;
}


