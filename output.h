#ifndef __OUTPUT_H
#define __OUTPUT_H

#include <string>
#include <iostream>

namespace dolfin {
#ifdef DISABLE_DUMP
#define dump_full_tensor(...) ;
#define dump_raw_matrix(...) ;
#else
  class GenericMatrix;
  class GenericVector;
  
  // HACK: I don't know how to use the common interface of
  // GenericTensor and I don't have the time to learn it
  void dump_full_tensor(const GenericMatrix& A, int precision=14,
                        const std::string& name="", bool asfile=true);
  void dump_full_tensor(const GenericVector& A, int precision=14,
                        const std::string& name="", bool asfile=true);
  void dump_raw_matrix(const double* A, int m, int n, int precision=14,
                       const std::string& name="", bool asfile=true);
  void dump_raw_matrix(const std::vector<double>& A, int m, int n,
                       int precision=14,
                       const std::string& name="", bool asfile=true);
#endif
}


#endif // __OUTPUT_H
