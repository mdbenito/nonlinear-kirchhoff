#ifndef __OUTPUT_H
#define __OUTPUT_H

#include <string>
#include <iostream>

namespace dolfin {
  class GenericMatrix;
  class GenericVector;
}

namespace NLK {

  extern int DEBUG;

  // HACK(S): I don't know how to use the common interface of
  // GenericTensor and I don't have the time to learn it

  /*! Writes a tensor in format understandable by numpy.loadtxt() */
  void dump_full_tensor(const dolfin::GenericMatrix& A, int precision=14,
                        const std::string& name="", bool asfile=true);
  /*! Writes a vector in format understandable by numpy.loadtxt() */  
  void dump_full_tensor(const dolfin::GenericVector& A, int precision=14,
                        const std::string& name="", bool asfile=true);
  /*! Writes a matrix in format understandable by numpy.loadtxt() */
  void dump_raw_matrix(const double* A, int m, int n, int precision=14,
                       const std::string& name="", bool asfile=true);
  /*! Writes a vector in format understandable by numpy.loadtxt() */  
  void dump_raw_matrix(const std::vector<double>& A, int m, int n,
                       int precision=14,
                       const std::string& name="", bool asfile=true);
}

#endif // __OUTPUT_H
