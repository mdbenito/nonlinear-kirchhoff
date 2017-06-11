#ifndef __OUTPUT_H
#define __OUTPUT_H

#include <iostream>
#include <iomanip>
#include <Eigen/Dense>

namespace dolfin {

  // HACK: I don't know how to use the common interface of
  // GenericTensor and I don't have the time to learn it
  void dump_full_tensor(const GenericMatrix& A, int precision=14);
  void dump_full_tensor(const GenericVector& A, int precision=14);
  
  template<typename mat_t>
  void
  dump_eigen(const mat_t& A)
  {
    std::cout << std::setprecision(4);
    for (std::size_t i = 0; i < A.rows(); ++i) {
      for (std::size_t j = 0; j < A.cols(); ++j) {
        std::cout << std::setw(8) << A(i,j);
      }
      std::cout << std::endl;
    }
  }
}


#endif // __OUTPUT_H
