#ifndef __OUTPUT_H
#define __OUTPUT_H

#include <string>

namespace dolfin {
#ifdef __OUTPUT_H  //DISABLE_DUMP
#define dump_full_tensor(...) ;
#else  
  // HACK: I don't know how to use the common interface of
  // GenericTensor and I don't have the time to learn it
  void dump_full_tensor(const GenericMatrix& A, int precision=14, const std::string& msg="");
  void dump_full_tensor(const GenericVector& A, int precision=14, const std::string& msg="" );
#endif
}


#endif // __OUTPUT_H
