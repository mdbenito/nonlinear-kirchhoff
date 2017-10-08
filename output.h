#ifndef __OUTPUT_H
#define __OUTPUT_H

#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <numeric>

///// stepping_iota():
// Original code from https://codereview.stackexchange.com/a/136158
// The specifications [of std::iota] say that it is not required for
// decltype(*first) and T to match. So we can implement a very small
// proxy, which will have all the needed functionality.
namespace impl_details
{
  template <typename T, typename UnaryAdvanceOp>
  class proxy {
    T value;
    UnaryAdvanceOp op;
  public:
    proxy(T init_value, UnaryAdvanceOp op_) : value(init_value), op(op_) {}

    operator T() { return value; }
    T operator++() { value = op(value); return value; }
    T operator++(int) { T old = value; value = op(value); return old; }
  };
}

template <typename OutputIterator, typename T, typename UnaryAdavanceOp>
void stepping_iota(OutputIterator first, OutputIterator last,
                   T init_value, UnaryAdavanceOp op)
{
  impl_details::proxy<T, UnaryAdavanceOp> p(init_value, op);
  std::iota(first, last, p);
}
  
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
                        const std::string& name="", bool asfile=true,
                        bool force=false);
  void dump_full_tensor(std::shared_ptr<const dolfin::GenericMatrix> A,
                        int precision=14,
                        const std::string& name="", bool asfile=true,
                        bool force=false);

  /*! Writes a vector in format understandable by numpy.loadtxt() */  
  void dump_full_tensor(const dolfin::GenericVector& A, int precision=14,
                        const std::string& name="", bool asfile=true,
                        bool force=false);
  void dump_full_tensor(std::shared_ptr<const dolfin::GenericVector> A,
                        int precision=14,
                        const std::string& name="", bool asfile=true,
                        bool force=false);

  /*! Writes a matrix in format understandable by numpy.loadtxt() */
  void dump_raw_matrix(const double* A, int m, int n, int precision=14,
                       const std::string& name="", bool asfile=true,
                       bool force=false);
    
  /*! Writes a vector in format understandable by numpy.loadtxt() */  
  void dump_raw_matrix(const std::vector<double>& A, int m, int n,
                       int precision=14,
                       const std::string& name="", bool asfile=true,
                       bool force=false);

  /*! Serializes std containers to strings.
   * Requires available conversion to string for the contained type. */
  template<typename T>
  std::string v2s(const T& v, int precision=6)
  {
    std::stringstream ss;
    ss << std::setprecision(precision);
    for (const auto& x: v)
      ss << x << " ";
    return ss.str();
  }
}

#endif // __OUTPUT_H
