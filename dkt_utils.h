#ifndef __DKTUTILS_H
#define __DKTUTILS_H

#include <memory>
#include <vector>
#include <dolfin.h>

namespace dolfin
{ 
  /*! An expression with gradient.
   * This is to be used with eval_dkt()
   */
  class DiffExpression : public Expression
  {
  public:
    virtual void gradient(Array<double>& grad,
                          const Array<double>& x) const = 0;
  };

  /*! Compute |nablaT y nabla y - id|
   * FIXME!! I'm computing average distance across all vertices instead
   * of integrating
   */
  double distance_to_isometry(Function& y);
  
  /*!
   * 
   */
  std::unique_ptr<Function> eval_dkt(std::shared_ptr<const DiffExpression> fexp,
                                     std::shared_ptr<const FunctionSpace> W3);

  /*!
   * 
   */
  std::unique_ptr<Function>
  project_dkt(std::shared_ptr<const GenericFunction> what,
              std::shared_ptr<const FunctionSpace> where);
  /*!
   * 
   */
  std::shared_ptr<std::vector<int>>
  nodal_indices(std::shared_ptr<const FunctionSpace> W3);
  
  /*!
   * 
   */
  double dkt_inner(std::shared_ptr<const GenericVector> v1,
                   std::shared_ptr<const GenericVector> v2,
                   std::shared_ptr<const FunctionSpace> W3);

  /*!
   * 
   */
  double dkt_inner(const Function& f1, const Function& f2);
  
  /*!
   * 
   */
  std::unique_ptr<std::vector<la_index>>
  dofs_which_differ(std::shared_ptr<const Function> u,
                    std::shared_ptr<const Function> v,
                    double eps=1e-8);
  
  /*! Rounds small (local) entries in a GenericVector to zero. 
   * This makes only (limited) sense for y0, because the projection of the 
   * identity function from CG3 to DKT produces lots of noisy entries.
   */
  void round_zeros(GenericVector& v, double precision = 1e-6);
}

#endif