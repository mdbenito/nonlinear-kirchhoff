#ifndef __DKTUTILS_H
#define __DKTUTILS_H

#include <memory>
#include <vector>
#include <dolfin.h>

namespace dolfin
{ 
  class DiffExpression : public Expression
  {
  public:
    virtual void gradient(Array<double>& grad,
                          const Array<double>& x) const = 0;
  };

  double distance_to_isometry(Function& y);
  
  std::unique_ptr<Function> eval_dkt(std::shared_ptr<const DiffExpression> fexp,
                                     std::shared_ptr<const FunctionSpace> W3);

  std::unique_ptr<Function>
  project_dkt(std::shared_ptr<const GenericFunction> what,
              std::shared_ptr<const FunctionSpace> where);

  std::unique_ptr<std::vector<int>>
  nodal_indices(std::shared_ptr<const FunctionSpace> W3);

  double dkt_inner(std::shared_ptr<const GenericVector> v1,
                   std::shared_ptr<const GenericVector> v2,
                   std::shared_ptr<const FunctionSpace> W3);

  double dkt_inner(const Function& f1, const Function& f2);
}

#endif