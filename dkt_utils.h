#ifndef __DKTUTILS_H
#define __DKTUTILS_H

#include <memory>
#include <vector>

namespace dolfin
{
  std::unique_ptr<Function> eval_dkt(std::shared_ptr<const BoundaryData> fexp,
                                     std::shared_ptr<const FunctionSpace> W3);

  std::unique_ptr<Function> project_dkt(std::shared_ptr<const GenericFunction> what,
                                        std::shared_ptr<const FunctionSpace> where);
  
  std::unique_ptr<std::vector<la_index>>
  nodal_indices(std::shared_ptr<const FunctionSpace> W3);

  double dkt_inner(std::shared_ptr<const GenericVector> v1,
                   std::shared_ptr<const GenericVector> v2,
                   std::shared_ptr<const FunctionSpace> W3);

  double dkt_inner(const Function& f1, const Function& f2);
}

#endif __DKTUTILS_H
