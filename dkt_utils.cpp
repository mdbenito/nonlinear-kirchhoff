#include <algorithm>
#include "dkt_utils.h"
#include <dolfin.h>
#include "NonlinearKirchhoff.h"

namespace dolfin
{
  namespace NLK { using namespace NonlinearKirchhoff; }

  double
  distance_to_isometry(Function& y)
  {
    auto W3 = y.function_space();
    auto mesh = W3->mesh();
    auto N = mesh->num_vertices();
    auto v2d = vertex_to_dof_map(*W3);
    std::vector<la_index> dxdofs, dydofs;
    dxdofs.reserve(N*3);
    dydofs.reserve(N*3);
    for (VertexIterator vit(*mesh); !vit.end(); ++vit) {
      // The following should be a process-local index
      auto idx = static_cast<la_index> (vit->index());
      for (int sub = 0; sub < 3; ++sub) {  
        auto dofdx = v2d[9 * idx + 3 * sub + 1];
        auto dofdy = v2d[9 * idx + 3 * sub + 2];
        dxdofs.push_back(dofdx);
        dydofs.push_back(dofdy);
      }
    }
    std::vector<double> dx(dxdofs.size()), dy(dydofs.size());
    y.vector()->get(dx.data(), dxdofs.size(), dxdofs.data());
    y.vector()->get(dy.data(), dydofs.size(), dydofs.data());

    double frob = 0.0;
    for(auto i=0; i < N; ++i) {
      double xx = 0, yy = 0, xy = 0;
      for(int j=0; j < 3; ++j) {
        xx += dx[3*i+j]*dx[3*i+j];
        yy += dy[3*i+j]*dy[3*i+j];
        xy += dx[3*i+j]*dy[3*i+j];
      }
      frob += std::sqrt((xx - 1.0)*(xx - 1.0) + 2.0*xy*xy + (yy - 1)*(yy - 1));
    }

    // HACK! approximate the integral of a nodal basis function by the area
    // of the minimal triangle. This is VERY inaccurate.
    double fudge_factor = 0.5*std::pow(y.function_space()->mesh()->hmin(), 2);
    return frob * fudge_factor;
  }

  std::unique_ptr<Function>
  eval_dkt(std::shared_ptr<const DiffExpression> fexp,
           std::shared_ptr<const FunctionSpace> W3)
  {
    auto mesh = W3->mesh();
    auto N = mesh->num_vertices();
    std::vector<la_index> evdofs, dxdofs, dydofs;
    evdofs.reserve(N*3);
    dxdofs.reserve(N*3);
    dydofs.reserve(N*3);
    std::vector<double> evvals, dxvals, dyvals;
    evvals.reserve(N*3);
    dxvals.reserve(N*3);
    dyvals.reserve(N*3);
    Array<double> values(3), grad(6);
    auto v2d = vertex_to_dof_map(*W3);
    for (VertexIterator vit(*mesh); !vit.end(); ++vit) {
      Array<double> coord(2, const_cast<double*>(vit->x()));
      fexp->eval(values, coord);
      fexp->gradient(grad, coord);
      // The following should be a process-local index
      auto idx = static_cast<la_index> (vit->index());
      for (int sub=0; sub < 3; ++sub) {
        auto dofev = v2d[9*idx + 3*sub + 0];
        auto dofdx = v2d[9*idx + 3*sub + 1];
        auto dofdy = v2d[9*idx + 3*sub + 2];
        evdofs.push_back(dofev);
        dxdofs.push_back(dofdx);
        dydofs.push_back(dofdy);

        evvals.push_back(values[sub]);
        dxvals.push_back(grad[2*sub]);
        dyvals.push_back(grad[2*sub+1]);
      }
    }

    std::unique_ptr<Function> f(new Function(W3));
    f->vector()->set(evvals.data(), evdofs.size(), evdofs.data());
    f->vector()->set(dxvals.data(), dxdofs.size(), dxdofs.data());
    f->vector()->set(dyvals.data(), dydofs.size(), dydofs.data());

    return f;
  }

  /// FIXME! This seems not to work properly!
  /// Projects a GenericFunction, which should be in a P^3x2 space onto
  /// the given FunctionSpace, which should be DKT.  In order for this
  /// to be general, I'd need to prepare a variational problem here and
  /// compile it on the fly with ffc, etc. instead of "hardcoding" stuff
  /// in the UFL file.  One should be returning unique_ptr, remember
  /// your Gurus of the week... 
  /// https://herbsutter.com/2013/05/30/gotw-90-solution-factories/
  std::unique_ptr<Function>
  project_dkt(std::shared_ptr<const GenericFunction> what,
              std::shared_ptr<const FunctionSpace> where)
  {
    Matrix Ap;
    Vector bp;
    LUSolver solver("mumps");
  
    NLK::Form_project_lhs project_lhs(where, where);
    NLK::Form_project_rhs project_rhs(where);
    std::unique_ptr<Function> f(new Function(where));
    project_rhs.g = what;  // g is a Coefficient in a P3 space (see .ufl)
    assemble_system(Ap, bp, project_lhs, project_rhs, {});
    solver.solve(Ap, *(f->vector()), bp);
    return f;
  }

  std::unique_ptr<std::vector<la_index>>
  nodal_indices(std::shared_ptr<const FunctionSpace> W3)
  {
    int N = W3->mesh()->num_vertices();
    std::unique_ptr<std::vector<la_index>> indices(new std::vector<la_index>());
    indices->reserve(N);
    auto v2d = vertex_to_dof_map(*W3);
    for (VertexIterator v(*(W3->mesh())); !v.end(); ++v) {
      // The following should be a process-local index
      auto idx = static_cast<la_index>(v->index());
      for (int sub = 0; sub < 3; ++sub) {     // iterate over the 3 subspaces
        auto dof = v2d[9*idx + 3*sub];
        indices->push_back(dof);
      }
    }
    return indices;
  }

  double
  dkt_inner(std::shared_ptr<const GenericVector> v1,
            std::shared_ptr<const GenericVector> v2,
            std::shared_ptr<const FunctionSpace> W3)
  {
    assert(v1->size() == v2->size());
    // FIXME: call only once, not at every inner product!
    const auto& indices = nodal_indices(W3);
    auto N = indices->size();
    std::vector<double> vals1(N), vals2(N);
    v1->get_local(vals1.data(), N, indices->data());
    v2->get_local(vals2.data(), N, indices->data());  

    return std::inner_product(vals1.begin(), vals1.end(), vals2.begin(), 0.0);
  }

  double
  dkt_inner(const Function& f1, const Function& f2)
  {
    return dkt_inner(f1.vector(), f2.vector(), f1.function_space());
  }

  std::unique_ptr<std::vector<la_index>>
  dofs_which_differ(std::shared_ptr<const Function> f,
                    std::shared_ptr<const Function> g,
                    double eps)
  {
    dolfin_assert(f->function_space() == g->function_space());
    
    auto u = f->vector();
    auto v = g->vector();
    auto dim = f->function_space()->dim();
    std::vector<la_index> dofs(dim);
    std::iota(dofs.begin(), dofs.end(), 0);
    std::vector<double> uu(dim), vv(dim);
    u->get(uu.data(), dofs.size(), dofs.data());
    v->get(vv.data(), dofs.size(), dofs.data());
    
    auto absdiff = [] (double x, double y) -> double { return std::abs(x-y); };
    std::transform(uu.begin(), uu.end(), vv.begin(), vv.begin(), absdiff);
    
    std::unique_ptr<std::vector<la_index>> ret(new std::vector<la_index>());
    for (auto d: dofs)
      if (vv[d] > eps)
        ret->push_back(d);
    
    return ret;
  }

  void
  round_zeros(GenericVector& v, double precision)
  {
    auto range = v.local_range();
    auto numrows = range.second - range.first;
    std::vector<la_index> rows(numrows);
    std::iota(rows.begin(), rows.end(), 0);
    std::vector<double> block(numrows);
    v.get_local(block.data(), numrows, rows.data());

    std::transform(block.begin(), block.end(), block.begin(),
                   [] (double v) -> double {
                     return (std::abs(v) < 1e-8) ? 0.0 : v;
                   });
    v.set_local(block.data(), numrows, rows.data());
  }
}
