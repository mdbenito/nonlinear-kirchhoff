#include <algorithm>
#include "dkt_utils.h"
#include <dolfin.h>
#include "NonlinearKirchhoff.h"

namespace dolfin
{
  namespace NLK { using namespace NonlinearKirchhoff; }

  // compute |nablaT y nabla y - id|
  // FIXME!! I'm computing average distance across all vertices instead
  // of integrating

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
    for (int sub = 0; sub < 3; ++sub) {
      // auto dm = (*W3)[sub]->dofmap();
      // The following should be a process-local index
      for (VertexIterator vit(*mesh); !vit.end(); ++vit) {
        auto idx = static_cast<la_index> (vit->index());
        auto dofdx = v2d[9 * idx + 3 * sub + 1];
        auto dofdy = v2d[9 * idx + 3 * sub + 2];
        dxdofs.push_back(dofdx);
        dydofs.push_back(dofdy);
      }
    }
    std::vector<double> dx, dy;
    dx.reserve(dxdofs.size());
    dy.reserve(dydofs.size());
    y.vector()->get(dx.data(), dxdofs.size(), dxdofs.data());
    y.vector()->get(dy.data(), dydofs.size(), dydofs.data());

    double dxdx = std::inner_product(dx.begin(), dx.end(), dx.begin(), 0),
           dxdy = std::inner_product(dx.begin(), dx.end(), dy.begin(), 0),
           dydy = std::inner_product(dy.begin(), dy.end(), dy.begin(), 0);
    // subtract identity at each vertex
    double frobsq = (std::pow(dxdx - N, 2) +
                     std::pow(dydy - N, 2) +
                     2 * std::pow(dxdy, 2)) / N;
    return std::sqrt(frobsq);
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
      Array<double>(2, const_cast<double*> coord(vit->x()));
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
    std::vector<double> vals1, vals2;
    vals1.reserve(indices->size());
    vals2.reserve(indices->size());
    v1->get_local(vals1.data(), indices->size(), indices->data());
    v2->get_local(vals2.data(), indices->size(), indices->data());  

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
    std::vector<la_index> dofs;
    dofs.reserve(dim);
    std::iota(dofs.begin(), dofs.end(), 0);
    std::vector<double> uu, vv;
    uu.reserve(dim);
    vv.reserve(dim);
    u->get(uu.data(), dofs.size(), dofs.data());
    v->get(vv.data(), dofs.size(), dofs.data());
    
    auto absdiff = [] (double x, double y) -> int { return std::abs(x-y); };
    std::transform(uu.begin(), uu.end(), vv.begin(), vv.begin(), absdiff);
    
    std::unique_ptr<std::vector<la_index>> ret(new std::vector<la_index>());
    for (auto d: dofs)
      if (vv[d] > eps)
        ret->push_back(d);
    
    return ret;
  }
}