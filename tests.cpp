#include <vector>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <memory>
#include <dolfin.h>

#include "DKTGradient.h"
#include "tests.h"
#include "output.h"
#include "NonlinearKirchhoff.h"
#include "BlockVectorAdapter.h"
#include "dkt_utils.h"

using namespace dolfin;
namespace NLK { 
  using namespace NonlinearKirchhoff;

  template<>
  std::string
  v2s<Array<double>>(const Array<double>& a, int precision)
  {
    std::stringstream ss;
    ss << std::setprecision(precision);
    for (int i = 0; i < a.size(); ++i)
      ss << a[i] << " ";
    return ss.str();
  }
}

//////////////////////////////////////////////////////////////////////
// FIXME!!! THIS IS COPIED & PASTED FROM main.cpp. FACTOR OUT!      //
//////////////////////////////////////////////////////////////////////
std::unique_ptr<Function>
project(std::shared_ptr<const GenericFunction> what,
        std::shared_ptr<const FunctionSpace> where)
{
  Matrix Ap;
  Vector bp;
  LUSolver solver("mumps");
  
  NonlinearKirchhoff::Form_project_lhs project_lhs(where, where);
  NonlinearKirchhoff::Form_project_rhs project_rhs(where);
  std::unique_ptr<Function> f(new Function(where));
  project_rhs.g = what;  // g is a Coefficient in a P3 space (see .ufl)
  assemble_system(Ap, bp, project_lhs, project_rhs, {});
  solver.solve(Ap, *(f->vector()), bp);
  return f;
}

/// TODO: make this into an automated regression test
int
test_dofs()
{
  auto mesh = std::make_shared<dolfin::RectangleMesh>(MPI_COMM_WORLD,
                                                      Point (0.0, 0.0),
                                                      Point (1.0, 1.0),
                                                      2, 2, "right");
  auto W3 =
    std::make_shared<NonlinearKirchhoff::Form_dkt_FunctionSpace_0>(mesh);
  auto T3 =
    std::make_shared<NonlinearKirchhoff::Form_p26_FunctionSpace_0>(mesh);

  auto y0 = project(std::make_shared<Constant>(1.0, 2.0, 3.0), W3);
  NLK::dump_full_tensor(*(y0->vector()), 2, "y0", false);

  for (CellIterator cell(*mesh); !cell.end(); ++cell)
  {
    auto dofs = W3->dofmap()->cell_dofs(cell->index());
    std::cout << "Cell: " << cell->index() << ", DOFs: ";
    for (int i = 0; i < dofs.size()-1; ++i)
      std::cout << dofs[i] << ", ";
    std::cout << dofs[dofs.size()-1] << std::endl;
  }

  return 0;
}

/// TODO: make this into an automated regression test
int
test_BlockVectorAdapter()
{
  const int N = 10;
  auto v0 = std::make_shared<Vector>(MPI_COMM_WORLD, N);
  auto v1 = std::make_shared<Vector>(MPI_COMM_WORLD, N);

  // Fill with some data
  std::vector<double> dd(N);  // actual data
  std::iota(dd.begin(), dd.end(), 0);
  std::vector<int> ind(N);    // indices for set()
  std::iota(ind.begin(), ind.end(), 0);  // 0, 1, ..., N-1
  
  v0->set(dd.data(), ind.size(), ind.data());
  v0->apply("insert");
  v1->set(dd.data(), ind.size(), ind.data());
  v1->apply("insert");

  auto block_v = std::make_shared<BlockVector>(2);
  block_v->set_block(0, v0);
  block_v->set_block(1, v1);
  BlockVectorAdapter flat_v(block_v);
  auto flat_v_content = flat_v.get();
  NLK::dump_full_tensor(flat_v_content, 0, "should be 0s", false);
  
  // Copy blocks 0, 1 into flattened vector
  flat_v.read(0);
  flat_v.read(1);
  // Should be 1...N, 1...N
  NLK::dump_full_tensor(flat_v_content, 0, "should be 1..N, 1...N", false);

  // Fill block 0 with 0s
  v0->zero();
  v0->apply("insert");
  // std::fill(dd.begin(), dd.end(), 0);
  // v0->set(dd.data(), ind.size(), ind.data());
  // v0->apply("insert");

  flat_v.read(0);
  NLK::dump_full_tensor(flat_v_content, 0, "should be 0,..,0,1,..N", false);

  // Modify the beginning of the flattened copy and copy back into
  // BlockVector
  flat_v_content->set(dd.data(), ind.size(), ind.data());
  flat_v.write(0);
  NLK::dump_full_tensor(v0, 2, "should be 1...N", false);

  return 0;
}

/* Python code to produce cc, p22tensor, M, D:
from dolfin import *
import nbimporter
from discrete_gradient import DKTCellGradient
import numpy as np
np.set_printoptions(precision=4, linewidth=120)

domain = UnitSquareMesh(1,1)
V = VectorFunctionSpace(domain, "Lagrange", dim=2, degree=2)
u = TrialFunction(V)
v = TestFunction(V)

a = inner(nabla_grad(u), nabla_grad(v))*dx
A = assemble(a)

grad = DKTCellGradient()
Aa = A.array()
A2 = np.zeros((12,12))
dm = V.dofmap()
for cell_id in range(V.mesh().num_cells()):
    cell = Cell(V.mesh(), cell_id)
    print(cell.get_coordinate_dofs())
    grad.update(cell)
    print(grad.M)
    print("******")
    l2g = dm.cell_dofs(cell_id)   # local to global mapping for cell 
    for i,j in np.ndindex(A2.shape):
        A2[i,j] = Aa[l2g[i], l2g[j]]
    print(A2)
    print("=============")
    print(grad.M.transpose() @ A2 @ grad.M)
*/
int
test_DKT(void)
{
  DKTGradient dg;
  std::vector<double> cc = {0,0,1,0,1,1};
  DKTGradient::P22Vector p22coeffs;
  DKTGradient::P3Vector p3rcoeffs = {1,0,0,0,0,0,0,0,0};
  
  dg.update(cc);
  DKTGradient::M_t M_ok;
  M_ok << 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
          0.0, 0.0, 0.0, -0.0, 0.5, 0.0, 0.0, 0.5, 0.0,
          0.0, 0.0, 0.0, -1.5, 0.0, -0.25, 1.5, 0.0, -0.25,
          -0.75, 0.125, -0.375, 0.0, 0.0, 0.0, 0.75, 0.125, -0.375,
          -0.75, -0.375, 0.125, 0.0, 0.0, 0.0, 0.75, -0.375, 0.125,
          -1.5, -0.25, 0.0, 1.5, -0.25, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0;
  
  DKTGradient::permutation_hack(M_ok);
  auto Mdiff = (M_ok - dg.M()).sum();
  bool ok = near(Mdiff, 0.0, 1e-12);
  std::cout << "Local gradient operator " << (ok ? "OK" : "differs") << "\n";

//const auto& M = dg.M();
//  for(int i =0; i < 12; ++i) {
//    for(int j = 0; j < 9; ++j)
//      std::cout << M(i,j) - M_ok(i, j) << " ";
//    std::cout << "\n";
//  }
  
  dg.apply_vec(p3rcoeffs, p22coeffs);
  
  // TODO: check p22coeffs
  // FIXME: this test is bogus: we need to take into account
  // the permutation_hack()
  DKTGradient::P22Tensor p22tensor = {
    1.0, 0.1667, -0.0, 0.0, 0.0, -0.6667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.1667, 1.0, 0.1667, -0.6667, 0.0, -0.6667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    -0.0, 0.1667, 1.0, -0.6667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, -0.6667, -0.6667, 2.6667, -1.3333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, -1.3333, 5.3333, -1.3333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    -0.6667, -0.6667, 0.0, 0.0, -1.3333, 2.6667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1667, -0.0, 0.0, 0.0, -0.6667, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1667, 1.0, 0.1667, -0.6667, 0.0, -0.6667, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.1667, 1.0, -0.6667, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.6667, -0.6667, 2.6667, -1.3333, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.3333, 5.3333, -1.3333, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.6667, -0.6667, 0.0, 0.0, -1.3333, 2.6667 };

  DKTGradient::P3Tensor D;
  DKTGradient::permutation_hack(dg._M, true); // HACK!!!!
  dg.apply(p22tensor, D);
  DKTGradient::permutation_hack(dg._M); // UNDO HACK!!!!
  
  DKTGradient::P3Tensor D_ok = {
    10.3125, 1.5312, 1.1563, -11.0625, 1.75, 0.9062, 0.75, -0.2188, 0.0625, 
    1.5312, 1.5365, 0.099, -1.6562, 0.2083, 0.099, 0.125, 0.3281, -0.9688, 
    1.1563, 0.099, 1.9115, -0.5313, 0.2083, 0.1198, -0.625, -0.4427, -0.3021, 
    -11.0625, -1.6562, -0.5313, 14.25, -2.125, -0.125, -3.1875, 0.2188, 0.3437, 
    1.75, 0.2083, 0.2083, -2.125, 1.5833, -0.6875, 0.375, 0.125, 0.0208, 
    0.9062, 0.099, 0.1198, -0.125, -0.6875, 3.5625, -0.7812, -1.5885, 0.1823, 
    0.75, 0.125, -0.625, -3.1875, 0.375, -0.7812, 2.4375, 0.0, -0.4063, 
    -0.2188, 0.3281, -0.4427, 0.2188, 0.125, -1.5885, 0.0, 6.0365, -1.6979, 
    0.0625, -0.9688, -0.3021, 0.3437, 0.0208, 0.1823, -0.4063, -1.6979, 3.0469 };

  auto absdiff = [] (double x, double y) { return std::abs(x-y); };
  std::transform(D.begin(), D.end(), D_ok.begin(), D.begin(), absdiff);

  auto diff = std::accumulate(D.begin(), D.end(), 0.0);
  std::cout << "Total difference between gradients:" << std::setprecision(3) 
            << diff << "\n";
  ok = ok && near(diff, 0.0, 1e-12);
  
  int entry=0;
  for (auto d: D)
      std::cout << std::setprecision(3) 
                << d << ((++entry % 9 == 0) ? "\n" : " ");
  
  std::cout << "\n";
  return ok ? 0 : 1;
}

class IdentityDiff : public DiffExpression
{
public:
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = x[0];
    values[1] = x[1];
    values[2] = 0;
  }

  /// Gradient is stored in row format: f_11, f_12, f_21, f_22, f_31, f_32
  void gradient(Array<double>& grad, const Array<double>& x) const
  {
    grad[0] = 1.0; grad[1] = 0.0;
    grad[2] = 0.0; grad[3] = 1.0;
    grad[4] = 0.0; grad[5] = 0.0;
  }
  
  std::size_t value_rank() const { return 1; }
  std::size_t value_dimension(std::size_t i) const { return 3;}
};

class Poly2Diff : public DiffExpression
{
public:
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = x[0] * x[0] + 5*x[1] * x[1];
    values[1] = x[0] * x[1];
    values[2] = x[0] + 3*x[1];
  }

  /// Gradient is stored in row format: f_11, f_12, f_21, f_22, f_31, f_32
  void 
  gradient(Array<double>& grad, const Array<double>& x) const override {
    grad[0] = 2.0*x[0]; grad[1] = 10.0*x[1];
    grad[2] = x[1];     grad[3] = x[0];
    grad[4] = 1.0;      grad[5] = 3.0;
  }
  
  std::size_t value_rank() const { return 1; }
  std::size_t value_dimension(std::size_t i) const { return 3;}
};

int
test_DKT_projection()
{
  double eps = 1e-8;
  auto  fexp = std::make_shared<Poly2Diff>();
  auto  mesh = std::make_shared<UnitSquareMesh>(MPI_COMM_WORLD, 4, 4, "right");
  auto    W3 = std::make_shared<NLK::Form_dkt_FunctionSpace_0>(mesh);
  auto     f = std::shared_ptr<Function>(std::move(project_dkt(fexp, W3)));
  auto    fd = std::shared_ptr<Function>(std::move(eval_dkt(fexp, W3)));

  auto ddofs = dofs_which_differ(f, fd, eps);
  bool ok = ddofs->size() == 0;
  
  if (! ok) {
    std::cout << "dof test FAILED.\n";
/*  std::vector<double> valsf(ddofs->size()), valsfd(ddofs->size());
    f->vector()->get(valsf.data(), ddofs->size(), ddofs->data());
    fd->vector()->get(valsfd.data(), ddofs->size(), ddofs->data());
    std::cout << "dofs: " << NLK::v2s(*ddofs) << "\n";
    std::cout << "f: " << NLK::v2s(valsf, 2) << "\n";
    std::cout << "fd: " << NLK::v2s(valsfd, 2) << "\n";
    
    std::vector<double> allf(W3->dim()), allfd(W3->dim());
    std::vector<la_index> alldofs(W3->dim());
    std::iota(alldofs.begin(), alldofs.end(), 0);
    f->vector()->get(allf.data(), alldofs.size(), alldofs.data());
    fd->vector()->get(allfd.data(), alldofs.size(), alldofs.data());
    std::cout << "f: " << NLK::v2s(allf, 2) << "\n";
    std::cout << "fd: " << NLK::v2s(allfd, 2) << "\n"; */
  } else {
    std::cout << "dof test ok.\n";
  }
  
  return ok ? 0 : 1;
}

int
test_DKT_expression(std::shared_ptr<DiffExpression> fexp)
{
  double eps = 1e-8;
  bool ok = true;
  
  auto mesh = std::make_shared<UnitSquareMesh>(MPI_COMM_WORLD, 4, 4, "right");
  auto W3 = std::make_shared<NLK::Form_dkt_FunctionSpace_0>(mesh);
  auto T3 = std::make_shared<NLK::Form_p26_FunctionSpace_0>(mesh);  
  auto f = std::shared_ptr<Function>(std::move(eval_dkt(fexp, W3)));
  
  DKTGradient dg;
  auto gradvec = std::shared_ptr<Vector>
                 (std::move(dg.apply_vec(T3, W3, f->vector())));
  Function grad(T3, gradvec);
  
  Array<double> values_grad(6), values_gradexp(6);
  auto dist = [](const Array<double>& a, const Array<double>& b) -> double
  {
    double d = 0.0;
    for(int i=0; i<6; ++i)
      d += (a[i]-b[i])*(a[i]-b[i]);
    return d;
  };

  double diff = 0.0;
  for (VertexIterator vit(*mesh); !vit.end(); ++vit) {
      // HACK! careful not to touch the data
    Array<double> coord(2, const_cast<double *>(vit->x()));
    fexp->gradient(values_gradexp, coord);
    grad.eval(values_grad, coord);
    diff += dist(values_gradexp, values_grad);
    
//    std::cout << "x = " << NLK::v2s(coord, 2) << "\n";
//    std::cout << "gradexp = " << NLK::v2s(values_gradexp, 3) << "\n";
//    std::cout << "dktgrad = " << NLK::v2s(values_grad, 3) << "\n";
  }
  diff = std::sqrt(diff);
  ok = diff < eps;
  
  if (!ok)
    std::cout << "Computed and analytic gradient differ by = " << diff << ".\n";
  else
    std::cout << "Computed and analytic gradient agree.\n";

  return ok ? 0 : 1;
}

int
test_DKT_identity()
{
  auto id = std::make_shared<IdentityDiff>();
  return test_DKT_expression(id);
}

int
test_DKT_polynomial()
{
  auto poly = std::make_shared<Poly2Diff>();
  return test_DKT_expression(poly);
}

int
test_initial_condition(std::shared_ptr<const Function> f)
{
  
}