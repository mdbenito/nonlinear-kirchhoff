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

using namespace dolfin;

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
np.set_printoptions(precision=2, linewidth=120)

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
  DKTGradient::P3Tensor D;

  std::vector<double> cc = {0,0,1,0,1,1};
  DKTGradient::P22Vector p22coeffs;
  DKTGradient::P3Vector p3rcoeffs = {1,0,0,0,0,0,0,0,0};
  dg.update(cc);
  dg.apply_vec(p3rcoeffs, p22coeffs);

//  std::vector<double> M_result = {
//    0.,    1.     0.,    0.,    0.,    0.,    0.,    0.,    0.,  
//    0.,    0.,    1.,    0.,    0.,    0.,    0.,    0.,    0.,  
//    0.,    0.,    0.,    0.,    1.,    0.,    0.,    0.,    0.,  
//    0.,    0.,    0.,    0.,    0.,    1.,    0.,    0.,    0.,  
//    0.,    0.,    0.,    0.,    0.,    0.,    0.,    1.,    0.,  
//    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    1.,  
//    0.,    0.,    0.,   -0.,    0.5,   0.,    0.,    0.5,   0.,  
//    0.,    0.,    0.,   -1.5,   0.,   -0.25,  1.5,   0.,   -0.25,
//   -0.75,  0.13, -0.37,  0.,    0.,    0.,    0.75,  0.13, -0.37,
//   -0.75, -0.37,  0.13,  0.,    0.,    0.,    0.75, -0.37,  0.13,
//   -1.5,  -0.25,  0.,    1.5,  -0.25,  0.,    0.,    0.,    0.,  
//   -0.,    0.,    0.5,   0.,    0.,    0.5,   0.,    0.,    0. };
  
  DKTGradient::P22Tensor p22tensor = {
     1.  ,  0.17,  0.  ,  0.  ,  0.  , -0.67,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
     0.17,  1.  ,  0.17, -0.67,  0.  , -0.67,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
     0.  ,  0.17,  1.  , -0.67,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
     0.  , -0.67, -0.67,  2.67, -1.33,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
     0.  ,  0.  ,  0.  , -1.33,  5.33, -1.33,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
    -0.67, -0.67,  0.  ,  0.  , -1.33,  2.67,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
     0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  1.  ,  0.17,  0.  ,  0.  ,  0.  , -0.67,
     0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.17,  1.  ,  0.17, -0.67,  0.  , -0.67,
     0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.17,  1.  , -0.67,  0.  ,  0.  ,
     0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  , -0.67, -0.67,  2.67, -1.33,  0.  ,
     0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  , -1.33,  5.33, -1.33,
     0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  , -0.67, -0.67,  0.  ,  0.  , -1.33,  2.67 };

  dg.apply(p22tensor, D);

  DKTGradient::P3Tensor D_ok = {
    10.31,   1.53,  1.16, -11.06,  1.75,  0.91,  0.75, -0.22,  0.06, 
    1.53,    1.54,   0.1,  -1.66,  0.21,   0.1,  0.12,  0.33, -0.97, 
    1.16,     0.1,  1.91,  -0.53,  0.21,  0.12, -0.62, -0.44,  -0.3, 
    -11.06, -1.66, -0.53,  14.25, -2.12, -0.12, -3.19,  0.22,  0.34, 
    1.75,    0.21,  0.21,  -2.12,  1.58, -0.69,  0.38,  0.12,  0.02, 
    0.91,     0.1,  0.12,  -0.12, -0.69,  3.56, -0.78, -1.59,  0.18, 
    0.75,    0.12, -0.62,  -3.19,  0.38, -0.78,  2.44,    0., -0.41, 
    -0.22,   0.33, -0.44,   0.22,  0.12, -1.59,    0.,  6.04,  -1.7, 
    0.06,   -0.97,  -0.3,   0.34,  0.02,  0.18, -0.41,  -1.7,  3.05 };

  auto absdiff = [] (double x, double y) { return std::abs(x-y); };
  std::transform(D.begin(), D.end(), D_ok.begin(), D.begin(), absdiff);

  auto diff = std::accumulate(D.begin(), D.end(), 0.0);
  std::cout << "Total difference:" << std::setprecision(3) << diff << "\n";
 
  return near(diff, 0.0) ? 0 : 1;
}

bool
test_dkt_identity()
{
  return false;
}
