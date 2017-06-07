#include <iostream>
#include <iomanip>
#include <dolfin.h>

#include "DKTGradient.h"

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
void
testDKT(void)
{
  DKTGradient dg;
  DKTGradient::P3Tensor D;

  std::vector<double> cc = {0,0,1,0,1,1};
  DKTGradient::P22Vector p22coeffs;
  std::vector<double> p3rcoeffs = {1,0,0,0,0,0,0,0,0};
  dg.update(cc);
  dg.apply_vec(p3rcoeffs, p22coeffs);

  for (int i = 0; i < 12; ++i) {
    for (int j = 0; j < 9; ++j)
      std::cout << std::setw(8) << dg.M(i,j) << " ";
    std::cout << std::endl;
  }

  std::cout << "\n";

  std::vector<double> p22tensor = {
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

  int l = 0;
  for (auto c : D)
    std::cout << std::setprecision(2) << std::setw(6) << c
              << (++l % 9 == 0 ? "\n" : " ");

  std::cout << "\n";
}
