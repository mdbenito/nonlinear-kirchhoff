#include <iostream>
#include <iomanip>
#include <numeric>
#include <vector>
#include <dolfin.h>

#include <petscmat.h>
#include <petscsys.h>

#include "KirchhoffAssembler.h"
#include "NonlinearKirchhoff.h"
#include "output.h"

using namespace dolfin;

class Force : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = -9.8;  // TODO put some sensible value here
  }
};

class LeftBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary && near(x[0], 0);
  }
};

class RightBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary && near(x[0], M_PI);
  }
};


/// In order for this to be general, I'd need to prepare a variational
/// problem, compile it on the fly with ffc, etc.
// I should be returning unique_ptr, remember your Gurus of the week...
// https://herbsutter.com/2013/05/30/gotw-90-solution-factories/
// But everyone expects a shared_ptr and I just don't wan't to be writing
// std::shared_ptr<const Function> f(std::move(project_dkt...))
//std::unique_ptr<Function>
std::shared_ptr<Function>
project_dkt(std::shared_ptr<const GenericFunction> what,
            std::shared_ptr<const FunctionSpace> where)
{
  Matrix Ap;
  Vector bp;
  LUSolver solver;

  NonlinearKirchhoff::Form_project_lhs project_lhs(where, where);
  NonlinearKirchhoff::Form_project_rhs project_rhs(where);
  std::unique_ptr<Function> f(new Function(where));
  project_rhs.g = what;  // g is a Coefficient in a P3 space (see .ufl)
  assemble_system(Ap, bp, project_lhs, project_rhs, {});
  solver.solve(Ap, *(f->vector()), bp);
  return f;
}

#if 0
/* 
 *
 */
int
dostuff(void)
{
  KirchhoffAssembler assembler;
  Assembler rhs_assembler;
  LUSolver solver;
  auto mesh = std::make_shared<RectangleMesh>(MPI_COMM_WORLD,
                                              Point (0, -M_PI/2), Point (M_PI, M_PI/2),
                                              20, 20, "crossed");
  auto W3 = std::make_shared<NonlinearKirchhoff::Form_dkt_FunctionSpace_0>(mesh);
  auto T3 = std::make_shared<NonlinearKirchhoff::Form_p22_FunctionSpace_0>(mesh);

  
  // factor multiplying the bending energy term
  double alpha = 1.0;
  
  // time step size. In the paper the triangulation consists of halved squares
  // and tau is the length of the sides, not the diagonal, i.e. hmin().
  double tau = mesh->hmin();

  // Initial data: careful that it fulfils the BCs.
  auto y0 = project_dkt(std::make_shared<Constant>({0,0,0}), W3);
  
  // The discretised isometry constraint includes the condition for
  // the nodes on the Dirichlet boundary to be zero. This ensures that
  // the updates don't change the values of the initial condition,
  // which should fulfill the BC
  auto  left = std::make_shared<LeftBoundary>();  
  auto right = std::make_shared<RightBoundary>();
  VertexFunction<bool> dirichlet_boundary(mesh, false);
  left.mark(dirichlet_boundary, true);
  right.mark(dirichlet_boundary, true);
  IsometryConstraint B(W3, dirichlet_boundary);
  
  Function y(W3);

  // Upper left block in the full matrix (constant)
  auto A = std::make_shared<Matrix>();

  // Lower right block:
  // HACK: I really don't know how to create an empty 4x4 Matrix,
  // so I use PETSc... duh
  Mat* tmpMat;
  MatCreateDense(mesh.mpi_comm(), 4, 4, 4, 4, NULL, tmpMat);
  auto zeroMat = std::make_shared<PETScMatrix>(tmpMat);
  auto zeroVec = std::make_shared<Vector>();
  zeroMat->init_vector(*zeroVec, 0);
  
  BlockMatrix Mk(2, 2);
  Mk.set_block(0, 0, A);
  Mk.set_block(1, 1, Zero);

  Table table("Assembly and application of BCs");
  
  std::cout << "Projecting force onto W^3... ";
  tic();
  auto force = std::make_shared<Force>();
  auto f = project_dkt(force, W3);
  table("Projection", "time") = toc();
  std::cout << "Done.\n";

  std::cout << "Assembling bilinear form... ";
  NonlinearKirchhoff::Form_dkt a(W3, W3);
  NonlinearKirchhoff::Form_p22 p22(Th3, Th3);
  tic();
  assembler.assemble(*A, a, p22);
  auto Ao = A.copy();   // Store copy to use in the computation of the RHS
  *A *= 1 + alpha*tau;  // Because we transform A here
  table("Form assembly", "time") = toc();
  std::cout << "Done.\n";

  std::cout << "Assembling force vector... ";
  NonlinearKirchhoff::Form_force L(W3);
  Vector b;
  tic();
  L.f = f;
  rhs_assembler.assemble(b, L);
  table("Force assembly", "time") = toc();
  std::cout << "Done.\n";


  auto dkt_grad = DKTGradient()
  auto Fk = std::make_shared<Vector>();  // Non-zero part of the RHS at step k
  auto tmp = std::make_shared<Vector>();
  bool stop = false;
  while (! stop) {
    std::cout << "Computing RHS... ";
    tic();
    // This isn't exactly elegant...
    A.mult(*y.vector(), *tmp);
    *tmp -= *b.vector();
    *tmp *= -tau;
    table("Compute RHS", "time") += toc();
    Fk.set_block(0, tmp);
    Mk.set_block(1, zeroVec);
    std::cout << "Done.\n";
    
    std::cout << "Updating discrete isometry constraint... ";
    tic();
    B.update_with(y);
    table("Update constraint", "time") += toc();
    Mk.set_block(1, 0, B.get());
    Mk.set_block(0, 1, B.get_transposed());
    std::cout << "Done.\n";

    std::cout << "Solving... ";
    tic();
    solver.solve(Mk, *(yy.vector()), Fk);
    table("Solution", "time") += toc();
    std::cout << "Done.\n";
  }
  
  // info(table);  // outputs "<Table of size 5 x 1>"
  std::cout << table.str(true) << std::endl;

  // std::cout << std::endl;
  // dump_full_tensor(A, 3);
  // std::cout << std::endl;
  // dump_full_tensor(b, 3);
  // std::cout << std::endl;
  // dump_full_tensor(*u.vector(), 2);
  // std::cout << std::endl;
  // Save solution in VTK format
  File file("solution.pvd");
  file << u;
  
  return 1;
}
#endif

int
main(void)
{
  auto mesh = std::make_shared<RectangleMesh>(MPI_COMM_WORLD,
                                              Point (0, -M_PI/2), Point (M_PI, M_PI/2),
                                              1, 1);//, "crossed");
  auto W3 = std::make_shared<NonlinearKirchhoff::Form_dkt_FunctionSpace_0>(mesh);
  auto T3 = std::make_shared<NonlinearKirchhoff::Form_p22_FunctionSpace_0>(mesh);

  auto y0 = project_dkt(std::make_shared<Constant>(1.0, 2.0, 3.0), W3);

  dump_full_tensor(*(y0->vector()), 1);

  for (CellIterator cell(*mesh); !cell.end(); ++cell) {
    auto dofs = W3->dofmap()->cell_dofs(cell->index());
    std::cout << "Cell: " << cell->index() << ", DOFs: ";
    for (int i = 0; i < dofs.size()-1; ++i)
      std::cout << dofs[i] << ", ";
    std::cout << dofs[dofs.size()-1] << std::endl;
  }
  
  File file("y0.pvd");
  file << *y0;

  // dostuff();
  return 1;
  
}
