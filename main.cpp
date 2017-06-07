#include <iostream>
#include <iomanip>
#include <numeric>
#include <vector>
#include <dolfin.h>

#include "KirchhoffAssembler.h"
#include "LinearKirchhoff.h"
#include "HermiteDirichletBC.h"
#include "output.h"

using namespace dolfin;

class Force : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = -9.8;  // TODO put some sensible value here
  }
};

class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary;
      // near(x[0], 0) || near(x[0], M_PI) ||
      // near(x[1], -M_PI/2) || near(x[1], M_PI/2);
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

  LinearKirchhoff::Form_project_lhs project_lhs(where, where);
  LinearKirchhoff::Form_project_rhs project_rhs(where);
  std::unique_ptr<Function> f(new Function(where));
  project_rhs.g = what;  // g is a Coefficient in a P3 space (see .ufl)
  assemble_system(Ap, bp, project_lhs, project_rhs, {});
  solver.solve(Ap, *(f->vector()), bp);
  return f;
}

/* 
 *
 */
int
dostuff(void)
{
  auto mesh = std::make_shared<RectangleMesh>(MPI_COMM_WORLD,
                                              Point (0, -M_PI/2), Point (M_PI, M_PI/2),
                                              20, 20); //, "crossed");
  auto W = std::make_shared<LinearKirchhoff::Form_dkt_FunctionSpace_0>(mesh);
  auto Theta = std::make_shared<LinearKirchhoff::Form_p22_FunctionSpace_0>(mesh);
  
  auto u0 = project_dkt(std::make_shared<Constant>(0.0), W);
  auto boundary = std::make_shared<DirichletBoundary>();
  auto force = std::make_shared<Force>();

  HermiteDirichletBC bc(W, u0, boundary);
  
  LinearKirchhoff::Form_dkt a(W, W);
  LinearKirchhoff::Form_force L(W);
  LinearKirchhoff::Form_p22 p22(Theta, Theta);


  Function u(W);
  Matrix A;
  Vector b;
  KirchhoffAssembler assembler;
  Assembler rhs_assembler;
  LUSolver solver;

  Table table("Assembly and application of BCs");
    
  std::cout << "Projecting force onto W... ";
  tic();
  auto f = project_dkt(force, W);
  table("Projection", "time") = toc();
  std::cout << "Done.\n";

  std::cout << "Assembling LHS... ";
  tic();
  assembler.assemble(A, a, p22);
  table("LHS assembly", "time") = toc();
  std::cout << "Done.\n";

  std::cout << "Assembling RHS... ";
  tic();
  L.f = f;
  rhs_assembler.assemble(b, L);
  table("RHS assembly", "time") = toc();
  std::cout << "Done.\n";

  dump_full_tensor(A, 3);
  
  std::cout << "Applying BCs... ";
  tic();
  bc.apply(A, b);
  table("BC application", "time") = toc();
  std::cout << "Done.\n";

  std::cout << "Solving... ";
  tic();
  solver.solve(A, *(u.vector()), b);
  table("Solution", "time") = toc();
  std::cout << "Done.\n";

  // info(table);  // outputs "<Table of size 5 x 1>"
  std::cout << table.str(true) << std::endl;

  std::cout << std::endl;
  dump_full_tensor(A, 3);
  std::cout << std::endl;
  dump_full_tensor(b, 3);
  std::cout << std::endl;
  dump_full_tensor(*u.vector(), 2);
  std::cout << std::endl;
  // Save solution in VTK format
  File file("solution.pvd");
  file << u;
  
  return 1;
}

int
main(void)
{
  auto mesh = std::make_shared<RectangleMesh>(MPI_COMM_WORLD,
                                              Point (0, -M_PI/2), Point (M_PI, M_PI/2),
                                              1, 1);//, "crossed");
  auto W = std::make_shared<LinearKirchhoff::Form_dkt_FunctionSpace_0>(mesh);
  auto Theta = std::make_shared<LinearKirchhoff::Form_p22_FunctionSpace_0>(mesh);
  
  auto u0 = project_dkt(std::make_shared<Constant>(1.0), W); 

  dump_full_tensor(*(u0->vector()), 1);

  for (CellIterator cell(*mesh); !cell.end(); ++cell) {
    auto dofs = W->dofmap()->cell_dofs(cell->index());
    std::cout << "Cell: " << cell->index() << ", DOFs: ";
    for (int i = 0; i < dofs.size()-1; ++i)
      std::cout << dofs[i] << ", ";
    std::cout << dofs[dofs.size()-1] << std::endl;
  }
  
  File file("u0.pvd");
  file << *u0;

  dostuff();
  return 1;
  
}
