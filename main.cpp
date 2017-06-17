#include <iostream>
#include <iomanip>
#include <numeric>
#include <vector>
#include <tuple>
#include <dolfin.h>

#include "NonlinearKirchhoff.h"
#include "IsometryConstraint.h"
#include "KirchhoffAssembler.h"
#include "BlockMatrixAdapter.h"
#include "BlockVectorAdapter.h"
#include "output.h"

using namespace dolfin;

class Force : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = 0;
    values[1] = 0;
    values[2] = -9.8;   // TODO put some sensible value here
  }
  std::size_t value_rank() const { return 1; }
  std::size_t value_dimension(std::size_t i) const { return 3;}
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
std::unique_ptr<Function>
// std::shared_ptr<Function>
project_dkt(std::shared_ptr<const GenericFunction> what,
            std::shared_ptr<const FunctionSpace> where)
{
  // std::cout << "project_dkt()" << "\n";
  Matrix Ap;
  Vector bp;
  LUSolver solver;
  
  NonlinearKirchhoff::Form_project_lhs project_lhs(where, where);
  NonlinearKirchhoff::Form_project_rhs project_rhs(where);
  std::unique_ptr<Function> f(new Function(where));
  project_rhs.g = what;  // g is a Coefficient in a P3 space (see .ufl)
  // std::cout << "    coefficient set." << "\n";
  assemble_system(Ap, bp, project_lhs, project_rhs, {});
  // std::cout << "    system assembled." << "\n";
  solver.solve(Ap, *(f->vector()), bp);
  // std::cout << "    system solved." << "\n";
  return f;
}


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
                                              1, 1); //, "crossed");
  auto W3 = std::make_shared<NonlinearKirchhoff::Form_dkt_FunctionSpace_0>(mesh);
  auto T3 = std::make_shared<NonlinearKirchhoff::Form_p22_FunctionSpace_0>(mesh);

  
  // factor multiplying the bending energy term
  double alpha = 1.0;
  
  // time step size. In the paper the triangulation consists of halved squares
  // and tau is the length of the sides, not the diagonal, i.e. hmin().
  double tau = mesh->hmin();

  // Initial data: careful that it fulfils the BCs.
  auto y0 = project_dkt(std::make_shared<Constant>(0, 0, 0), W3);
  
  // The discretised isometry constraint includes the condition for
  // the nodes on the Dirichlet boundary to be zero. This ensures that
  // the updates don't change the values of the initial condition,
  // which should fulfill the BC
  auto  left = std::make_shared<LeftBoundary>();  
  auto right = std::make_shared<RightBoundary>();
  VertexFunction<bool> dirichlet_boundary(mesh, false);
  left->mark(dirichlet_boundary, true);
  right->mark(dirichlet_boundary, true);
  IsometryConstraint B(*W3, dirichlet_boundary);
  B.update_with(*y0);

  // Upper left block in the full matrix (constant)
  auto A = std::make_shared<Matrix>();

  // Lower right block:
  // HACK: I really don't know how to create an empty 4x4 Matrix,
  // so I use PETSc... duh
  Mat tmp;
  MatCreateSeqAIJ(MPI_COMM_WORLD, 4, 4, 0, NULL, &tmp);
  auto zeroMat = std::make_shared<PETScMatrix>(tmp);
  auto zeroVec = std::make_shared<Vector>();
  zeroMat->init_vector(*zeroVec, 0);   // second arg is dim, meaning *zeroVec = Ax for some x

  auto block_Mk = std::make_shared<BlockMatrix>(2, 2);
  block_Mk->set_block(0, 0, A);
  block_Mk->set_block(1, 0, B.get());
  block_Mk->set_block(0, 1, B.get_transposed());
  block_Mk->set_block(1, 1, zeroMat);

  Table table("Assembly and application of BCs");
  
  std::cout << "Projecting force onto W^3... ";
  tic();
  auto force = std::make_shared<Force>();
  auto f = std::shared_ptr<const Function>(std::move(project_dkt(force, W3)));
  table("Projection", "time") = toc();
  std::cout << "Done.\n";

  std::cout << "Assembling bilinear form... ";
  NonlinearKirchhoff::Form_dkt a(W3, W3);
  NonlinearKirchhoff::Form_p22 p22(T3, T3);
  tic();
  assembler.assemble(*A, a, p22);
  auto Ao = A->copy();   // Store copy to use in the computation of the RHS
  *A *= 1 + alpha*tau;   // Because we transform A here
  table("Form assembly", "time") = toc();
  std::cout << "Done.\n";

  // dump_full_tensor(*A, 2);
  
  // This requires that the nonzeros for the blocks be already set up
  BlockMatrixAdapter Mk(block_Mk); 
  Mk.read(0,0);
  
  std::cout << "Assembling force vector... ";
  NonlinearKirchhoff::Form_force l(W3);
  Vector L;
  tic();
  l.f = f;
  rhs_assembler.assemble(L, l);
  table("Force assembly", "time") = toc();
  std::cout << "Done.\n";


  // Setup system solution at step k: The first block is the update
  // for the deformation y_{k+1}, the second is ignored (its just 4
  // entries so it shouldn't hurt performance anyway)
  auto dtY = std::make_shared<Vector>();
  auto ignored = std::make_shared<Vector>();
  A->init_vector(*dtY, 0);
  zeroMat->init_vector(*ignored, 0);
  auto block_dtY_L = std::make_shared<BlockVector>(2);
  block_dtY_L->set_block(0, dtY);
  block_dtY_L->set_block(1, ignored);
  
  BlockVectorAdapter dtY_L(block_dtY_L);

  Function y(*y0);       // Deformation y_{k+1}, begin with initial condition

  // Setup right hand side at step k. The content of the first block
  // is set in the loop, the second is always zero
  auto block_Fk = std::make_shared<BlockVector>(2);
  auto top_Fk = std::make_shared<Vector>();
  A->init_vector(*top_Fk, 0);  // second arg is dim, meaning *top_Fk = Ax for some x
  block_Fk->set_block(0, top_Fk);
  block_Fk->set_block(1, zeroVec);

  BlockVectorAdapter Fk(block_Fk);
    
  bool stop = false;
  int max_steps = 10;
  int step = 0;
  table("Compute RHS", "time") = 0;
  table("Update constraint", "time") = 0;
  table("Solution", "time") = 0;
  while (! stop && ++step <= max_steps) {
    std::cout << "\n## Step " << step << " ##\n\n";
    std::cout << "Computing RHS... ";
    tic();
    // This isn't exactly elegant...
    A->mult(*(y.vector()), *top_Fk);
    *top_Fk -= L;
    *top_Fk *= -tau;
    Fk.read(0);
    table("Compute RHS", "time") =
      table.get_value("Compute RHS", "time") + toc();
    std::cout << "Done.\n";
    
    std::cout << "Updating discrete isometry constraint... ";
    tic();
    B.update_with(y);
    Mk.read(0,1);  // This is *extremely* inefficient. At least I could
    Mk.read(1,0);  // update B in place inside Mk.
    table("Update constraint", "time") =
      table.get_value("Update constraint", "time") + toc();
    std::cout << "Done.\n";

    std::cout << "Solving... ";
    tic();
    solver.solve(Mk.get(), dtY_L.get(), Fk.get());
    dtY_L.write(0);  // Update block_dtY_L back from dty_L
    table("Solution", "time") =
      table.get_value("Solution", "time") + toc();
    std::cout << "Done.\n";

    y.vector()->axpy(-tau, *dtY);  // y = y - tau*dty
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
  file << y;
  
  return 1;
}


int
main(void)
{
  // auto mesh = std::make_shared<RectangleMesh>(MPI_COMM_WORLD,
  //                                             Point (0, -M_PI/2),
  //                                             Point (M_PI, M_PI/2),
  //                                             1, 1);//, "crossed");
  // auto W3 = std::make_shared<NonlinearKirchhoff::Form_dkt_FunctionSpace_0>(mesh);
  // auto T3 = std::make_shared<NonlinearKirchhoff::Form_p22_FunctionSpace_0>(mesh);

  // auto y0 = project_dkt(std::make_shared<Constant>(1.0, 2.0, 3.0), W3);

  // dump_full_tensor(*(y0->vector()), 1);

  // for (CellIterator cell(*mesh); !cell.end(); ++cell)
  // {
  //   auto dofs = W3->dofmap()->cell_dofs(cell->index());
  //   std::cout << "Cell: " << cell->index() << ", DOFs: ";
  //   for (int i = 0; i < dofs.size()-1; ++i)
  //     std::cout << dofs[i] << ", ";
  //   std::cout << dofs[dofs.size()-1] << std::endl;
  // }
  
  // File file("y0.pvd");
  // file << *y0;

  return dostuff();
}
