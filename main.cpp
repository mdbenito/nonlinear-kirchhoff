#include <iostream>
#include <iomanip>
#include <numeric>
#include <vector>
#include <tuple>
#include <cmath>
#include <dolfin.h>
#include <dolfin/fem/DirichletBC.h>

#include "NonlinearKirchhoff.h"
#include "IsometryConstraint.h"
#include "KirchhoffAssembler.h"
#include "BlockMatrixAdapter.h"
#include "BlockVectorAdapter.h"
#include "output.h"
#include "sweet/options.hpp"

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
    return near(x[0], 0);
  }
};

class RightBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return near(x[0], 1);
  }
};

class FullBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary;
  }
};


class BoundaryData : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = x[0];
    values[1] = x[1];
    values[2] = 0;
  }
  std::size_t value_rank() const { return 1; }
  std::size_t value_dimension(std::size_t i) const { return 3;}
};


/// In order for this to be general, I'd need to prepare a variational
/// problem, compile it on the fly with ffc, etc.
// I should be returning unique_ptr, remember your Gurus of the week...
// https://herbsutter.com/2013/05/30/gotw-90-solution-factories/
std::unique_ptr<Function>
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
  // std::cout << "    coefficient set." << "\n";
  assemble_system(Ap, bp, project_lhs, project_rhs, {});
  // std::cout << "    system assembled." << "\n";
  solver.solve(Ap, *(f->vector()), bp);
  // std::cout << "    system solved." << "\n";
  return f;
}


/// Rounds small (local) entries in a GenericVector to zero.  This
/// makes only (limited) sense for y0, because the projection of the
/// identity function from CG3 to DKT produces lots of noisy entries.
void
round_zeros(GenericVector& v, double precision=1e-6)
{
    auto range = v.local_range();
    auto numrows = range.second - range.first;
    std::vector<la_index> rows(numrows);
    std::iota(rows.begin(), rows.end(), 0);
    std::vector<double> block(numrows);
    v.get_local(block.data(), numrows, rows.data());
    
    std::transform(block.begin(), block.end(), block.begin(),
                   [] (double v) -> double {
                     return (std::abs(v)< 1e-8) ? 0.0 : v;
                   });
    v.set_local(block.data(), numrows, rows.data());
}


/// Does the magic.
///
///   mesh: duh.
///   alpha: Factor multiplying the bending energy term
///   tau: time step size
///
/// TODO: allow an adaptive step size policy
int
dostuff(std::shared_ptr<RectangleMesh> mesh, double alpha, double tau,
        int max_steps)
{
  KirchhoffAssembler assembler;
  Assembler rhs_assembler;
  PETScLUSolver solver;

  auto W3 = std::make_shared<NonlinearKirchhoff::Form_dkt_FunctionSpace_0>(mesh);
  auto T3 = std::make_shared<NonlinearKirchhoff::Form_p22_FunctionSpace_0>(mesh);

  std::cout << "Running on a mesh with " << mesh->num_cells() << " cells.\n";
  std::cout << "FE space has " << W3->dim() << " dofs.\n";
  std::cout << "Using alpha = " << alpha << ", tau = " << tau << ".\n";

  // The stiffness matrix includes the condition for the nodes on the
  // Dirichlet boundary to be zero. This ensures that the updates
  // don't change the values of the initial condition, which should
  // fulfill the BC
  auto bdry = std::make_shared<FullBoundary>();
  auto zero = std::shared_ptr<Function>(std::move(project_dkt(std::make_shared<Constant>(0.0, 0.0, 0.0), W3)));
  DirichletBC bc(W3, zero, bdry);

  Table table("Assembly and application of BCs");

  std::cout << "Projecting initial data onto W^3... ";
  tic();
  // Initial data: careful that it fulfils the BCs.
  auto y0 = project_dkt(std::make_shared<BoundaryData>(), W3);
  table("Projection of y0", "time") = toc();
  std::cout << "Done.\n";
  
  auto& v = *(y0->vector());
  round_zeros(v);
  dump_full_tensor(v, 4, "y0.txt");
  
  std::cout << "Initialising constraint... ";
  IsometryConstraint B(*W3);
  std::cout << "Done.\n";

  std::cout << "Populating constraint... ";
  tic();
  B.update_with(*y0);
  table("Constraint updates", "time") = toc();
  std::cout << "Done.\n";

  // dump_full_tensor(*B.get(), 12, "B0.txt");

  // Upper left block in the full matrix (constant)
  auto A = std::make_shared<Matrix>();

  // Lower right block:
  auto zeroMat = B.get_zero_padding();
  auto zeroVec = std::make_shared<Vector>();
  // second arg is dim, meaning *zeroVec = Ax for some x
  zeroMat->init_vector(*zeroVec, 0);

  auto block_Mk = std::make_shared<BlockMatrix>(2, 2);
  block_Mk->set_block(0, 0, A);
  block_Mk->set_block(1, 0, B.get());
  block_Mk->set_block(0, 1, B.get_transposed());
  block_Mk->set_block(1, 1, zeroMat);

  std::cout << "Projecting force onto W^3... ";
  tic();
  auto force = std::make_shared<Force>();
  auto f = std::shared_ptr<const Function>(std::move(project_dkt(force,
                                                                 W3)));
  table("Projection of f", "time") = toc();
  std::cout << "Done.\n";

  std::cout << "Assembling bilinear form... ";
  NonlinearKirchhoff::Form_dkt a(W3, W3);
  NonlinearKirchhoff::Form_p22 p22(T3, T3);
  tic();
  assembler.assemble(*A, a, p22);
  auto Ao = A->copy();   // Store copy to compute the RHS later,
  *A *= 1 + alpha*tau;   // because we transform A here
  bc.apply(*A);
  table("Assembly", "time") = toc();
  dump_full_tensor(*A, 12, "A.txt");
  std::cout << "Done.\n";
  
  // This requires that the nonzeros for the blocks be already set up
  BlockMatrixAdapter Mk(block_Mk); 
  Mk.read(0,0);  // Read in A, we read the rest in the loop
  
  std::cout << "Assembling force vector... ";
  NonlinearKirchhoff::Form_force l(W3);
  Vector L;
  tic();
  l.f = f;
  rhs_assembler.assemble(L, l);
  table("Assembly", "time") = toc();
  std::cout << "Done.\n";


  // Setup system solution at step k: The first block is the update
  // for the deformation y_{k+1}, the second is ignored (its just 7
  // entries so it shouldn't hurt performance anyway)
  auto dtY = std::make_shared<Vector>();
  auto ignored = std::make_shared<Vector>();
  A->init_vector(*dtY, 0);
  zeroMat->init_vector(*ignored, 0);
  auto block_dtY_L = std::make_shared<BlockVector>(2);
  block_dtY_L->set_block(0, dtY);
  block_dtY_L->set_block(1, ignored);
  
  BlockVectorAdapter dtY_L(block_dtY_L);
  
  Function y(*y0);  // Deformation y_{k+1}, begin with initial condition

  // Setup right hand side at step k. The content of the first block
  // is set in the loop, the second is always zero
  auto block_Fk = std::make_shared<BlockVector>(2);
  auto top_Fk = std::make_shared<Vector>();
  // second arg is dim, meaning *top_Fk = Ax for some x
  A->init_vector(*top_Fk, 0);
  block_Fk->set_block(0, top_Fk);
  block_Fk->set_block(1, zeroVec);

  BlockVectorAdapter Fk(block_Fk);

  bool stop = false;
  int step = 0;
  table("RHS computation", "time") = 0;
  table("Solution", "time") = 0;
  while (! stop && ++step <= max_steps) {
    std::cout << "\n## Step " << step << " ##\n\n";
    std::cout << "Computing RHS... ";
    tic();
    // This isn't exactly elegant...
    A->mult(*(y.vector()), *top_Fk);
    *top_Fk -= L;
    *top_Fk *= -tau;
    bc.apply(*top_Fk);
    Fk.read(0);
    table("RHS computation", "time") =
      table.get_value("RHS computation", "time") + toc();
    std::cout << "Done.\n";
    // dump_full_tensor(Fk.get(), 12, "Fk.txt");
    
    std::cout << "Updating discrete isometry constraint... ";
    tic();
    B.update_with(y);
    Mk.read(0,1);  // This is *extremely* inefficient. At least I
    Mk.read(1,0);  // could update B in place inside Mk.
    table("Constraint updates", "time") =
      table.get_value("Constraint updates", "time") + toc();
    std::cout << "Done.\n";

    std::cout << "Solving... ";
    tic();
    solver.solve(Mk.get(), dtY_L.get(), Fk.get());
    dtY_L.write(0);  // Update block_dtY_L(0,0) back from dty_L
    table("Solution", "time") =
      table.get_value("Solution", "time") + toc();
    std::cout << "Done.\n";
    dump_full_tensor(dtY_L.get(), 12, "dtY_L.txt");
    // dump_full_tensor(dtY_L.get(), 12, "Update", false);
    
    y.vector()->axpy(-tau, *dtY);  // y = y - tau*dty
    
    dump_full_tensor(*(y.vector()), 12, "yk.txt");
    // dump_full_tensor(*(y.vector()), 12, "Solution", false);
  }
  
  dump_full_tensor(Mk.get(), 12, "Mk.txt");
  
  // info(table);  // outputs "<Table of size 5 x 1>"
  std::cout << table.str(true) << std::endl;

  // Save solution in VTK format
  File file("solution.pvd");
  file << y;
  
  return 1;
}


void
test_dofs(std::shared_ptr<RectangleMesh> mesh)
{
  auto W3 = std::make_shared<NonlinearKirchhoff::
                             Form_dkt_FunctionSpace_0>(mesh);
  auto T3 = std::make_shared<NonlinearKirchhoff::
                             Form_p22_FunctionSpace_0>(mesh);

  auto y0 = project_dkt(std::make_shared<Constant>(1.0, 2.0, 3.0),
                        W3);

  dump_full_tensor(*(y0->vector()), 2, "y0", false);

  for (CellIterator cell(*mesh); !cell.end(); ++cell)
  {
    auto dofs = W3->dofmap()->cell_dofs(cell->index());
    std::cout << "Cell: " << cell->index() << ", DOFs: ";
    for (int i = 0; i < dofs.size()-1; ++i)
      std::cout << dofs[i] << ", ";
    std::cout << dofs[dofs.size()-1] << std::endl;
  }
  
}

int
main(int argc, char** argv)
{
  //// Default options
  int m = 4, n = 4;
  double alpha = 1.0;
  double tau = 1e-6;
  int max_steps = 24;
  double eps = 1e-6;  // TODO: unused
  int verbose = 0;    // TODO: unused
  std::string diagonal = "right";

  bool help = false;
  sweet::Options opt(argc, const_cast<char**>(argv),
                     "Nonlinear Kirchhoff model on the unit square.");
  opt.get("-h", "--help", "Show this help", help);
  opt.get("-v", "--verbose", "Verbosity level, 0-3 (TODO)", verbose);
  opt.get("-m", "--num_vertical", "number of vertical subdivisions", m);
  opt.get("-n", "--num_horizontal", "number of horizontal subdivisions", n);
  opt.get("-d", "--diagonal", "Direction of diagonals: \"left\", \"right\", \"left/right\", \"crossed\"", diagonal);
  opt.get("-a", "--alpha", "alpha", alpha);
  opt.get("-t", "--tau", "timestep *scaling*", tau);
  opt.get("-x", "--max_steps", "Maximum number of time steps", max_steps);
  opt.get("-e", "--eps_stop", "Stopping threshold (TODO)", eps);
  if (opt.help_requested())
    return 1;

  auto mesh = std::make_shared<RectangleMesh>(MPI_COMM_WORLD,
                                              Point (0.0, 0.0), Point (1.0, 1.0),
                                              m, n, diagonal);
  
  // In the paper the triangulation consists of halved squares
  // and tau is the length of the sides i.e. hmin() in our case.
  tau *= mesh->hmin();
  
  return dostuff(mesh, alpha, tau, max_steps);
}
