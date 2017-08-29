#include <iostream>
#include <iomanip>
#include <numeric>
#include <vector>
#include <tuple>
#include <cmath>
#include <dolfin.h>
#include <unistd.h>

#include "NonlinearKirchhoff.h"
#include "IsometryConstraint.h"
#include "KirchhoffAssembler.h"
#include "BlockMatrixAdapter.h"
#include "BlockVectorAdapter.h"
#include "output.h"
#include "tests.h"
#include "sweet/options.hpp"

using namespace dolfin;

namespace NLK { using namespace NonlinearKirchhoff; }

const double LEFT = -2.0, RIGHT = 2.0, BOTTOM = 0.0, TOP = 1.0;
  
class Force : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = 0;
    values[1] = 0;
    values[2] = -1e-5;
  }
  std::size_t value_rank() const { return 1; }
  std::size_t value_dimension(std::size_t i) const { return 3;}
};

class FullBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary;
  }
};

class LateralBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return near(x[0], LEFT) || near(x[0], RIGHT);
  }
};

class BoundaryData : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = x[0] * 0.3;  // Strong lateral compression
    values[1] = x[1];
    values[2] = 0;
  }
  std::size_t value_rank() const { return 1; }
  std::size_t value_dimension(std::size_t i) const { return 3;}
};


///  Projects a GenericFunction, which should be in a P^3x2 space onto
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


double
discrete_energy(double alpha,
                std::shared_ptr<const GenericMatrix> Ao,  /* Matrix T^t S T */
                const Function& y, /* solution */
                const Function& L  /* Force    */)
{
  Vector tmp;
  Ao->init_vector(tmp, 0);

  /// Compute the quadratic term of the energy using the system matrix:
  Ao->mult(*(y.vector()), tmp);
  double energy = 0.5*alpha * y.vector()->inner(tmp);

  /// Compute the force term:
  
  // FIXME: this should be the integral of the nodal interpolant,
  // i.e. $\sum_{y \in N_h } f(z) \cdot y(z) \int \phi_z d x$, where
  // \phi_z is the scalar nodal basis function coming from the
  // evaluation linear form. However we cannot compute these integrals
  // due to missing implementation in uflacs (in particular the
  // hermite mapping is missing in
  // apply_single_function_pullbacks.py). Because they are constant,
  // we just ignore them. 

  const auto& v1 = y.vector();
  const auto& v2 = L.vector();
  assert(v1->size() == v2->size());
  // take only coefficients for evaluations
  for (int j = 0; j < v1->size(); j+=3) {
    double val1(0), val2(0);
    v1->get_local(&val1, 1, &j);
    v2->get_local(&val2, 1, &j);
    energy += val1 * val2;
  }
  return energy;
}


/// Does the magic.
///
///   mesh: duh.
///   alpha: Factor multiplying the bending energy term
///   tau: time step size
///   max_steps: stop after at most so many iterations
///   eps: stop after the gradient of the solution changes by
///        at most so much (TODO)
/// TODO: allow an adaptive step size policy
int
dostuff(std::shared_ptr<Mesh> mesh, double alpha, double tau,
        int max_steps, double eps=1e-5)
{
  KirchhoffAssembler assembler;
  Assembler rhs_assembler;
  PETScLUSolver solver("mumps");

  auto W3 = std::make_shared<NLK::Form_dkt_FunctionSpace_0>(mesh);
  auto T3 = std::make_shared<NLK::Form_p26_FunctionSpace_0>(mesh);

  std::cout << "Running on a mesh with " << mesh->num_cells()
            << " cells.\n";
  std::cout << "FE space has " << W3->dim() << " dofs.\n";
  std::cout << "Using alpha = " << alpha << ", tau = " << tau << ".\n";

  // The stiffness matrix includes the condition for the nodes on the
  // Dirichlet boundary to be zero. This ensures that the updates
  // during gradient descent don't change the values of the initial
  // condition, which should fulfill the BC
  auto bdry = std::make_shared<LateralBoundary>();
  auto zero = std::shared_ptr<Function>
    (std::move(project_dkt(std::make_shared<Constant>(0.0, 0.0, 0.0),
                           W3)));
  DirichletBC bc(W3, zero, bdry);

  Table table("Compute times");

  std::cout << "Projecting initial data onto W^3... ";
  tic();
  // Initial data: careful that it fulfils the BCs.
  auto y0 = project_dkt(std::make_shared<BoundaryData>(), W3);
  table("Projection of data", "time") = toc();
  std::cout << "Done.\n";
  
  auto& v = *(y0->vector());
  round_zeros(v);  // This modifies y0
  NLK::dump_full_tensor(v, 4, "y0.data");

  ////////////////////////////////////////////////////////////////////
  // Assembly of system matrix

  /// Top right and lower left blocks
  std::cout << "Initialising constraint... ";
  IsometryConstraint Bk(*W3);
  std::cout << "Done.\n";

  std::cout << "Populating constraint... ";
  tic();
  Bk.update_with(*y0);
  table("Constraint updates", "time") = toc();
  std::cout << "Done.\n";

  /// Upper left block (constant)
  auto A = std::make_shared<Matrix>();
  std::cout << "Assembling bilinear form... ";
  NLK::Form_dkt a(W3, W3);
  NLK::Form_p26 p26(T3, T3);
  tic();
  assembler.assemble(*A, a, p26);
  auto Ao = A->copy();   // Store copy to compute the RHS later,
  *A *= 1 + alpha*tau;   // because we transform A here
  bc.apply(*A);
  table("Assembly", "time") = toc();
  NLK::dump_full_tensor(A, 12, "A.data");
  std::cout << "Done.\n";

  /// Lower right block
  auto paddingMat = Bk.get_padding();
  NLK::dump_full_tensor(*paddingMat, 12, "D.data");
  
  auto block_Mk = std::make_shared<BlockMatrix>(2, 2);
  block_Mk->set_block(0, 0, A);
  block_Mk->set_block(1, 0, Bk.get());
  block_Mk->set_block(0, 1, Bk.get_transposed());
  block_Mk->set_block(1, 1, paddingMat);
  // This requires that the nonzeros for the blocks be already set up
  BlockMatrixAdapter Mk(block_Mk); 
  Mk.read(0,0);  // Read in A ...
  Mk.read(1,1);  // and padding, we read the rest in the loop

  ////////////////////////////////////////////////////////////////////
  // Assembly / setup of RHS
  
  auto zeroVec = std::make_shared<Vector>();
  // *zeroVec can hold a product Ax for some x
  paddingMat->init_vector(*zeroVec, 0);

  std::cout << "Projecting force onto W^3... ";
  tic();
  auto force = std::make_shared<Force>();
  auto f = std::shared_ptr<const Function>(std::move(project_dkt(force,
                                                                 W3)));
  table("Projection of data", "time") =
    table.get_value("Projection of data", "time") + toc();
  std::cout << "Done.\n";

  std::cout << "Assembling force vector... ";
  NLK::Form_force l(W3);
  Function L(W3);
  tic();
  l.f = f;
  rhs_assembler.assemble(*(L.vector()), l);
  table("Assembly", "time") = toc();
  std::cout << "Done.\n";

  // The content of the first block in Fk is set in the loop using the
  // assembled force vector and the copy Ao of the stiffness matrix.
  // The second is always zero.
  auto top_Fk = std::make_shared<Vector>();
  A->init_vector(*top_Fk, 0); //  *top_Fk = Ax for some x
  auto block_Fk = std::make_shared<BlockVector>(2);
  block_Fk->set_block(0, top_Fk);
  block_Fk->set_block(1, zeroVec);
  BlockVectorAdapter Fk(block_Fk);

  ////////////////////////////////////////////////////////////////////
  // Setup system solution at step k
  
  // The first block is the update for the deformation y_{k+1},
  // the second is ignored.
  auto dtY = std::make_shared<Vector>();
  auto ignored = std::make_shared<Vector>();
  A->init_vector(*dtY, 0);
  paddingMat->init_vector(*ignored, 0);
  auto block_dtY_L = std::make_shared<BlockVector>(2);
  block_dtY_L->set_block(0, dtY);
  block_dtY_L->set_block(1, ignored);
  BlockVectorAdapter dtY_L(block_dtY_L);
  
  Function y(*y0);  // Deformation y_{k+1}, begin with initial condition

  ////////////////////////////////////////////////////////////////////
  // Main loop
  
  bool stop = false;
  int step = 0;
  table("RHS computation", "time") = 0;
  table("Solution", "time") = 0;
  table("Stopping condition", "time") = 0;
  Vector tmp;  // intermediate value for norm computations
  Ao->init_vector(tmp, 0);
  std::vector<double> energy_values;
  while (! stop && ++step <= max_steps)
  {
    std::cout << "\n## Step " << step << " ##\n\n";
    std::cout << "Computing RHS... ";
    tic();
    // This isn't exactly elegant...
    Ao->mult(*(y.vector()), *top_Fk);  // Careful! use the copy Ao
    *top_Fk *= -alpha;
    *top_Fk += *(L.vector());
    bc.apply(*top_Fk);
    Fk.read(0);
    table("RHS computation", "time") =
      table.get_value("RHS computation", "time") + toc();
    std::cout << "Done.\n";
    NLK::dump_full_tensor(Fk.get(), 12, "Fk.data");
    
    std::cout << "Updating discrete isometry constraint... ";
    tic();
    Bk.update_with(y);
    Mk.read(0,1);  // This is *extremely* inefficient. At least I
    Mk.read(1,0);  // could update B in place inside Mk.
    table("Constraint updates", "time") =
      table.get_value("Constraint updates", "time") + toc();
    std::cout << "Done.\n";
    NLK::dump_full_tensor(Bk.get(), 12, "Bk.data");
    
    std::cout << "Solving... ";
    tic();
    solver.solve(*(Mk.get()), *(dtY_L.get()), *(Fk.get()));
    dtY_L.write(0);  // Update block_dtY_L(0) back from dty_L, i.e. dtY
    table("Solution", "time") =
      table.get_value("Solution", "time") + toc();
    std::cout << "Done with norm = "
              << norm(*(dtY_L.get())) << "\n";
    NLK::dump_full_tensor(dtY_L.get(), 12, "dtY_L.data");

    std::cout << "Testing whether we should stop... ";
    tic();
    Ao->mult(*dtY, tmp);
    auto nr = std::sqrt(tmp.inner(*dtY));
    std::cout << "norm of \\nabla theta_h dtY =  " << nr << "\n";
    stop = nr < eps || nr > 1;  // FIXME: if nr > 1 we are diverging...
    table("Stopping condition", "time") =
      table.get_value("Stopping condition", "time") + toc();

    y.vector()->axpy(tau, *dtY);  // y = y + tau*dty
    
    energy_values.push_back(discrete_energy(alpha, Ao, y, L));
    std::cout << "Energy = " << energy_values.back() << "\n";
    NLK::dump_full_tensor(y.vector(), 12, "yk.data");
  }
  NLK::dump_raw_matrix(energy_values, 1, energy_values.size(), 14,
                       "energy.data", true, true);
  NLK::dump_full_tensor(Mk.get(), 12, "Mk.data");
  
  // info(table);  // outputs "<Table of size 5 x 1>"
  std::cout << table.str(true) << std::endl;

  // Save solution in VTK format
  File file("solution.pvd");
  file << y;
  
  return 0;   // mpirun expects 0 for success
}

int
main(int argc, char** argv)
{
  //// Default options
  int m = 4, n = 4;
  double alpha = 1.0;
  double tau = 0.7;
  int max_steps = 24;
  double eps = 1e-6;
  int pause = 0;
  std::string diagonal = "right";
  std::string test = "none";
  
  bool help = false;
  sweet::Options opt(argc, const_cast<char**>(argv),
                     "Nonlinear Kirchhoff model on the unit square.");
  opt.get("-h", "--help", "Show this help", help);
  opt.get("-v", "--verbose", "Debug verbosity level, 0: no output, ... 3: write data files with vectors and matrices", NLK::DEBUG);
  opt.get("-m", "--num_vertical", "number of vertical subdivisions", m);
  opt.get("-n", "--num_horizontal", "number of horizontal subdivisions", n);
  opt.get("-d", "--diagonal", "Direction of diagonals: \"left\", \"right\", \"left/right\", \"crossed\"", diagonal);
  opt.get("-a", "--alpha", "alpha", alpha);
  opt.get("-t", "--tau", "timestep *scaling* wrt. minimal cell size", tau);
  opt.get("-x", "--max_steps", "Maximum number of time steps", max_steps);
  opt.get("-e", "--eps_stop", "Stopping threshold.", eps);
  opt.get("-p", "--pause", "Pause each worker for so many seconds before starting, in order to attach a debugger", pause);
  opt.get("-T", "--test", "Run the specified test", test);
  if (opt.help_requested())
    return 1;

  if (test == "dofs") {
    std::cout << "Running test " << test << "...\n";
    NLK::DEBUG = 3;
    return test_dofs();
  } else if (test == "blockvector") {
    std::cout << "Running test " << test << "...\n";
    NLK::DEBUG = 3;
    return test_BlockVectorAdapter();
  }

  // else...
  
  auto mesh = std::make_shared<RectangleMesh>(MPI_COMM_WORLD,
                                              Point (LEFT, BOTTOM),
                                              Point (RIGHT, TOP),
                                              m, n, diagonal);
  
  // In the paper the triangulation consists of halved squares and tau
  // is 2^{-1/2} times the length of the sides i.e. hmin() in our case
  tau *= mesh->hmin();

  /// A trick from the MPI FAQ
  /// (https://www.open-mpi.org/faq/?category=debugging)
  //
  // This code will output a line to stdout outputting the name of the
  // host where the process is running and the PID to attach to. It
  // will then spin on the sleep() function forever waiting for you to
  // attach with a debugger. Using sleep() as the inside of the loop
  // means that the processor won't be pegged at 100% while waiting
  // for you to attach.  Once you attach with a debugger, go up the
  // function stack until you are in this block of code (you'll likely
  // attach during the sleep()) then set the variable pause to a nonzero
  // value. With GDB, the syntax is: (gdb) set var pause = 0. Then set a
  // breakpoint after your block of code and continue execution until
  // the breakpoint is hit. Now you have control of your live MPI
  // application and use the full functionality of the debugger.  You
  // can even add conditionals to only allow this "pause" in the
  // application for specific MPI processes (e.g., MPI_COMM_WORLD rank
  // 0, or whatever process is misbehaving).
  
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    std::cout << "PID " << getpid() << " on " << processor_name
              << " (rank " << world_rank << ") ready for attach\n" << std::flush;
    while (pause > 0)
        sleep(pause);
  }
  
  return dostuff(mesh, alpha, tau, max_steps, eps);
}

