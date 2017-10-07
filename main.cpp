#include <iostream>
#include <memory>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <vector>
#include <tuple>
#include <cmath>
#include <unistd.h>
#include <dolfin.h>

#include "NonlinearKirchhoff.h"
#include "IsometryConstraint.h"
#include "KirchhoffAssembler.h"
#include "BlockMatrixAdapter.h"
#include "BlockVectorAdapter.h"
#include "output.h"
#include "dkt_utils.h"
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
    values[2] = 1e-5;
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
    return on_boundary && (near(x[0], LEFT) || near(x[0], RIGHT));
  }
};

class InitialData : public DiffExpression
{
public: 
  void eval(Array<double>& values, const Array<double>& x) const
  {
    const double pi6 = M_PI / 6;
    const double pi3 = M_PI / 3;
    const double pi2 = M_PI / 2;
    
    double t = x[0];
    values[1] = x[1];

    if (-2.0 <= t && t < -2.0 + pi6) {
      values[0] = -2.0/3.0 + (1.0/3) * std::cos(3.0*(t+2.0)-pi2);
      values[2] =  1.0/3.0 + (1.0/3) * std::sin(3.0*(t+2.0)-pi2);
    } else if (-2.0 + pi6 <= t && t < -pi6) {
      values[0] = -1.0/3.0;
      values[2] =  1.0/3.0 + t - (-2.0 + pi6);
    } else if (-pi6 <= t && t < pi6) {
      values[0] =               0 + (1.0/3.0) * std::cos(M_PI - 3.0*(t+pi6));
      values[2] = 1.0/3.0+2.0-pi3 + (1.0/3.0) * std::sin(3.0*(t+pi6));
    } else if (pi6 <= t && t < 2.0 - pi6) {
      values[0] = 1.0/3.0;
      values[2] = 1.0/3.0 + 2.0 - pi3 - t + pi6;
    } else if (2.0 - pi6 <= t && t <= 2.0 ) {
      values[0] = 2.0/3.0 + (1.0/3.0)*std::cos(3.0*(t-2.0+pi6)+M_PI);
      values[2] = 1.0/3.0 + (1.0/3.0)*std::sin(3.0*(t-2.0+pi6)+M_PI);
    }      
  }

  /// Gradient is stored in row format: f_11, f_12, f_21, f_22, f_31, f_32
  void gradient(Array<double>& grad, const Array<double>& x) const
  {
    const double pi6 = M_PI / 6;
    const double pi3 = M_PI / 3;
    const double pi2 = M_PI / 2;
    
    double t = x[0];
    grad[1] = grad[2] = grad[5] = 0;
    grad[3] = 1;
    if (-2.0 <= t && t < -2.0 + pi6) {
      grad[0] = - std::sin(3.0*(t+2.0)-pi2);
      grad[4] =   std::cos(3.0*(t+2.0)-pi2);
    } else if (-2.0 + pi6 <= t && t < -pi6) {
      grad[0] = 0;
      grad[4] = 1;
    } else if (-pi6 <= t && t < pi6) {
      grad[0] = std::sin(M_PI - 3.0*(t+pi6));
      grad[4] = std::cos(3.0*(t+pi6));
    } else if (pi6 <= t && t < 2.0 - pi6) {
      grad[0] = 0;
      grad[4] = -1;
    } else if (2.0 - pi6 <= t && t <= 2.0 ) {
      grad[0] = - std::sin(3.0*(t-2.0+pi6)+M_PI);
      grad[4] = std::cos(3.0*(t-2.0+pi6)+M_PI);
    }
  }
    
  std::size_t value_rank() const { return 1; }
  std::size_t value_dimension(std::size_t i) const { return 3; }
};

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
  energy += dkt_inner(y, L);
  return energy;
}

void
hack_values(Function& y)
{
  auto W3 = y.function_space();
  auto mesh = W3->mesh();
  std::vector<la_index> dxdofs, dydofs;
  auto v2d = vertex_to_dof_map(*W3);
  for (int sub=0; sub < 3; ++sub) {
    // auto dm = (*W3)[sub]->dofmap();
    // The following should be a process-local index
    for (VertexIterator vit(*mesh); !vit.end(); ++vit) {
      auto idx = static_cast<la_index>(vit->index());
      auto dofdx = v2d[9*idx + 3*sub + 1];
      auto dofdy = v2d[9*idx + 3*sub + 2];
      dxdofs.push_back(dofdx);
      dydofs.push_back(dofdy);
    }
  }
  std::vector<double> dx(dxdofs.size()), dy(dydofs.size());
  size_t ctr = 0;
  std::generate(dx.begin(), dx.end(), [&ctr]() { return (ctr++ % 3 == 0) ? 1.0 : 0.0; });
  ctr = 0;
  std::generate(dy.begin(), dy.end(), [&ctr]() { return (ctr++ % 3 == 1) ? 1.0 : 0.0; });
  y.vector()->set(dx.data(), dxdofs.size(), dxdofs.data());
  y.vector()->set(dy.data(), dydofs.size(), dydofs.data());
}

/// Hack to initialise the boundary condition to what we need
void
hack_boundary_values(const SubDomain& subdomain, Function& u)
{
  /// Extract info and ensure the mesh (connectivity) has been initialised
  auto W = u.function_space();
  // for (int i=0; i < 3; ++i)
  //   dms.push_back(std::move(W[i]->dofmap()));
  // auto dm = W->dofmap();
  auto mesh = W->mesh();
  auto D = mesh->topology().dim();
  mesh->init(D);
  mesh->init(D - 1);
  mesh->init(D - 1, D);
  
  /// Build mesh function using the SubDomain and iterate facets using it
  /// (TODO: there must be some other way)
  FacetFunction<std::size_t> subdomains(mesh, 1);
  subdomain.mark(subdomains, 42, false);  // don't check midpoints

  /// This is *very* inefficient...
  for (int sub = 0; sub < 3; ++sub) {
    auto dm = (*W)[sub]->dofmap();
    for (FacetIterator facet(*mesh); !facet.end(); ++facet) {
      if (subdomains[*facet] != 42)   // nothing to do
        continue;

      // Get cell to which facet belongs and local index.
      dolfin_assert(facet.num_entities(D) > 0);
      auto cell_index = facet->entities(D)[0];
      Cell cell(*mesh, cell_index);
      auto facet_local_index = cell.index(*facet);

      // local-global dof mapping for cell
      auto cell_dofs = dm->cell_dofs(cell.index());
      // local-local dof mapping of dofs on the facet
      std::vector<std::size_t> local_facet_dofs;
      dm->tabulate_facet_dofs(local_facet_dofs, facet_local_index);
      // local-global mapping for facet dofs
      // Vector::get() doesn't like size_t... duh!
      std::vector<la_index> facet_dofs(local_facet_dofs.size());
      std::transform(local_facet_dofs.begin(), local_facet_dofs.end(),
                     facet_dofs.begin(),
                     [&cell_dofs] (std::size_t& dof) {
                       return static_cast<la_index>(cell_dofs[dof]);
                     });
      // Test values for facet
      la_index numrows = facet_dofs.size();
      std::vector<double> uvalues(numrows);
      u.vector()->get(uvalues.data(), numrows, facet_dofs.data());
      for(auto i = 0; i < numrows; ++i) {  // 6 values per facet and subspace
        // std::cout << "ldof " << i << ", vdof " << i % 3  << ", gdof "
        //           << facet_dofs[i] << "\n";
        switch (i % 3) {
        case 0:
          if (sub == 0)
            uvalues[i] = uvalues[i] < 0 ? -0.6 : 0.6;
              break;
        case 1: uvalues[i] = (sub == 0) ? 1.0 : 0.0; break;
        case 2: uvalues[i] = (sub == 1) ? 1.0 : 0.0; break;
        }
      }
      u.vector()->set(uvalues.data(), numrows, facet_dofs.data());
    }
  }
}

bool
equal_at_p(const SubDomain& subdomain, const Function& u, const Function& v)
{
  /// Extract info and ensure the mesh (connectivity) has been initialised
  auto W = v.function_space();
  auto dm = W->dofmap();
  auto mesh = W->mesh();
  auto D = mesh->topology().dim();
  W->mesh()->init(D);
  W->mesh()->init(D - 1);
  W->mesh()->init(D - 1, D);

  /// Build mesh function using the SubDomain and iterate facets using it
  /// (TODO: there must be some other way)
  FacetFunction<std::size_t> subdomains(mesh, 1);
  subdomain.mark(subdomains, 42, false);  // don't check midpoints

  for (FacetIterator facet(*mesh); !facet.end(); ++facet) {
    if (subdomains[*facet] != 42)   // nothing to do
      continue;

    // Get cell to which facet belongs and local index.
    dolfin_assert(facet.num_entities(D) > 0);
    const auto cell_index = facet->entities(D)[0];
    const Cell cell(*mesh, cell_index);
    const auto facet_local_index = cell.index(*facet);

    // local-global dof mapping for cell
    const auto cell_dofs = dm->cell_dofs(cell.index());
    // local-local dof mapping of dofs on the facet
    std::vector<std::size_t> local_facet_dofs;
    dm->tabulate_facet_dofs(local_facet_dofs, facet_local_index);
    // Vector::get() doesn't like size_t... duh!
    std::vector<la_index> facet_dofs(local_facet_dofs.size());
    // local-global mapping for facet dofs
    std::transform(local_facet_dofs.begin(), local_facet_dofs.end(),
                   facet_dofs.begin(),
                   [&cell_dofs] (std::size_t& dof) {
                     return static_cast<la_index>(cell_dofs[dof]);
                   });
    // Test values for facet
    la_index numrows = facet_dofs.size();
    std::vector<double> uvalues(numrows), vvalues(numrows);
    u.vector()->get(uvalues.data(), numrows, facet_dofs.data());
    v.vector()->get(vvalues.data(), numrows, facet_dofs.data());
    if (uvalues != vvalues)
      return false;
  }
  return true;
}

/// Does the magic.
///
///   mesh: duh.
///   alpha: Factor multiplying the bending energy term
///   tau: time step size
///   max_steps: stop after at most so many iterations
///   eps: stop after the gradient of the solution changes by at most so much
///   adaptive_steps: change time step size every so many steps
///   adaptive_factor: change step size by this factor
int
dostuff(std::shared_ptr<Mesh> mesh, double alpha, int max_steps, double eps,
        double tau, double adaptive_steps, double adaptive_factor)
{
  KirchhoffAssembler assembler;
  Assembler rhs_assembler;
  PETScLUSolver solver("mumps");

  auto W3 = std::make_shared<NLK::Form_dkt_FunctionSpace_0>(mesh);
  auto T3 = std::make_shared<NLK::Form_p26_FunctionSpace_0>(mesh);

  std::cout << "Running on a mesh with " << mesh->num_cells()
            << " cells.\n"
            << "FE space has " << W3->dim() << " dofs.\n"
            << "Using alpha = " << alpha << ", tau = " << tau << ".\n"
            << "Scaling timestep every " << static_cast<int>(adaptive_steps)
            << " by " << adaptive_factor << ".\n";
            
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
  auto y0 = eval_dkt(std::make_shared<InitialData>(), W3);
  // CAREFUL!! The discontinuities introduced by this propagate and
  // crumple the solution!! Or so it seems...
  // hack_boundary_values(*bdry, *y0);
  // hack_values(*y0);
  table("Projection of data", "time") = toc();
  std::cout << "Done.\n";
  
  auto& v = *(y0->vector());
  round_zeros(v);  // This modifies y0
  NLK::dump_full_tensor(v, 6, "y0.data", true, true);

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
  // NLK::dump_full_tensor(*paddingMat, 12, "D.data");
  
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
  
#if 0
  //////////////////////////////////////////////////////////////////////////////
  // FIXME!!!!  instead of fixing the RHS with the integral of the
  // nodal basis funtions, this screws the third component of the
  // solution, setting it to 0!!?!?! It could be that the ordering
  // I've assumed for the global dofs is wrong.

  // Computing integrals of CG1 basis functions
  auto CG = std::make_shared<NLK::Form_beta_FunctionSpace_0>(mesh);
  NLK::Form_beta beta(CG);
  Function integrals(CG);
  rhs_assembler.assemble(*(integrals.vector()), beta);
  std::cout << "Computed CG1 integrals.\n";
  
  // Copy values onto real RHS:
  std::vector<la_index> from, to;
  auto v2d_DKT = vertex_to_dof_map(*W3);
  auto v2d_CG = vertex_to_dof_map(*CG);
  for (VertexIterator v(*(W3->mesh())); !v.end(); ++v) {
    // The following should be a process-local index
    auto idx = static_cast<la_index>(v->index());
    auto dofcg = v2d_CG[idx];
    for (int sub = 0; sub < 3; ++sub) {     // iterate over the 3 subspaces
      auto dofdkt = v2d_DKT[9*idx + 3*sub];
      from.push_back(dofcg);
      to.push_back(dofdkt);
    }
  }
  // std::vector<double> saved(to.size());
  // L.vector()->get_local(saved.data(), to.size(), to.data());
  std::vector<double> data(from.size());
  integrals.vector()->get_local(data.data(), from.size(), from.data());
  for(auto& d: data) d *= 1e-5;  /// HACK! scale with force
  L.vector()->set_local(data.data(), to.size(), to.data());
#endif
  
  // std::vector<double> diffs(data.size());
  // std::transform(data.begin(), data.end(), saved.begin(), diffs.begin(),
  //                [](double x, double y) { return std::abs(x-y); });
  // auto normdiffs = std::accumulate(diffs.begin(), diffs.end(), 0.0);
  // std::cout << "Total difference: " << normdiffs << "\n";

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
  DKTGradient DG;
  
  ////////////////////////////////////////////////////////////////////
  // Main loop
  
  File file("solution.pvd");  // Save solution in VTK format
  file << y;   // Save initial condition
  bool stop = false;
  int step = 0, dec_ctr = 0;  // dec_ctr keeps track of tau decreasing
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
    top_Fk->axpy(tau, *(L.vector()));
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
              << std::sqrt(dkt_inner(dtY_L.get(), dtY_L.get(), W3)) << "\n";
    NLK::dump_full_tensor(dtY_L.get(), 12, "dtY_L.data");

    std::cout << "Testing whether we should stop... ";
    tic();
    Ao->mult(*dtY, tmp);
    auto nr = std::sqrt(tmp.inner(*dtY));
    std::shared_ptr<const GenericVector> gr(std::move(DG.apply_vec(T3, W3, dtY)));
    auto nr2 = std::sqrt(dkt_inner(gr, gr, W3));
    std::cout << "norm of \\nabla theta_h dtY =  " << nr << " -- " << nr2 << "\n";
    stop = nr < eps; // || nr > 1;  // FIXME: if nr > 1 we are diverging (?)
    table("Stopping condition", "time") =
      table.get_value("Stopping condition", "time") + toc();

    y.vector()->axpy(tau, *dtY);  // y = y + tau*dty
    std::cout << "Testing boundary conditions... "
              << (equal_at_p(*bdry, y, *y0) ? "OK." : "WRONG!!!")
              << "\n";
    std::cout << "Distance to isometry: "
              << distance_to_isometry(y) << "\n";
    
    double energy = discrete_energy(alpha, Ao, y, L); 
    energy_values.push_back(energy);
    if (step > 2) {
      double prev = *(energy_values.end()-2);
      std::cout << "Energy change: " << 100*(energy - prev)/prev << "% ("
                // check unconditional stability:
                << ((energy + tau*nr*nr <= prev) ? "OK" : "WRONG") << ").\n";
    }

    if(std::floor(step / adaptive_steps) > dec_ctr) {
      dec_ctr = std::floor(step / adaptive_steps);
      tau *= adaptive_factor;
      std::cout << "Corrected tau = " << tau << "\n";
      
      // output current solution
      file << y;
      NLK::dump_full_tensor(y.vector(), 12, "yk.data", true, true);
    }
  }
  NLK::dump_raw_matrix(energy_values, 1, energy_values.size(), 14,
                       "energy.data", true, true);
  NLK::dump_full_tensor(Mk.get(), 12, "Mk.data");

  file << y;  // output last solution
  
  // info(table);  // outputs "<Table of size 5 x 1>"
  std::cout << table.str(true) << std::endl;

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
  std::vector<double> adaptive;  // scale tau every x steps by a factor y
  bool help = false;
  sweet::Options opt(argc, const_cast<char**>(argv),
                     "Nonlinear Kirchhoff model on the unit square.");
  opt.get("-h", "--help",
          "Show this help", help);
  opt.get("-v", "--verbose",
          "Debug verbosity level, 0: no output, ... 3: output vectors and matrices",
          NLK::DEBUG);
  opt.get("-m", "--num_vertical",
          "number of vertical subdivisions", m);
  opt.get("-n", "--num_horizontal",
          "number of horizontal subdivisions", n);
  opt.get("-d", "--diagonal",
          "Direction of diagonals: \"left\", \"right\", \"left/right\", \"crossed\"",
          diagonal);
  opt.get("-a", "--alpha",
          "Constant scaling of the bending energy", alpha);
  opt.get("-t", "--tau",
          "Scaling of the timestep wrt. minimal cell size", tau);
  opt.getMultiple("-c", "--correction",
                  "Adaptive parameters for tau (number of timesteps, multiplier)",
                  adaptive);
  opt.get("-x", "--max_steps",
          "Maximum number of time steps", max_steps);
  opt.get("-e", "--eps_stop",
          "Stopping threshold.", eps);
  opt.get("-p", "--pause",
          "Pause each worker for so many seconds in order to attach a debugger",
          pause);
  opt.get("-s", "--test",
          "Run the specified test (dofs, blockvector, dkt)", test);

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
  } else if (test == "dkt") {
    std::cout << "Running test " << test << "...\n";
    NLK::DEBUG = 3;
    return test_DKT();
  } else if (test != "none") {
    std::cout << "Unknown test type " << test << ".\n";
    return 1;
  }

  // else...
  
  auto mesh = std::make_shared<RectangleMesh>(MPI_COMM_WORLD,
                                              Point (LEFT, BOTTOM),
                                              Point (RIGHT, TOP),
                                              m, n, diagonal);
  
  // In the paper the triangulation consists of halved squares and tau
  // is 2^{-1/2} times the length of the sides i.e. hmin() in our case
  tau *= mesh->hmin();
  // Set default values:
  if (adaptive.size() < 1) adaptive.push_back(100);
  if (adaptive.size() < 2) adaptive.push_back(0.9);
  double adaptive_steps = adaptive[0];  // decrease tau every so many steps
  double adaptive_factor = adaptive[1]; // by such a multiplier
  
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

  return dostuff(mesh, alpha, max_steps, eps, tau,
                 adaptive_steps, adaptive_factor);
}

