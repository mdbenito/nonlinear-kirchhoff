#include <vector>
#include <numeric>
#include <dolfin.h>
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

