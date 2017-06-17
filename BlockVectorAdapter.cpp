#include <stdexcept>
#include <algorithm>
#include <vector>
#include <dolfin.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/la/PETScObject.h>
#include "BlockVectorAdapter.h"

using namespace dolfin;
#define TEST_PETSC_ERROR(__ierr, __funname) \
  if (__ierr != 0) PETScObject::petsc_error(__ierr, __FILE__, __funname);


// BlockVectorAdapter::BlockVectorAdapter(std::shared_ptr<const BlockVector> AA)

/// Rebuilds the aggregated vector and its sparsity pattern from
/// scratch. This is useful only when the size of the blocks
/// changes. To update values in the aggregated vector use
/// update_block() or update()
/// NOTE:
/// "Some algorithms require diagonal entries, so it's sometimes
/// better to preallocate them and put an explicit zero (...) than to
/// skip them"
void
BlockVectorAdapter::assemble()
{
  auto nrows = _VV->size();
  PetscErrorCode ierr;
  
  //// Extract row offsets for the blocks in _VV
  std::size_t l = 0;

  _row_offsets.empty();
  for (auto r = 0; r < nrows; ++r)
  {
    _row_offsets.push_back(l);
    const auto& V = _VV->get_block(r);
    l += V->size(0);
  }
  _nrows = l;   // total number of rows 
 
  // (re)build _V

  // FIXME: all of that getting the ranges and parallel stuff is
  // obviously useless if I create a sequential vector here, but I
  // don't know how to translate later the local offset indices (of
  // the blocks in the BlockVector) into whatever the global ordering
  // is, so this will break badly if run on more than one process
  Vec v;
  ierr = VecCreate(MPI_COMM_WORLD, &v);
  TEST_PETSC_ERROR(ierr, "VecCreate");
  ierr = VecSetType(v, VECMPI);
  TEST_PETSC_ERROR(ierr, "VecCreate");
  ierr = VecSetSizes(v, PETSC_DECIDE, _nrows);
  TEST_PETSC_ERROR(ierr, "VecSetSizes");

  // FIXME: I should ensure that there are no references left around
  _V = std::make_shared<PETScVector>(v);
}


void
BlockVectorAdapter::read(int i)
{
  auto V = _VV->get_block(i);
  auto roff = _row_offsets.at(i);
  // auto range = V->local_range(0);    // local row (0th dim) range

  // Extract local values
  auto nrows = V->local_size();
  std::vector<double> values(nrows);
  std::vector<la_index> rows(nrows);
  std::iota(rows.begin(), rows.end(), 0);
  V->get_local(values.data(), nrows , rows.data());

  // Set values in aggregated vector NOTE that this assumes that we
  // can just offset local indices of each subblock to obtain global
  // indices for _V. This should be true as long as it is a sequential
  // vector, I guess.
  std::transform(rows.begin(), rows.end(), rows.begin(),
                 [&roff] (std::size_t x) { return x + roff; });
  _V->set(values.data(), nrows, rows.data());
  _V->apply("insert");
}


/// FIXME: this is EXTREMELY inefficient. No need to rebuild all row
/// indices on each call. Probably no need to copy anything either if
/// we are running locally.
void
BlockVectorAdapter::write(int i)
{
  auto V = _VV->get_block(i);
  auto roff = _row_offsets.at(i);
  auto nrows = V->local_size();
  std::vector<double> values(nrows);
  std::vector<la_index> rows(nrows);
  std::iota(rows.begin(), rows.end(), roff);
  _V->get_local(values.data(), nrows , rows.data());
  std::iota(rows.begin(), rows.end(), 0);
  V->set_local(values.data(), nrows, rows.data());
}
