#include <stdexcept>
#include <dolfin.h>
#include <dolfin/la/PETScMatrix.h>
#include "BlockMatrixAdapter.h"
#include <petscmat.h>

using namespace dolfin;

// BlockMatrixAdapter::BlockMatrixAdapter(std::shared_ptr<const BlockMatrix> AA)

// global index of non zero row -> vec of global indices of non zero cols:
typedef std::map<PetscInt, std::vector<PetscInt>> nz_data_t;

/// UNTESTED: Extracts sparsity pattern from a PETScMatrix.
/// out param: nzentries (std::map...)
/// Returns total number of non zeros in M
std::size_t
extract_nonzeros(const PETScMatrix& M, nz_data_t& nzentries,
                 PetscInt roffset, PetscInt coffset)
{
  std::size_t nnz = 0;

  // auto local_range = M.local_range(0);    // local row (0th dim) range
  // do stuff
  auto m = M.mat();
  PetscInt rstart = 0, rend = 0, ncols = 0;
  MatGetOwnershipRange(m, &rstart, &rend);
  const PetscInt** cols;  // const ptr to buffer, allocated by PETSc
  for (auto irow = rstart; irow < rend; ++irow)
  {  
    MatGetRow(m, irow, &ncols, cols, NULL);
    if (ncols > 0)
    {
      std::vector<PetscInt> nzcols(ncols);
      std::cout << "Found row " << irow << " with "
                << ncols << " nonzero columns.\n"
                << "nzcols has size " << nzcols.size() << ".\n";
      for (int i = 0; i < ncols; ++i)
        nzcols.push_back(*((*cols)+i) + coffset);

      auto nzrow = nzentries.find(irow + roffset);
      if (nzrow != nzentries.end())
        // FIXME: Maybe I should insert in the larger one and swap, to
        // avoid copying too many entries...
        nzrow->second.insert(nzrow->second.begin(), nzcols.begin(), nzcols.end());
      else
        // FIXME: will this move the std::vector or copy it?
        // Should/could I use forwarding, whatever other stuff there is?
        nzentries.emplace(std::make_pair(irow + roffset, std::move(nzcols)));
      nnz += ncols;
    }
    MatRestoreRow(m, irow, &ncols, cols, NULL);   // free memory in *cols
  }
  return nnz;
}

/// Rebuilds the aggregated matrix and its sparsity pattern from
/// scratch. This is useful only when the size of the blocks
/// changes. To update values in the aggregated matrix use
/// update_block() or update()
void
BlockMatrixAdapter::rebuild()
{
  auto nrows = _AA->size(0);
  auto ncols = _AA->size(1);
  
  ///// Extract row and column offsets (in the aggregated matrix) for the blocks
  std::size_t w = 0, h = 0;

  _row_offsets.empty();
  for (auto i = 0; i < nrows; ++i)
  {
    _row_offsets.push_back(h);
    const auto& B = _AA->get_block(i, 0);
    h += B->size(0);
  }
  _nrows = h;   // total number of rows 
  
  _col_offsets.empty();
  for (auto j = 0; j < ncols; ++j)
  {
    _col_offsets.push_back(w);
    const auto& B = _AA->get_block(0, j);
    w += B->size(1);
  }      
  _ncols = w;  // total number of cols

  // iterate block rows, cols to extract sparsity
  nz_data_t nzentries;
  std::size_t nnz = 0;
  for (int i = 0; i < nrows; ++i)
  {
    for (int j = 0; j < ncols; ++j)
    {
      const auto& B = _AA->get_block(i, j);
      nnz += extract_nonzeros(as_type<const PETScMatrix>(*B), nzentries, _row_offsets[i], _col_offsets[j]);
    }
  }


  // "Some algorithms require diagonal entries, so it's sometimes
  // better to preallocate them and put an explicit zero (...) than to
  // skip them"


  // rebuild _A

  // This has length nrows+1, so the length of the last row is known
  std::vector<PetscInt> row_indices_in_col_indices;
  row_indices_in_col_indices.reserve(_nrows+1);
  std::vector<PetscInt> col_indices;
  col_indices.reserve(nnz);

  row_indices_in_col_indices.push_back(0);
  for(int r = 1; r <= _nrows; ++r)
  {
    try {
      const auto& row = nzentries.at(r-1);
      auto cols_in_row = row.size();
      col_indices.insert(col_indices.end(), row.begin(), row.end());
      // Set index to next row
      row_indices_in_col_indices.push_back(row_indices_in_col_indices.back()
                                           + cols_in_row);
    } catch (const std::out_of_range& e) {
      row_indices_in_col_indices.push_back(row_indices_in_col_indices.back());
    }
  }

  // FIXME: all of that getting the ranges and parallel stuff is obviously
  // useless if I create a sequential matrix here...
  
  Mat m;
  MatCreate(MPI_COMM_WORLD, &m);
  MatSetType(m, MATSEQAIJ);
  MatSetSizes(m, PETSC_DECIDE, PETSC_DECIDE, _nrows, _ncols);
  // This copies the index dat (which is ok, we seldom call rebuild())
  // TODO: I could use the data from the matrices
  MatSeqAIJSetPreallocationCSR(m, row_indices_in_col_indices.data(),
                               col_indices.data(), NULL);

  MatAssemblyBegin(m, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(m, MAT_FINAL_ASSEMBLY);
}


void
BlockMatrixAdapter::update(int i, int j)
{
  auto B = _AA->get_block(i,j);
  la_index ioff = 0, joff = 0;
  
}
