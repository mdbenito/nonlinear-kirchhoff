#include <stdexcept>
#include <algorithm>
#include <cassert>
#include <dolfin.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/log/Progress.h>
#include <dolfin/log/Logger.h>
#include "BlockMatrixAdapter.h"
#include <petscmat.h>
#include <dolfin/la/PETScObject.h>

using namespace dolfin;
#define TEST_PETSC_ERROR(__ierr, __funname) \
  if (__ierr != 0) PETScObject::petsc_error(__ierr, __FILE__, __funname);


// BlockMatrixAdapter::BlockMatrixAdapter(std::shared_ptr<const BlockMatrix> AA)

// global index of non zero row -> vec of global indices of non zero cols:
typedef std::map<PetscInt, std::vector<PetscInt>> nz_data_t;

/// UNTESTED: Extracts sparsity pattern from a PETScMatrix.
/// out param: nzentries (std::map...)
/// Returns total number of non zeros in M
///
/// IMPORTANT! We assume that we are called "from left to right" in
/// the block matrix. That is: we process block (i,j) *after* block
/// (i,j-1). This is important for the column indices returned in
/// nzentries to be ordered.
std::size_t
extract_nonzeros(const PETScMatrix& M, nz_data_t& nzentries,
                 PetscInt roffset, PetscInt coffset)
{
  // std::cout << "Extracting nonzeros...\n";
  std::size_t nnz = 0;

  // auto local_range = M.local_range(0);    // local row (0th dim) range
  auto m = M.mat();
  PetscInt rstart = 0, rend = 0, ncols = 0;
  PetscErrorCode ierr;
  ierr = MatGetOwnershipRange(m, &rstart, &rend);
  TEST_PETSC_ERROR(ierr, "MatGetOwnershipRange");
  
  const PetscInt* cols;  // allocated by PETSc
  std::vector<PetscInt> nzcols(M.size(1), -1);
  for (auto irow = rstart; irow < rend; ++irow)
  {
    // std::cout << "  Retrieving row " << irow << "...";
    ierr = MatGetRow(m, irow, &ncols, &cols, NULL);
    TEST_PETSC_ERROR(ierr, "MatGetRow");

    if (ncols > 0 && cols)
    {
      std::transform(cols, cols+ncols, nzcols.begin(),
                     [&coffset](PetscInt c) { return c + coffset;});
      auto nzrow = nzentries.find(irow + roffset);
      if (nzrow != nzentries.end()) {
        // std::cout << " appending " << ncols << " indices to previous "
        //           << nzrow->second.size() << " ones.";

        // FIXME: Maybe I should insert in the larger one and swap, to
        // avoid copying too many entries...
        nzrow->second.insert(nzrow->second.end(),
                             nzcols.begin(), nzcols.begin()+ncols);
      } else {
        // std::cout << " adding new entry to nzentries with "
        //           << ncols << " indices.";
        // FIXME: will this move the std::vector or copy it?
        // Should/could I use forwarding, whatever other stuff there is?
        nzentries.emplace(std::make_pair(irow + roffset,
                               std::vector<PetscInt>(nzcols.begin(),
                                                     nzcols.begin()+ncols)));
      }
    }
    nnz += ncols;
    // std::cout << "\n";
    // std::cout << " Found " << ncols << " non zero entries.\n";
    ierr = MatRestoreRow(m, irow, &ncols, &cols, NULL);   // free memory in *cols
    TEST_PETSC_ERROR(ierr, "MatRestoreRow");
  }
  return nnz;
}

/// Rebuilds the aggregated matrix and its sparsity pattern from
/// scratch. This is useful only when the size of the blocks
/// changes. To update values in the aggregated matrix use
/// update_block() or update()
/// NOTE:
/// "Some algorithms require diagonal entries, so it's sometimes
/// better to preallocate them and put an explicit zero (...) than to
/// skip them"
void
BlockMatrixAdapter::assemble()
{
  // std::cout << "Assembling flat matrix:\n";
  auto nrows = _AA->size(0);
  auto ncols = _AA->size(1);
  
  //// Extract row and column offsets for the blocks in _A
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

  //// iterate block rows & cols to extract sparsity
  nz_data_t nzentries;
  std::size_t nnz = 0;
  for (int i = 0; i < nrows; ++i)
  {
    for (int j = 0; j < ncols; ++j)
    {
      // std::cout << "Extracting sparsity info from block ("
      //           << i << ", " << j << ")\n";
      const auto& B = _AA->get_block(i, j);
      nnz += extract_nonzeros(as_type<const PETScMatrix>(*B), nzentries,
                              _row_offsets[i], _col_offsets[j]);
    }
  }

  //// (re)build _A
  //// We need to format the sparsity data into CSR for
  // MatMPIAIJSetPreallocationCSR() and MatSeqAIJSetPreallocationCSR()
  // NOTE: it is best to call both functions to avoid crashing when not
  // working in either mode (sequential/parallel)

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

  std::cout << "Preallocating flat AIJ matrix of size "
            << _nrows << " x " << _ncols << " with " << col_indices.size()
            << " entries... ";
  
  assert(row_indices_in_col_indices.size() == _nrows+1);
  assert(row_indices_in_col_indices.back() == col_indices.size());
  
  // std::cout << "\nRow indices in column array:\n";
  // for (auto r: row_indices_in_col_indices)
  //   std::cout << r << ", ";
  // std::cout << "\n\nColumn indices:\n";
  // for (auto c: col_indices)
  //   std::cout << c << ", ";
  // std::cout << "\n";
  
  // FIXME: all of that getting the ranges and parallel stuff is
  // obviously useless if I create a sequential matrix here, but most
  // of the code will break if used in parallel.
  // FIXME: I should use dolfin's GenericMatrix interface instead of
  // forcing a dependency on PETSc
  Mat mat;
  PetscErrorCode ierr;

  ierr = MatCreate(MPI_COMM_WORLD, &mat);
  TEST_PETSC_ERROR(ierr, "MatCreate");
  ierr = MatSetType(mat, MATAIJ);
  TEST_PETSC_ERROR(ierr, "MatSetType");
  ierr = MatSetSizes(mat, _nrows, _ncols, _nrows, _ncols); // PETSC_DECIDE, PETSC_DECIDE, _nrows, _ncols);
  TEST_PETSC_ERROR(ierr, "MatSetSizes");

  // // HACK, TEST
  // PetscInt lm, ln, gm, gn;
  // MatGetLocalSize(mat, &lm, &ln);
  // MatGetSize(mat, &gm, &gn);
  // assert(lm == gm);
  // assert(ln == gn);
  
  // This copies the index data (which is ok, we seldom call rebuild())
  // TODO: I could use the data from the matrices right away if available...
  // FIXME!! I'm using local indices as global and viceversa ALL OVER THE PLACE
  ierr = MatMPIAIJSetPreallocationCSR(mat, row_indices_in_col_indices.data(),
                                      col_indices.data(), NULL);
  TEST_PETSC_ERROR(ierr,"MatMPIAIJSetPreallocationCSR");
  ierr = MatSeqAIJSetPreallocationCSR(mat, row_indices_in_col_indices.data(),
                                      col_indices.data(), NULL);
  TEST_PETSC_ERROR(ierr,"MatSeqAIJSetPreallocationCSR");
  ierr = MatAssemblyBegin(mat, MAT_FLUSH_ASSEMBLY);
  TEST_PETSC_ERROR(ierr,"MatAssemblyBegin");
  ierr = MatAssemblyEnd(mat, MAT_FLUSH_ASSEMBLY);
  TEST_PETSC_ERROR(ierr,"MatAssemblyEnd");

  std::cout << " done.\n";

  // FIXME: I should ensure that there are no references left around
  _A = std::make_shared<PETScMatrix>(mat);
}


void
BlockMatrixAdapter::read(int i, int j)
{
  // std::cout << "Reading from block (" << i << ", " << j << ").\n";
  auto B = _AA->get_block(i, j);
  Mat mB = as_type<const PETScMatrix>(*B).mat();
  auto roff = _row_offsets.at(i), coff = _col_offsets.at(j);
  auto range = B->local_range(0);    // local row (0th dim) range
  const PetscInt* cols;     // allocated by PETSc
  const PetscScalar* vals;  // allocated by PETSc
  PetscInt row, ncols;
  PetscErrorCode ierr;
  std::vector<PetscInt> columns(B->size(1), -1);
  Mat mA = as_type<PETScMatrix>(*_A).mat();
  for (auto r = range.first; r < range.second; ++r)
  {
    /*
    std::vector<std::size_t> columns;
    std::vector<double> values;
    B->getrow(r, columns, values);
    
    std::transform(columns.begin(), columns.end(), columns.begin(),
                   [&coff] (std::size_t x) { return x+coff; });
    std::cout << "Reading row " << r << " into row " << r+roff
              << " with " << values.size() << " values in columns:\n";
    for (auto c: columns)
      std::cout << c << ", ";
    
    _A->setrow(r+roff, columns, values);
    */
    ierr = MatGetRow(mB, r, &ncols, &cols, &vals);
    TEST_PETSC_ERROR(ierr, "MatGetRow");

    // std::cout << "\tReading row " << r << " into row " << r+roff
    //           << " with " << ncols << " values.\n";// in columns: ";
    // for (int i=0; i<ncols; ++i)
    //   std::cout << cols[i] << ", ";
    // std::cout << "\n";

    if (ncols > 0)
    {
      // std::cout << "Applying column offset of " << coff << ".\n"
      //           << "Values are: ";
      // for (int i=0; i<ncols; ++i)
      //   std::cout << vals[i] << ", ";
      std::transform(cols, cols + ncols, columns.begin(),
                     [&coff] (std::size_t x) { return x+coff; });

      // std::cout << "\nAnd offset columns are: ";
      // for (auto c: columns)
      //   std::cout << c << ", ";
      // std::cout << "\n";
      
      row = r+roff;
      ierr = MatSetValues(mA, 1, &row, ncols, columns.data(), vals,
                          INSERT_VALUES);
      TEST_PETSC_ERROR(ierr, "MatSetValues");

      // std::cout << "\tWrote " << ncols << " values.\n";
    }
    ierr = MatRestoreRow(mB, r, &ncols, &cols, &vals);   // free memory in cols, vals
    TEST_PETSC_ERROR(ierr, "MatRestoreRow");
  }
  // _A->apply("insert");
  ierr = MatAssemblyBegin(mA, MAT_FINAL_ASSEMBLY);
  TEST_PETSC_ERROR(ierr, "MatAssemblyBegin");
  ierr = MatAssemblyEnd(mA, MAT_FINAL_ASSEMBLY);
  TEST_PETSC_ERROR(ierr, "MatAssemblyEnd");
}
