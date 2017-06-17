#ifndef __BLOCK_MATRIX_ADAPTER_H
#define __BLOCK_MATRIX_ADAPTER_H

#include <memory>
#include <vector>

namespace dolfin {

  class BlockMatrix;
  class PETScMatrix;
  
  /// Ideally I'd implmement the interface of a GenericLinearOperator,
  /// but I don't know how hard that could become if solve() is to be
  /// supported.  For now I just build a bigger matrix using the
  /// sparsity patterns of the blocks and copying the data.
  class BlockMatrixAdapter
  {
    std::shared_ptr<const BlockMatrix> _AA;   // Block matrix
    std::shared_ptr<PETScMatrix> _A;          // Flat matrix
    std::vector<std::size_t> _row_offsets;
    std::vector<std::size_t> _col_offsets;

    std::size_t _nrows, _ncols;
  public:
    BlockMatrixAdapter(std::shared_ptr<const BlockMatrix> AA)
      : _AA(AA) { rebuild(); }

    /// Initialise offsets and patterns of non zeros from blocks in
    /// the BlockMatrix
    void rebuild();
    /// Read the contents of the blocks into the flattened Matrix
    void read(int i, int j);

    const PETScMatrix& get() const { return *_A; }
    PETScMatrix& get() { return *_A; }
  };  
}


#endif // __BLOCK_MATRIX_ADAPTER_H
