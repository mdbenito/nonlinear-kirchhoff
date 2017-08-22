#ifndef __BLOCK_MATRIX_ADAPTER_H
#define __BLOCK_MATRIX_ADAPTER_H

#include <memory>
#include <vector>

namespace dolfin {

  class BlockMatrix;
  class PETScMatrix;

  ///// A BlockMatrixAdapter is a flattened interface to a BlockMatrix.
  /// Ideally I'd implement the interface of a GenericLinearOperator,
  /// but I don't know how hard that could become if solve() is to be
  /// supported. For now I just build a bigger matrix using the
  /// sparsity patterns of the blocks and copying the data.
  /// This is extremely inefficient but not much of an issue in our
  /// use case.
  class BlockMatrixAdapter
  {
    std::shared_ptr<const BlockMatrix> _AA;   // Block matrix
    std::shared_ptr<PETScMatrix> _A;          // Flat matrix
    std::vector<std::size_t> _row_offsets;
    std::vector<std::size_t> _col_offsets;

    std::size_t _nrows, _ncols;
  public:
    /// Constructor. Takes a BlockMatrix and flattens it. All blocks
    /// must already be initialised (sparsity patterns set), in
    /// particular they need to be in a state in which they can be
    /// queried for their entries.
    BlockMatrixAdapter(std::shared_ptr<const BlockMatrix> AA)
      : _AA(AA) { assemble(); }

    /// Read the contents of the blocks into the flattened Matrix
    void read(int i, int j);

    const PETScMatrix& get() const { return *_A; }
    PETScMatrix& get() { return *_A; }
  protected:
    /// Initialise offsets and patterns of non zeros from blocks in
    /// the BlockMatrix
    void assemble();
  };  
}


#endif // __BLOCK_MATRIX_ADAPTER_H
