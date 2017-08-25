#ifndef __BLOCK_VECTOR_ADAPTER_H
#define __BLOCK_VECTOR_ADAPTER_H

#include <memory>
#include <vector>

namespace dolfin {

  class BlockVector;
  class PETScVector;

  ///// A BlockVectorAdapter is a flattened interface to a BlockVector.
  /// Ideally I'd implement the interface of a GenericLinearOperator,
  /// but I don't know how hard that could become if solve() is to be
  /// supported. For now I just build a bigger vector using the
  /// sparsity patterns of the blocks and copying the data.
  /// This is extremely inefficient but not much of an issue in our
  /// use case.
  class BlockVectorAdapter
  {
    std::shared_ptr<BlockVector> _VV;       // Block vector
    std::shared_ptr<PETScVector> _V;        // Flat vector
    std::vector<std::size_t> _row_offsets;
    std::size_t _nrows;

  public:
    /// Constructor. Takes a BlockVector and flattens it.
    BlockVectorAdapter(std::shared_ptr<BlockVector> VV)
      : _VV(VV) { assemble(); }
    
    /// Read the contents of one block into the flattened Vector
    void read(int i);
    /// Write the contents of some section of the flattened Vector
    /// into the corresponding block
    void write(int i);

    std::shared_ptr<PETScVector> get() { return _V; }

  protected:
    /// Initialise offsets from blocks in the BlockVector
    void assemble();
  };  
}


#endif // __BLOCK_VECTOR_ADAPTER_H
