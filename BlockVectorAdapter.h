#ifndef __BLOCK_VECTOR_ADAPTER_H
#define __BLOCK_VECTOR_ADAPTER_H

#include <memory>
#include <vector>

namespace dolfin {

  class BlockVector;
  class PETScVector;
  
  /// Ideally I'd implmement the interface of a GenericLinearOperator,
  /// but I don't know how hard that could become if solve() is to be
  /// supported.  For now I just build a bigger vector using the
  /// sparsity patterns of the blocks and copying the data.
  class BlockVectorAdapter
  {
    std::shared_ptr<const BlockVector> _AA;   // Block vector
    std::shared_ptr<PETScVector> _A;          // Flat vector
    std::vector<std::size_t> _row_offsets;

    std::size_t _nrows;
  public:
    BlockVectorAdapter(std::shared_ptr<const BlockVector> AA)
      : _AA(AA) { }

    void rebuild();
    void update(int i);
  };  
}


#endif // __BLOCK_VECTOR_ADAPTER_H
