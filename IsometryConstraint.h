#ifndef __ISOMETRY_CONSTRAINT_H
#define __ISOMETRY_CONSTRAINT_H

#include <vector>
#include <memory>
#include <dolfin.h>

namespace dolfin {

  class IsometryConstraint
  {
    std::vector<la_index> _v2d;
    /// The Constraint matrix. This is going to be used in a BlockMatrix
    /// which requires shared ownership via a shared_ptr.
    std::shared_ptr<Matrix> _B;
    /// A copy of the transposed Constraint matrix. Stored for speed.
    /// TODO: test whether this is indeed faster
    std::shared_ptr<Matrix> _Bt;

    std::shared_ptr<TensorLayout> _B_tensor_layout;
    std::shared_ptr<TensorLayout> _Bt_tensor_layout;

  public:
    IsometryConstraint(const FunctionSpace& W,
                       const VertexFunction<bool>& boundary_marker);
    
    void update_with(const Function& Y);
    std::shared_ptr<Matrix> get() { return _B; }
    std::shared_ptr<Matrix> get_transposed() { return _Bt; }
  };
}

#endif // __ISOMETRY_CONSTRAINT_H
