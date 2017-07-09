#ifndef __ISOMETRY_CONSTRAINT_H
#define __ISOMETRY_CONSTRAINT_H

#include <vector>
#include <memory>

namespace dolfin {

  class GenericMatrix;
  class Matrix;
  class FunctionSpace;
  template <class T> class VertexFunction;
  class Function;
  class TensorLayout;

  /// IsometryConstraint implements the discrete isometry constraint
  /// and Dirichlet boundary conditions for the gradient flow. See the
  /// paper for more info.
  class IsometryConstraint
  {
    /// Vertex id to global dof numbers. Keep in mind the offsets due
    /// to DKT being (sort of) a "mixed" function space.
    std::vector<int> _v2d;
    /// The Constraint matrix. This is going to be used in a BlockMatrix
    /// which requires shared ownership via a shared_ptr.
    std::shared_ptr<Matrix> _B;
    /// A copy of the transposed Constraint matrix. Stored for speed.
    /// TODO: test whether this is indeed faster
    std::shared_ptr<Matrix> _Bt;
    
    std::shared_ptr<TensorLayout> _B_tensor_layout;
    std::shared_ptr<TensorLayout> _Bt_tensor_layout;

  public:
    /// Constructor
    ///
    /// Initialises the sparsity pattern for the constraint matrix.
    IsometryConstraint(const FunctionSpace& W);

    /// Updates the constraint with the values from Y. See the doc.
    void update_with(const Function& Y);
    
    /// Return the constraint matrix. Use this to build the system
    /// matrix. Note that BlockMatrix requires pointers to non-const
    /// Matrix so we cannot const it here
    std::shared_ptr<Matrix> get() { return _B; }

    /// Return the transposed constraint matrix. Use this to build the
    /// system matrix. Note that BlockMatrix requires pointers to
    /// non-const Matrix so we cannot const it here
    std::shared_ptr<Matrix> get_transposed() { return _Bt; }

    /// Return an empty matrix of the size required to complete the
    /// full system matrix after appending B and Bt. This is just a
    /// convenience function which cleans up a bit the main program.
    static std::shared_ptr<GenericMatrix> get_zero_padding();
  };
}

#endif // __ISOMETRY_CONSTRAINT_H
