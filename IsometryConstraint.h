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
    std::vector<int> _v2d;
    /// The Constraint matrix. This is going to be used in a BlockMatrix
    /// which requires shared ownership via a shared_ptr.
    std::shared_ptr<Matrix> _B;
    /// A copy of the transposed Constraint matrix. Stored for speed.
    /// TODO: test whether this is indeed faster
    std::shared_ptr<Matrix> _Bt;

    /// This marks points on the Dirichlet boundary
    std::shared_ptr<const VertexFunction<bool>> _boundary;
    
    std::shared_ptr<TensorLayout> _B_tensor_layout;
    std::shared_ptr<TensorLayout> _Bt_tensor_layout;

  public:
    /// Constructor
    ///
    /// Initialises the sparsity pattern for the constraint
    /// matrix. The VertexFunction boundary_marker tags all vertices
    /// at the Dirichlet boundary. Recall that the constraint matrix
    /// must encode the condition that the solutions to the system
    /// (the gradient updates) have homogenous Dirichlet conditions,
    /// in order for the updated solutions to fulfill the real
    /// Dirichlet BCs.
    IsometryConstraint(const FunctionSpace& W,
                       std::shared_ptr<const VertexFunction<bool>> boundary_marker);

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

    /// Return an emptz matrix of the size required to complete the
    /// full system matrix after appending B and Bt, i.e. 13x13. This
    /// is just a convenience function which cleans up a bit the main
    /// program.
    static std::shared_ptr<GenericMatrix> get_zero_padding();
  };
}

#endif // __ISOMETRY_CONSTRAINT_H
