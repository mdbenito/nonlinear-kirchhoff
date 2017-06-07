#ifndef __ISOMETRY_CONSTRAINT_H
#define __ISOMETRY_CONSTRAINT_H

namespace dolfin {

  class IsometryConstraint
  {
    std::vector<la_index> _v2d;
    /// The Constraint matrix. This is going to be used in a BlockMatrix
    /// which requires shared ownership via a shared_ptr.
    std::shared_ptr<Matrix> B;
    /// A copy of the transposed Constraint matrix. Stored for speed.
    /// TODO: test whether this is indeed faster
    std::shared_ptr<Matrix> Bt;
    
  public:
    IsometryConstraint(const FunctionSpace& W);
    
    void update(const Function& Y);
    std::shared_ptr<const Matrix> get() { return B; }
    std::shared_ptr<const Matrix> get_transposed() { return Bt; }
  };
}


#endif // __ISOMETRY_CONSTRAINT_H
