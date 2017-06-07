#ifndef __DKTGRADIENT_HPP
#define __DKTGRADIENT_HPP

#include <Eigen/Dense>
#include <vector>
#include <array>

namespace dolfin{
  class Cell;
  class GenericTensor;
}

/// Computes the transformation matrix for $\nabla_h$ over one cell.    
/// This assumes that the dofs of the "source" cell (in DKT) are ordered as
/// 
///      w1, w1_x, w1_y, w2, w2_x, w2_y, w3, w3_x, w3_y,
///      
/// i.e. point evaluation, partial derivatives for each vertex in turn, and
/// those of the "target" cell (in P_2^2) as
/// 
///     t1_0, t1_1, t2_0, t2_1, t3_0, t3_1,
///     ts1_0, ts1_1, ts2_0, ts2_1, ts3_0, ts3_1
///     
/// i.e. values at the three vertices, then values at the the midpoints of
/// the opposite sides.
class DKTGradient
{
public:
  // Use RowMajor for compatibility with python / fenics. Is this ok?
  typedef Eigen::Matrix<double, 12, 9, Eigen::RowMajor> M_t;
  typedef Eigen::Matrix<double, 9, 12, Eigen::RowMajor> Mt_t;
  M_t   M;  // gradient matrix
  Mt_t Mt; // transposed gradient matrix
  
  typedef std::array<double, 9*9> P3Tensor;
  typedef std::array<double, 12> P22Vector;
  
  /// Initialise the local cell gradient matrix.
  /// (interpolates from P_2^2 into DKT)
  DKTGradient();

  /// Updates the operator matrix for the given Cell
  void update(const dolfin::Cell& cell);
  void update(const std::vector<double>& cc);
  
  /// Compute $ M v $ for $ v \in P_3^{red} $
  /// Returns local coefficients in $ P_2^2 $
  void apply_vec(const std::vector<double>& p3coeffs, P22Vector& p22coeffs);
  
  /// Compute D = M^T A M where
  ///  A is the local tensor for (\nabla u, \nabla v) in a $ P_2^2 $ element
  ///  D is the local tensor for (\nabla \nabla_h u, \nabla \nabla_h v)
  void apply(const std::vector<double>& p22tensor, P3Tensor& D);

  
};

#endif // __DKTGRADIENT_HPP
