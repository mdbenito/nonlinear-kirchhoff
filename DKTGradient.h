#ifndef __DKTGRADIENT_HPP
#define __DKTGRADIENT_HPP

#include <Eigen/Dense>
#include <vector>
#include <array>
#include <memory>

namespace dolfin
{
  class Cell;
  class Vector;
  class Function;
  class FunctionSpace;
  class GenericTensor;

/* Computes the transformation matrix for $\nabla_h$ over one cell.
*
* WARNING: This was written for SCALAR function spaces *BUT* it can
* be used in the vector valued case for the computation of D = M^T A M,
* where M is the discrete gradient matrix thanks to block diagonal
* structure of the matrices
*
* The code assumes that the dofs of the "source" cell (in DKT) are
* ordered as
* 
*      w1, w1_x, w1_y, w2, w2_x, w2_y, w3, w3_x, w3_y,
*      
* i.e. point evaluation, partial derivatives for each vertex in
* turn, and those of the "target" cell (in P_2^2) as
* 
*     t1_0, t1_1, t2_0, t2_1, t3_0, t3_1,
*     ts1_0, ts1_1, ts2_0, ts2_1, ts3_0, ts3_1
*     
* i.e. values at the three vertices, then values at the the
* midpoints of the opposite sides.
*/
class DKTGradient
{
public:
  // Use RowMajor for compatibility with python / fenics. Is this ok?
  typedef Eigen::Matrix<double, 12, 9, Eigen::RowMajor> M_t;
  typedef Eigen::Matrix<double, 9, 12, Eigen::RowMajor> Mt_t;
  typedef std::array<double, 9*9> P3Tensor;
  typedef std::array<double, 12*12> P22Tensor;
  typedef std::array<double, 12> P22Vector;
  typedef std::array<double, 9> P3Vector;

  /* Initialize the local cell gradient matrix.
   * (interpolates a scalar function from P_2^2 into DKT)
   *
   * dim is the number of subspaces this gradient will operate on,
   * one by one.
   */
  DKTGradient(int dim=3);

  /* Updates the operator matrix for the given Cell */
  void update(const dolfin::Cell& cell);
  void update(const std::vector<double>& cc);
  
  /* Compute $ M v $ for $ v \in P_3^{red} $
   * Returns local coefficients in $ P_2^2 $
   */
  void apply_vec(const P3Vector& p3coeffs, P22Vector& p22coeffs);
  
  /* Compute D = M^T A M where
   *
   *  A is the local tensor for (\nabla u, \nabla v) in a $ P_2^2 $
   *  element D is the local tensor for (\nabla \nabla_h u, \nabla
   *  \nabla_h v)
   *
   * Works for dim > 1 ASSUMING that p22tensor is just a 12x12 chunk
   * of the actual matrix (better for KirchhoffAssembler)
   */
  void apply(const P22Tensor& p22tensor, P3Tensor& D);
  void apply(const double* p22tensor, P3Tensor& D);

  /* Computes the discrete gradient of the given function.
   *
   * FIXME: this is VERY INEFFICIENTLY done, cell by cell each time
   * we are called. Instead, one could compute the global gradient
   * matrix once, then just multiply the full vector of coefficients
   * by it.
   *
   * Arguments:
   *    T is a (P2^2)^3 function space
   *    W is a (DKT)^3 function space
   *    dktfun is a Vector of coefficients in (DKT)^3
   * Returns:
   *    Function in T
   */
  std::unique_ptr<dolfin::Vector>
  apply_vec(std::shared_ptr<const dolfin::FunctionSpace> T,
            std::shared_ptr<const dolfin::FunctionSpace> W,
            std::shared_ptr<const dolfin::Vector> dktfun);
                        
protected:
  M_t   _M;  // cell-local gradient matrix
  Mt_t _Mt;  // cell-local transposed gradient matrix
  int _dim;  // range dimension
};
}

#endif // __DKTGRADIENT_HPP
