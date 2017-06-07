#include <vector>
#include <array>
#include <cassert>

#include <dolfin.h>
#include "DKTGradient.h"

DKTGradient::DKTGradient()
{
  /* TODO: I can only fix the ones if they are correctly placed, 
   i.e. after removing the permutation_hack() and inserting the values
   of M in the right places...

  M.setZero();

  // Fill identity submatrices
  M(0,1) = 1; M(1,2) = 1;
  M(2,4) = 1; M(3,5) = 1;
  M(4,7) = 1; M(5,8) = 1;
  */
}

/// Updates the operator matrix for the given Cell
void
DKTGradient::update(const dolfin::Cell& cell)
{
  std::vector<double> cc;         // cell coordinates
  cell.get_vertex_coordinates(cc);
  update(cc);
}

void
permutation_hack(DKTGradient::M_t& M)
{
  const int rows = 12;
  Eigen::Matrix<double, 12, 12, Eigen::RowMajor> P;
  P.setZero();
  // position in array is destination row, value is source row:
  int permutations[] = {0,6,1,7,2,8,3,9,4,10,5,11};
  for (int i = 0; i < 12; ++i)
    P(permutations[i], i) = 1.0;
  M = (P * M).eval();
}

void
DKTGradient::update(const std::vector<double>& cc)
{
  assert(cc.size() == 6);
  // FIXME: do this in the constructor once permutation_hack is removed:
  // Fill identity submatrices
  M.setZero();
  M(0,1) = 1; M(1,2) = 1;
  M(2,4) = 1; M(3,5) = 1;
  M(4,7) = 1; M(5,8) = 1;

  // FIXME: use Eigen for these too
  std::array<double, 3*2> tt;     // tangent vectors
  std::array<double, 3*2*2> TT;   // tt_i x tt_i^T (tensor prods.)
  std::array<double, 3> ss;       // cell side lengths
 
  // Vector and matrix access helpers for tt and TT
  // respectively. I guess these will be optimized away...
  auto IJ = [](size_t i, size_t j) -> size_t { return i*2 + j; };
  auto IIJ = [](size_t k, size_t i, size_t j) -> size_t { return 4*k + i*2 + j; };

  // FIXME: shouldn't this depend on the orientation?
  tt[IJ(0,0)] = cc[IJ(2,0)] - cc[IJ(1,0)];
  tt[IJ(0,1)] = cc[IJ(2,1)] - cc[IJ(1,1)];      
  tt[IJ(1,0)] = cc[IJ(2,0)] - cc[IJ(0,0)];  // Why is this negated?
  tt[IJ(1,1)] = cc[IJ(2,1)] - cc[IJ(0,1)];
  tt[IJ(2,0)] = cc[IJ(1,0)] - cc[IJ(0,0)];
  tt[IJ(2,1)] = cc[IJ(1,1)] - cc[IJ(0,1)];
      
  auto outer = [&](double x0, double x1) -> std::array<double, 2*2> {
    return { x0*x0, x0*x1, x1*x0, x1*x1 };
  };

  for (int i=0; i < 3; ++i) {
    ss[i] = std::sqrt(tt[IJ(i,0)]*tt[IJ(i,0)] + tt[IJ(i,1)]*tt[IJ(i,1)]);
    tt[IJ(i,0)] /= ss[i];
    tt[IJ(i,1)] /= ss[i];
    auto m = outer(tt[IJ(i,0)], tt[IJ(i,1)]);
    TT[IIJ(i,0,0)] = 0.5 - 0.75*m[IJ(0,0)]; TT[IIJ(i,0,1)] =     - 0.75*m[IJ(0,1)];
    TT[IIJ(i,1,0)] =     - 0.75*m[IJ(1,0)]; TT[IIJ(i,1,1)] = 0.5 - 0.75*m[IJ(1,1)];
    tt[IJ(i,0)] *= -3/(2*ss[i]);
    tt[IJ(i,1)] *= -3/(2*ss[i]);
  }

  // Copy onto the gradient (sub) matrix (this is actually wasteful..)
  auto copytt = [&](size_t i, size_t r, size_t c) {
    M.coeffRef(r,c)   = tt[IJ(i,0)];
    M.coeffRef(r+1,c) = tt[IJ(i,1)];
  };
  auto copy_tt = [&](size_t i, size_t r, size_t c) {
    M.coeffRef(r,c)   = -tt[IJ(i,0)];
    M.coeffRef(r+1,c) = -tt[IJ(i,1)];
  };
  auto copyTT = [&](size_t i, size_t r, size_t c) {
    M.coeffRef(r,c)   = TT[IIJ(i,0,0)]; M.coeffRef(r,c+1)   = TT[IIJ(i,0,1)];
    M.coeffRef(r+1,c) = TT[IIJ(i,1,0)]; M.coeffRef(r+1,c+1) = TT[IIJ(i,1,1)];
  };

  copytt(0, 6, 3);
  copyTT(0, 6, 4);
  copy_tt(0, 6, 6);
  copyTT(0, 6, 7);
      
  copytt(1, 8, 0);
  copyTT(1, 8, 1);
  copy_tt(1, 8, 6);
  copyTT(1, 8, 7);
      
  copytt(2, 10, 0);
  copyTT(2, 10, 1);
  copy_tt(2, 10, 3);
  copyTT(2, 10, 4);

  permutation_hack(M);

  Mt = M.transpose();
}

/// Compute $ M v $ for $ v \in P_3^{red} $
/// Returns local coefficients in $ P_2^2 $
void
DKTGradient::apply_vec(const std::vector<double>& p3coeffs,
                       std::array<double, 12>& p22coeffs)
{
  Eigen::Map<const Eigen::Matrix<double, 9, 1>> arg(p3coeffs.data());
  Eigen::Map<Eigen::Matrix<double, 12, 1>> dest(p22coeffs.data());
  dest = M * arg;
}

/// Compute M^T A M
/// A will be the local tensor for (grad u, grad v) in a $ P_2^2 $ element
void
DKTGradient::apply(const std::vector<double>& p22tensor,
                   P3Tensor& dkttensor)
{
  Eigen::Map<const Eigen::Matrix<double, 12, 12, Eigen::RowMajor>> p22(p22tensor.data());
  Eigen::Map<Eigen::Matrix<double, 9, 9, Eigen::RowMajor>> dkt(dkttensor.data());
  dkt = Mt * p22 * M;
}
