#include <vector>
#include <array>
#include <cassert>

#include <dolfin.h>
#include "DKTGradient.h"
#include "output.h"

using namespace dolfin;

DKTGradient::DKTGradient(int dim)
  : _dim(dim)
{
  // Setting the OuterStride needs testing!
  
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
DKTGradient::update(const Cell& cell)
{
  std::vector<double> cc;         // cell coordinates
  cell.get_vertex_coordinates(cc);
  update(cc);
}

/// HACK: this permutation is applied to M during update()
// to adapt for the actual ordering of dofs local to a cell.
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
  _M.setZero();
  _M(0,1) = 1; _M(1,2) = 1;
  _M(2,4) = 1; _M(3,5) = 1;
  _M(4,7) = 1; _M(5,8) = 1;

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
    TT[IIJ(i,0,0)] = 0.5 - 0.75*m[IJ(0,0)];
    TT[IIJ(i,0,1)] =     - 0.75*m[IJ(0,1)];
    TT[IIJ(i,1,0)] =     - 0.75*m[IJ(1,0)];
    TT[IIJ(i,1,1)] = 0.5 - 0.75*m[IJ(1,1)];
    tt[IJ(i,0)] *= -3/(2*ss[i]);
    tt[IJ(i,1)] *= -3/(2*ss[i]);
  }

  // Copy onto the gradient (sub) matrix (this is actually wasteful..)
  auto copytt = [&](size_t i, size_t r, size_t c) {
    _M.coeffRef(r,c)   = tt[IJ(i,0)];
    _M.coeffRef(r+1,c) = tt[IJ(i,1)];
  };
  auto copy_tt = [&](size_t i, size_t r, size_t c) {
    _M.coeffRef(r,c)   = -tt[IJ(i,0)];
    _M.coeffRef(r+1,c) = -tt[IJ(i,1)];
  };
  auto copyTT = [&](size_t i, size_t r, size_t c) {
    _M.coeffRef(r,c)   = TT[IIJ(i,0,0)];
    _M.coeffRef(r,c+1)   = TT[IIJ(i,0,1)];
    _M.coeffRef(r+1,c) = TT[IIJ(i,1,0)];
    _M.coeffRef(r+1,c+1) = TT[IIJ(i,1,1)];
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

  permutation_hack(_M);

  _Mt = _M.transpose();
}

void
DKTGradient::apply(const double* p22tensor, P3Tensor& dkttensor)
{
  Eigen::Map<const Eigen::Matrix<double, 12, 12, Eigen::RowMajor>,
             0, Eigen::OuterStride<>> p22(p22tensor,
                                          Eigen::OuterStride<>(_dim*12));
  Eigen::Map<Eigen::Matrix<double, 9, 9, Eigen::RowMajor>>
    dkt(dkttensor.data());

  // Eigen::Matrix<double, 12, 12, Eigen::RowMajor> tmp(p22);
  // dump_raw_matrix(tmp.data(), 12, 12, "DKT", false);

  dkt = _Mt * p22 * _M;
}

void
DKTGradient::apply_vec(const P3Vector& p3coeffs, P22Vector& p22coeffs)
{
  Eigen::Map<const Eigen::Matrix<double, 9, 1>> arg(p3coeffs.data());
  Eigen::Map<Eigen::Matrix<double, 12, 1>> dest(p22coeffs.data());
  dest = _M * arg;
}

std::unique_ptr<Vector>
DKTGradient::apply_vec(std::shared_ptr<const FunctionSpace> T,   // (P2^2)^3
                       std::shared_ptr<const FunctionSpace> W,   // DKT^3
                       std::shared_ptr<const Vector> dktvec)
{
  std::unique_ptr<Vector> vec(new Vector(MPI_COMM_WORLD, T->dim()));
  
  P22Vector p22coeffs;
  auto p3coeffs = std::vector<double>(12);
  auto mesh = W->mesh();

  P3Vector block;
  
  for (CellIterator cell(*mesh); !cell.end(); ++cell)
  {
    update(*cell);
    for (int i = 0; i < _dim; ++i) {
      // std::cout << "\nSubspace " << i << "\n";
      // 1. extract dofs for cell using W's dofmap
      auto dmW = W->sub(i)->dofmap().get();
      auto dofsW = dmW->cell_dofs(cell->index());
      // std::cout << "Got dofs: " << v2s(dofsW) << "\n";
      // 1.1 Check ranges for parallel ???
      // auto range = vec->local_range();
      // std::cout << "Range: " << range.first << " to "
      //           << range.second << "\n";
      // 2. extract local coeffs from dktfun for this cell
      assert(block.size() == dofsW.size());
      dktvec->get(block.data(), dofsW.size(), dofsW.data());
      // std::cout << "Got local: " << v2s(block) << "\n";
      // 3. Compute local DKT gradient for this cell and subspace
      apply_vec(block, p22coeffs);
      // std::cout << "Computed local: " << v2s(p22coeffs) << "\n";
      // 4. Insert coeffs into fun using T's dofmap
      auto dmT = T->sub(i)->dofmap().get();
      auto dofsT = dmT->cell_dofs(cell->index());
      // std::cout << "Set dofs: " << v2s(dofsT) << "\n";
      assert(p22coeffs.size() == dofsT.size());
      vec->set(p22coeffs.data(), dofsT.size(), dofsT.data());
      // std::cout << "Done.\n";
      // 4.1 Check ranges for parallel ???
    }
  }
  // std::cout << "Apllying all...";
  vec->apply("insert");
  // std::cout << " Done.\n";
  return vec;
}
