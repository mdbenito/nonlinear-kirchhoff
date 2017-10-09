/* 
 *
 */
#ifndef DATA_H
#define DATA_H

#include "dkt_utils.h"
#include <dolfin.h>

using namespace dolfin;

const double LEFT = -2.0, RIGHT = 2.0, BOTTOM = 0.0, TOP = 1.0;

class Force : public DiffExpression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = 0;
    values[1] = 0;
    values[2] = 1e-5;
  }
  
  void
  gradient(Array<double>& grad, const Array<double>& x) const override {
    grad[0] = 0.0; grad[1] = 0.0;
    grad[2] = 0.0; grad[3] = 0.0;
    grad[4] = 0.0; grad[5] = 0.0;
  }
  
  std::size_t value_rank() const { return 1; }
  std::size_t value_dimension(std::size_t i) const { return 3;}
};

class FullBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary;
  }
};

class LateralBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary && (near(x[0], LEFT) || near(x[0], RIGHT));
  }
};

class InitialData : public DiffExpression
{
  const double pi6 = M_PI / 6;
  const double pi3 = M_PI / 3;
  const double pi2 = M_PI / 2;
  const double pi = M_PI;
  
  void eval(Array<double>& values, const Array<double>& x) const
  {
    
    double t = x[0];
    values[1] = x[1];

    if (-2.0 <= t && t < -2.0 + pi6) {
      values[0] = -2.0/3.0 + (1.0/3) * std::cos(3.0*(t+2.0)-pi2);
      values[2] =  1.0/3.0 + (1.0/3) * std::sin(3.0*(t+2.0)-pi2);
    } else if (-2.0 + pi6 <= t && t < -pi6) {
      values[0] = -1.0/3.0;
      values[2] =  1.0/3.0 + t - (-2.0 + pi6);
    } else if (-pi6 <= t && t < pi6) {
      values[0] =               0 + (1.0/3.0) * std::cos(pi - 3.0*(t+pi6));
      values[2] = 1.0/3.0+2.0-pi3 + (1.0/3.0) * std::sin(3.0*(t+pi6));
    } else if (pi6 <= t && t < 2.0 - pi6) {
      values[0] = 1.0/3.0;
      values[2] = 1.0/3.0 + 2.0 - pi3 - t + pi6;
    } else if (2.0 - pi6 <= t && t <= 2.0 ) {
      values[0] = 2.0/3.0 + (1.0/3.0)*std::cos(3.0*(t-2.0+pi6)+pi);
      values[2] = 1.0/3.0 + (1.0/3.0)*std::sin(3.0*(t-2.0+pi6)+pi);
    }      
  }

  /// Gradient is stored in row format: f_11, f_12, f_21, f_22, f_31, f_32
  void gradient(Array<double>& grad, const Array<double>& x) const
  {  
    double t = x[0];
    grad[1] = grad[2] = grad[5] = 0;
    grad[3] = 1;
    if (-2.0 <= t && t < -2.0 + pi6) {
      grad[0] = - std::sin(3.0*(t+2.0)-pi2);
      grad[4] =   std::cos(3.0*(t+2.0)-pi2);
    } else if (-2.0 + pi6 <= t && t < -pi6) {
      grad[0] = 0;
      grad[4] = 1;
    } else if (-pi6 <= t && t < pi6) {
      grad[0] = std::sin(M_PI - 3.0*(t+pi6));
      grad[4] = std::cos(3.0*(t+pi6));
    } else if (pi6 <= t && t < 2.0 - pi6) {
      grad[0] = 0;
      grad[4] = -1;
    } else if (2.0 - pi6 <= t && t <= 2.0 ) {
      grad[0] = - std::sin(3.0*(t-2.0+pi6)+pi);
      grad[4] = std::cos(3.0*(t-2.0+pi6)+pi);
    }
  }
    
  std::size_t value_rank() const { return 1; }
  std::size_t value_dimension(std::size_t i) const { return 3; }
};

#endif /* DATA_H */
