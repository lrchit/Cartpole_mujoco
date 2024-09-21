
#pragma once

#include <cppad/cppad.hpp>  // the CppAD package
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <iostream>  // standard input/output
#include <vector>    // standard vector

using namespace Eigen;
using CppAD::AD;
using CppAD::sparse_rc;
using CppAD::sparse_rcv;

typedef Matrix<double, Dynamic, 1> d_vector;
typedef Matrix<int, Dynamic, 1> s_vector;

#define pi 3.1416

class Cartpole_Dynamics
{
  public:
  Cartpole_Dynamics(double _dt, double _m_cart, double _m_pole, double _l);
  ~Cartpole_Dynamics();

  // dynamics
  template <typename T>
  Matrix<T, Dynamic, 1> cartpole_dynamics_model(const Matrix<T, Dynamic, 1>& x, const Matrix<T, Dynamic, 1>& u);

  // rollout
  template <typename T>
  Matrix<T, Dynamic, 1> cartpole_dynamics_integrate(const Matrix<T, Dynamic, 1>& x, const Matrix<T, Dynamic, 1>& u);

  // compute jacobian
  Matrix<double, 4, 5> get_dynamics_jacobian(const Matrix<double, Dynamic, 1>& x, const Matrix<double, Dynamic, 1>& u);

  private:
  const int nx = 4;
  const int nu = 1;
  double dt;
  double m_cart, m_pole;
  double l;
  double g;

  // dynamcis
  CppAD::ADFun<double> f;

  // setting to solve sparse jac
  int group_max;
  std::string coloring;
  CppAD::sparse_jac_work work;
  sparse_rc<s_vector> pattern_jac;
  sparse_rcv<s_vector, d_vector> subset;
};