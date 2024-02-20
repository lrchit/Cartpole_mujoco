
#pragma once

// #include <ColPackHeaders.h> //
// ColPack的主头文件（根据实际安装情况可能有所不同）
#include <cppad/cppad.hpp> // the CppAD package
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <iostream> // standard input/output
#include <vector>   // standard vector

using namespace Eigen;
using CppAD::AD;
using CppAD::sparse_rc;
using CppAD::sparse_rcv;
using std::vector; // use vector as abbreviation for std::vector

typedef vector<double> d_vector;
typedef vector<int> s_vector;

#define pi 3.1416

class Cartpole_Dynamics {
public:
  Cartpole_Dynamics(double _dt, double _m_cart, double _m_pole, double _l);
  ~Cartpole_Dynamics();

  // discretizised dynamics
  vector<AD<double>> cartpole_dynamics_discrete(const vector<AD<double>> &x);
  // continuous dynamics
  vector<AD<double>> cartpole_dynamics_continuous(const vector<AD<double>> &x);
  // compute jacobian
  Matrix<double, 4, 5>
  jacobian_cartpole_dynamics_discrete(const Vector<double, 4> &_x,
                                      const double &_u);

  // rollout dynamics
  Vector<double, 5> cartpole_dynamics_model(const Vector<double, 5> &x);
  Vector<double, 4> cartpole_dynamics_rollout(const Vector<double, 4> &_x,
                                              const double &_u);

private:
  int nx, nu;
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