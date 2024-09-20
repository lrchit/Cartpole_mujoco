
#pragma once

#include <chrono>
#include <auto_diff/Types.h>
#include <iostream>
#include <matplotlibcpp.h>
#include <mujoco/mujoco.h>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "pd_controller.h"
#include <dynamics.h>

namespace plt = matplotlibcpp;

struct DDP_Matrix {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ocs2::vector_t Qx;
  ocs2::vector_t Qu;
  ocs2::matrix_t Qxx;
  ocs2::matrix_t Quu;
  ocs2::matrix_t Quu_inverse;
  ocs2::matrix_t Qxu;
  ocs2::matrix_t Qux;
};

struct Derivatives {
  // cost 2nd order approximation
  std::vector<ocs2::vector_t> lx;
  std::vector<ocs2::vector_t> lu;
  std::vector<ocs2::matrix_t> lxx;
  std::vector<ocs2::matrix_t> lxu;
  std::vector<ocs2::matrix_t> luu;

  // dynamics 1st order approximation
  std::vector<ocs2::matrix_t> fx;
  std::vector<ocs2::vector_t> fu;
};

class Cartpole_iLQR {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  public:
  Cartpole_iLQR(std::string yaml_name);
  ~Cartpole_iLQR();

  void get_control(mjData* d);

  private:
  bool isPositiveDefinite(const ocs2::matrix_t& M);
  double vector_max(const std::vector<ocs2::vector_t>& v);

  double cost(const std::vector<ocs2::vector_t>& _xtraj, const std::vector<ocs2::vector_t>& _utraj);
  void calDerivatives();
  double backward_pass();
  double line_search(double delta_J, double J);
  void solve();

  void reset_solver(const ocs2::vector_t& xcur);

  void iLQR_algorithm(const ocs2::vector_t& xcur);

  void traj_plot();

  int nx, nu;
  double dt;
  double step;
  double Tfinal;
  int Nt;

  double m_cart, m_pole;
  double l;

  bool verbose_cal_time = false;

  bool first_run = true;

  std::vector<ocs2::vector_t> xtraj;
  std::vector<ocs2::vector_t> utraj;
  std::vector<double> Jtraj;
  ocs2::vector_t x0;
  ocs2::vector_t xgoal;

  ocs2::matrix_t Q, Qn;
  ocs2::matrix_t R;
  std::vector<ocs2::vector_t> p;
  std::vector<ocs2::matrix_t> P;
  std::vector<ocs2::vector_t> d;
  std::vector<ocs2::matrix_t> K;

  DDP_Matrix ddp_matrix;
  Derivatives derivatives;

  // line search param
  double sigma;
  double beta;
  double tolerance;

  std::vector<ocs2::scalar_t> derivativeTime_;
  std::vector<ocs2::scalar_t> backwardPassTime_;
  std::vector<ocs2::scalar_t> lineSeachTime_;

  std::vector<std::shared_ptr<Cartpole_Dynamics>> cartpole_dynamics_;
};