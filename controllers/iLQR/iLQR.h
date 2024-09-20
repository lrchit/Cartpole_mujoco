
#pragma once

#include <chrono>
#include <auto_diff/Types.h>
#include <iostream>
#include <matplotlibcpp.h>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "pd_controller.h"
#include <dynamics.h>

#define pi 3.1416

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
  std::vector<ocs2::matrix_t> lux;
  std::vector<ocs2::matrix_t> luu;

  // dynamics 1st order approximation
  std::vector<ocs2::matrix_t> fx;
  std::vector<ocs2::vector_t> fu;
};

class iLQR_Solver {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  public:
  iLQR_Solver(YAML::Node config, std::shared_ptr<Dynamics> dynamics_model);
  ~iLQR_Solver();

  std::vector<ocs2::vector_t> iLQR_algorithm(const ocs2::vector_t& xcur, const std::vector<ocs2::vector_t>& x_goal);

  void set_reference(const std::vector<ocs2::vector_t>& x_goal) { xgoal = x_goal; }

  private:
  double calCost(const std::vector<ocs2::vector_t>& _xtraj, const std::vector<ocs2::vector_t>& _utraj);
  void calDerivatives();
  double backward_pass();
  double line_search(double delta_J, double J);
  void solve();

  void reset_solver(const ocs2::vector_t& xcur, const std::vector<ocs2::vector_t>& x_goal);

  void traj_plot();

  int nx, nu;
  double dt;
  int Nt;

  bool verbose_cal_time = false;

  std::vector<ocs2::vector_t> xtraj;
  std::vector<ocs2::vector_t> utraj;
  std::vector<double> Jtraj;
  std::vector<ocs2::vector_t> xgoal;

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

  std::vector<std::shared_ptr<Dynamics>> dynamics_model_;
};