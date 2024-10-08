
#pragma once

#include <chrono>
#include <auto_diff/Types.h>
#include <iostream>
#include <matplotlibcpp.h>
#include <vector>
#include <yaml-cpp/yaml.h>

#include <dynamics.h>
#include <cost.h>

#include <controller.h>

// #define pi 3.1416

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

class iLQR_Solver : public ControllerBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  public:
  iLQR_Solver(YAML::Node config, std::shared_ptr<Dynamics> dynamics_model, std::shared_ptr<Cost> cost, const ocs2::matrix_t Kguess);
  ~iLQR_Solver();

  virtual void launch_controller(const ocs2::vector_t& xcur, const std::vector<ocs2::vector_t>& x_ref) override;

  virtual ocs2::matrix_t getFeedBackMatrix() override { return K[0]; }

  private:
  ocs2::scalar_t calCost(const std::vector<ocs2::vector_t>& _xtraj, const std::vector<ocs2::vector_t>& _utraj);
  void calDerivatives();
  ocs2::scalar_t backward_pass();
  ocs2::scalar_t line_search(ocs2::scalar_t delta_J, ocs2::scalar_t J);
  void solve();

  void traj_plot();

  int nx, nu;
  int Nt;

  bool verbose_cal_time = false;
  int max_iteration = 1000;

  std::vector<ocs2::scalar_t> Jtraj;

  std::vector<ocs2::vector_t> p;
  std::vector<ocs2::matrix_t> P;
  std::vector<ocs2::vector_t> d;
  std::vector<ocs2::matrix_t> K;

  // feedback gain for initial guess
  ocs2::matrix_t K_guess;

  DDP_Matrix ddp_matrix;
  std::unique_ptr<CostDerivatives> costDerivatives;
  std::unique_ptr<DynamicsDerivatives> dynamicsDerivatives;

  // line search param
  ocs2::scalar_t sigma;
  ocs2::scalar_t beta;
  ocs2::scalar_t tolerance;

  std::vector<ocs2::scalar_t> derivativeTime_;
  std::vector<ocs2::scalar_t> backwardPassTime_;
  std::vector<ocs2::scalar_t> lineSeachTime_;

  std::vector<std::shared_ptr<Cost>> cost_;
  std::vector<std::shared_ptr<Dynamics>> dynamics_model_;
};