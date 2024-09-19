
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

  std::vector<ocs2::vector_t> q;
  std::vector<ocs2::vector_t> r;
  std::vector<ocs2::matrix_t> A;
  std::vector<ocs2::vector_t> B;

  // line search param
  double sigma;
  double beta;

  std::vector<std::shared_ptr<ocs2::CppAdInterface>> systemFlowMapCppAdInterfacePtr_;  //!< CppAd code gen

  std::vector<ocs2::scalar_t> derivativeTime_;
  std::vector<ocs2::scalar_t> backwardPassTime_;
  std::vector<ocs2::scalar_t> lineSeachTime_;
};