
#pragma once

#include <chrono>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <matplotlibcpp.h>
#include <mujoco/mujoco.h>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "pd_controller.h"
#include <dynamics.h>

using namespace Eigen;
using std::vector;
namespace plt = matplotlibcpp;

class Cartpole_iLQR
{
  public:
  Cartpole_iLQR(std::string yaml_name);
  ~Cartpole_iLQR();

  double backward_pass();
  void iLQR_algorithm(const Matrix<double, 4, 1>& xcur, const double& ucur);

  double stage_cost(const Matrix<double, 4, 1>& x, const double& u);
  double terminal_cost(const Matrix<double, 4, 1>& x);
  double cost(const Matrix<double, 4, Dynamic>& _xtraj, const Matrix<double, 1, Dynamic>& _utraj);

  void get_control(mjData* d);
  void traj_plot();

  private:
  bool isPositiveDefinite(const MatrixXd& M);
  double vector_max(const vector<double>& v);

  int nx, nu;
  double dt;
  double step;
  double Tfinal;
  int Nt;

  double m_cart, m_pole;
  double l;

  Matrix<double, 4, Dynamic> xtraj;
  Matrix<double, 1, Dynamic> utraj;
  vector<double> Jtraj;
  Matrix<double, 4, 1> x0;
  Matrix<double, 4, 1> xgoal;

  MatrixXd Q, Qn;
  double R;
  vector<Matrix<double, 4, 1>> p;
  vector<Matrix<double, 4, 4>> P;
  vector<double> d;
  vector<Matrix<double, 1, 4>> K;

  Cartpole_Dynamics* cartpole_dynamics;
};