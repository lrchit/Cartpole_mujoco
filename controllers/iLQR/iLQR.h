
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

#define pi 3.1416

class Cartpole_iLQR {
public:
  Cartpole_iLQR(std::string yaml_name);
  ~Cartpole_iLQR();

  double backward_pass();
  void iLQR_algorithm(const Vector<double, 4> &xcur, const double &ucur);

  double stage_cost(const Vector<double, 4> &x, const double &u);
  double terminal_cost(const Vector<double, 4> &x);
  double cost(const Matrix<double, 4, -1> &_xtraj, const VectorXd &_utraj);

  void get_control(mjData *d);
  void traj_plot();

private:
  bool isPositiveDefinite(const MatrixXd &M);
  double vector_max(const vector<double> &v);

  int nx, nu;
  double dt;
  double step;
  double Tfinal;
  int Nt;

  double m_cart, m_pole;
  double l;

  Matrix<double, 4, -1> xtraj;
  VectorXd utraj;
  vector<double> Jtraj;
  Vector<double, 4> x0;
  Vector<double, 4> xgoal;

  MatrixXd Q, Qn;
  double R;
  vector<Vector<double, 4>> p;
  vector<Matrix<double, 4, 4>> P;
  vector<double> d;
  vector<Matrix<double, 1, 4>> K;

  Cartpole_Dynamics *cartpole_dynamics;
};