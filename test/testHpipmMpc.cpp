#include "hpipm_interface.h"

#include <chrono>
#include <iostream>
#include <vector>

#include <Types.h>
#include <yaml-cpp/yaml.h>

int main() {
  YAML::Node config = YAML::LoadFile("../test/config/hpipmMpc.yaml");
  HpipmInterface hpipmInterface(config);

  // setup QP
  const int N = config["horizon"].as<int>();

  // dynamics
  Eigen::MatrixXd A(12, 12), B(12, 4);
  A << 1., 0., 0., 0., 0., 0., 0.1, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.1, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.1, 0., 0., 0.,
      0.0488, 0., 0., 1., 0., 0., 0.0016, 0., 0., 0.0992, 0., 0., 0., -0.0488, 0., 0., 1., 0., 0., -0.0016, 0., 0., 0.0992, 0., 0., 0., 0., 0., 0., 1., 0., 0.,
      0., 0., 0., 0.0992, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
      0., 0., 0., 0.9734, 0., 0., 0., 0., 0., 0.0488, 0., 0., 0.9846, 0., 0., 0., -0.9734, 0., 0., 0., 0., 0., -0.0488, 0., 0., 0.9846, 0., 0., 0., 0., 0., 0.,
      0., 0., 0., 0., 0., 0., 0.9846;
  B << 0., -0.0726, 0., 0.0726, -0.0726, 0., 0.0726, 0., -0.0152, 0.0152, -0.0152, 0.0152, -0., -0.0006, -0., 0.0006, 0.0006, 0., -0.0006, 0.0000, 0.0106,
      0.0106, 0.0106, 0.0106, 0, -1.4512, 0., 1.4512, -1.4512, 0., 1.4512, 0., -0.3049, 0.3049, -0.3049, 0.3049, -0., -0.0236, 0., 0.0236, 0.0236, 0., -0.0236,
      0., 0.2107, 0.2107, 0.2107, 0.2107;
  const Eigen::VectorXd b = Eigen::VectorXd::Zero(12);
  // cost
  Eigen::MatrixXd Q(12, 12), S(4, 12), R(4, 4);
  Q.setZero();
  Q.diagonal() << 0, 0, 10., 10., 10., 10., 0, 0, 0, 5., 5., 5.;
  S.setZero();
  R.setZero();
  R.diagonal() << 0.1, 0.1, 0.1, 0.1;
  Eigen::VectorXd x_ref(12);
  x_ref << 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  const Eigen::VectorXd q = -Q * x_ref;
  const Eigen::VectorXd r = Eigen::VectorXd::Zero(4);

  std::vector<Eigen::MatrixXd> Q_vector;
  std::vector<Eigen::MatrixXd> S_vector;
  std::vector<Eigen::MatrixXd> R_vector;
  std::vector<Eigen::VectorXd> q_vector;
  std::vector<Eigen::VectorXd> r_vector;
  Q_vector.resize(N + 1, Q);
  S_vector.resize(N + 1, S);
  R_vector.resize(N + 1, R);
  q_vector.resize(N + 1, q);
  r_vector.resize(N + 1, r);

  std::vector<Eigen::MatrixXd> A_vector;
  std::vector<Eigen::MatrixXd> B_vector;
  std::vector<Eigen::VectorXd> b_vector;
  A_vector.resize(N + 1, A);
  B_vector.resize(N + 1, B);
  b_vector.resize(N + 1, b);

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(12);

  std::vector<Eigen::VectorXd> xtraj, utraj;
  xtraj.resize(N + 1);
  utraj.resize(N);
  for (int k = 0; k < N; ++k) {
    xtraj[k] = x0;
    utraj[k].setZero(4);
  }
  xtraj[N] = x0;

  const int sim_steps = 10;
  for (int t = 0; t < sim_steps; ++t) {
    std::cout << "t: " << t << ", x: " << xtraj[0].transpose() << std::endl;
    x0 = xtraj[0];

    // solve
    hpipmInterface.solve(xtraj, utraj, A_vector, B_vector, b_vector, Q_vector, S_vector, R_vector, q_vector, r_vector);

    const auto u0 = utraj[0];
    xtraj[0] = A * xtraj[0] + B * u0 + b;
  }

  std::cout << "t: " << sim_steps << ", x: " << xtraj[0].transpose() << std::endl;
  return 0;
}