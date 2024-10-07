#include <cartpole_cost.h>

Cartpole_Cost::Cartpole_Cost(YAML::Node config) {
  ocs2::scalar_t dt = config["dt"].as<double>();

  // Initialize weight matrix
  Q_.setZero();
  Qn_.setZero();
  R_.setZero();
  Q_(0, 0) = config["Q"]["q1"].as<double>();
  Q_(1, 1) = config["Q"]["q2"].as<double>();
  Q_(2, 2) = config["Q"]["q3"].as<double>();
  Q_(3, 3) = config["Q"]["q4"].as<double>();
  Q_ = Q_ * dt;
  Qn_(0, 0) = config["Qn"]["q1"].as<double>();
  Qn_(1, 1) = config["Qn"]["q2"].as<double>();
  Qn_(2, 2) = config["Qn"]["q3"].as<double>();
  Qn_(3, 3) = config["Qn"]["q4"].as<double>();
  R_(0, 0) = config["R"]["r1"].as<double>();
  R_ = R_ * dt;
}

Cartpole_Cost::~Cartpole_Cost() {}

ocs2::scalar_t Cartpole_Cost::getValue(const ocs2::vector_t& x, const ocs2::vector_t& u, const ocs2::vector_t& xref) {
  return (0.5 * (x - xref).transpose() * Q_ * (x - xref) + 0.5 * u.transpose() * R_ * u).value();
}
ocs2::scalar_t Cartpole_Cost::getValue(const ocs2::vector_t& x, const ocs2::vector_t& xref) {
  return (0.5 * (x - xref).transpose() * Qn_ * (x - xref)).value();
}

std::pair<ocs2::vector_t, ocs2::vector_t> Cartpole_Cost::getFirstDerivatives(const ocs2::vector_t& x, const ocs2::vector_t& u, const ocs2::vector_t& xref) {
  return std::pair((Q_ * (x - xref)).transpose(), (R_ * u).transpose());
}
std::pair<ocs2::vector_t, ocs2::vector_t> Cartpole_Cost::getFirstDerivatives(const ocs2::vector_t& x, const ocs2::vector_t& xref) {
  return std::pair((Qn_ * (x - xref)).transpose(), ocs2::vector_t::Zero(nu));
}

std::tuple<ocs2::matrix_t, ocs2::matrix_t, ocs2::matrix_t> Cartpole_Cost::getSecondDerivatives(const ocs2::vector_t& x,
    const ocs2::vector_t& u,
    const ocs2::vector_t& xref) {
  return std::tuple(Q_, ocs2::matrix_t::Zero(nu, nx), R_);
}
std::tuple<ocs2::matrix_t, ocs2::matrix_t, ocs2::matrix_t> Cartpole_Cost::getSecondDerivatives(const ocs2::vector_t& x, const ocs2::vector_t& xref) {
  return std::tuple(Qn_, ocs2::matrix_t::Zero(nu, nx), ocs2::matrix_t::Zero(nu, nu));
}