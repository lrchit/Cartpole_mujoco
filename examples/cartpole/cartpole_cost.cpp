#include <cartpole_cost.h>

Cartpole_Cost::Cartpole_Cost(YAML::Node config) {
  dt_ = config["dt"].as<double>();

  // Initialize weight matrix
  Q_(0, 0) = config["Q"]["q1"].as<double>();
  Q_(1, 1) = config["Q"]["q2"].as<double>();
  Q_(2, 2) = config["Q"]["q3"].as<double>();
  Q_(3, 3) = config["Q"]["q4"].as<double>();
  Q_ = Q_ * dt_;
  Qn_(0, 0) = config["Qn"]["q1"].as<double>();
  Qn_(1, 1) = config["Qn"]["q2"].as<double>();
  Qn_(2, 2) = config["Qn"]["q3"].as<double>();
  Qn_(3, 3) = config["Qn"]["q4"].as<double>();
  R_(0, 0) = config["R"]["r1"].as<double>();
}

Cartpole_Cost::~Cartpole_Cost() {}

ocs2::scalar_t Cartpole_Cost::getValue(const ocs2::vector_t& x, const ocs2::vector_t& u, const ocs2::vector_t& xgoal) {
  return (0.5 * (x - xgoal).transpose() * Q_ * (x - xgoal) + 0.5 * u.transpose() * R_ * u).value();
}
ocs2::scalar_t Cartpole_Cost::getValue(const ocs2::vector_t& x, const ocs2::vector_t& xgoal) {
  return (0.5 * (x - xgoal).transpose() * Qn_ * (x - xgoal)).value();
}

std::pair<ocs2::vector_t, ocs2::vector_t> Cartpole_Cost::getFirstDerivatives(const ocs2::vector_t& x, const ocs2::vector_t& u, const ocs2::vector_t& xgoal) {
  return std::pair<ocs2::vector_t, ocs2::vector_t>(Q_ * (x - xgoal), R_ * u);
}
std::pair<ocs2::vector_t, ocs2::vector_t> Cartpole_Cost::getFirstDerivatives(const ocs2::vector_t& x, const ocs2::vector_t& xgoal) {
  return std::pair<ocs2::vector_t, ocs2::vector_t>(Qn_ * (x - xgoal), ocs2::vector_t{});
}

std::tuple<ocs2::matrix_t, ocs2::matrix_t, ocs2::matrix_t> Cartpole_Cost::getSecondDerivatives(const ocs2::vector_t& x,
    const ocs2::vector_t& u,
    const ocs2::vector_t& xgoal) {
  return std::tuple<ocs2::matrix_t, ocs2::matrix_t, ocs2::matrix_t>(Q_, ocs2::matrix_t::Zero(nu, nx), R_);
}
std::tuple<ocs2::matrix_t, ocs2::matrix_t, ocs2::matrix_t> Cartpole_Cost::getSecondDerivatives(const ocs2::vector_t& x, const ocs2::vector_t& xgoal) {
  return std::tuple<ocs2::matrix_t, ocs2::matrix_t, ocs2::matrix_t>(Qn_, ocs2::matrix_t{}, ocs2::matrix_t{});
}