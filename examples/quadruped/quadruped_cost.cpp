#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <quadruped_cost.h>

Quadruped_Cost::Quadruped_Cost(YAML::Node config, const pinocchio::ModelTpl<ocs2::scalar_t>& model, const std::vector<size_t>& footId)
    : pinocchioModel_(model.template cast<ocs2::ad_scalar_t>()), pinocchioData_(pinocchio::DataTpl<ocs2::ad_scalar_t>(pinocchioModel_)) {
  ocs2::scalar_t dt = config["mpc"]["dt"].as<double>();

  // Initialize weight matrix
  Q_.setZero();
  Qn_.setZero();
  R_.setZero();
  std::vector<double> Q_vector = config["dms"]["Q"].as<std::vector<double>>();
  std::vector<double> Qn_vector = config["dms"]["Qn"].as<std::vector<double>>();
  for (int i = 0; i < nx; ++i) {
    Q_(i, i) = Q_vector[i] * dt;
    Qn_(i, i) = Qn_vector[i];
  }
  std::vector<double> R_vector = config["dms"]["R"].as<std::vector<double>>();
  for (int i = 0; i < nu; ++i) {
    R_(i, i) = R_vector[i] * dt;
  }
}

Quadruped_Cost::~Quadruped_Cost() {}

ocs2::scalar_t Quadruped_Cost::getValue(const ocs2::vector_t& x, const ocs2::vector_t& u, const ocs2::vector_t& xref) {
  return (0.5 * (x - xref).transpose() * Q_ * (x - xref) + 0.5 * u.transpose() * R_ * u).value();
}
ocs2::scalar_t Quadruped_Cost::getValue(const ocs2::vector_t& x, const ocs2::vector_t& xref) {
  return (0.5 * (x - xref).transpose() * Qn_ * (x - xref)).value();
}

std::pair<ocs2::vector_t, ocs2::vector_t> Quadruped_Cost::getFirstDerivatives(const ocs2::vector_t& x, const ocs2::vector_t& u, const ocs2::vector_t& xref) {
  return std::pair(Q_ * (x - xref), R_ * u);
}
std::pair<ocs2::vector_t, ocs2::vector_t> Quadruped_Cost::getFirstDerivatives(const ocs2::vector_t& x, const ocs2::vector_t& xref) {
  return std::pair(Qn_ * (x - xref), ocs2::vector_t{});
}

std::tuple<ocs2::matrix_t, ocs2::matrix_t, ocs2::matrix_t> Quadruped_Cost::getSecondDerivatives(const ocs2::vector_t& x,
    const ocs2::vector_t& u,
    const ocs2::vector_t& xref) {
  return std::tuple(Q_, ocs2::matrix_t::Zero(nu, nx), R_);
}
std::tuple<ocs2::matrix_t, ocs2::matrix_t, ocs2::matrix_t> Quadruped_Cost::getSecondDerivatives(const ocs2::vector_t& x, const ocs2::vector_t& xref) {
  return std::tuple(Qn_, ocs2::matrix_t{}, ocs2::matrix_t{});
}