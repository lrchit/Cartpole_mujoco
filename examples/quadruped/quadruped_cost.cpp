#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <quadruped_cost.h>

Quadruped_Cost::Quadruped_Cost(YAML::Node config, const pinocchio::ModelTpl<ocs2::scalar_t>& model, const std::vector<size_t>& footId)
    : pinocchioModel_(model.template cast<ocs2::ad_scalar_t>()), pinocchioData_(pinocchio::DataTpl<ocs2::ad_scalar_t>(pinocchioModel_)) {
  ocs2::scalar_t dt = config["dt"].as<double>();

  // Initialize weight matrix
  Q_(0, 0) = config["Q"]["pos"]["x"].as<double>();
  Q_(1, 1) = config["Q"]["pos"]["y"].as<double>();
  Q_(2, 2) = config["Q"]["pos"]["z"].as<double>();
  Q_(3, 3) = config["Q"]["zyxangle"]["z"].as<double>();
  Q_(4, 4) = config["Q"]["zyxangle"]["y"].as<double>();
  Q_(5, 5) = config["Q"]["zyxangle"]["x"].as<double>();
  Q_(6, 6) = config["Q"]["qj"]["abad"].as<double>();
  Q_(7, 7) = config["Q"]["qj"]["hip"].as<double>();
  Q_(8, 8) = config["Q"]["qj"]["knee"].as<double>();
  Q_(9, 9) = config["Q"]["qj"]["abad"].as<double>();
  Q_(10, 10) = config["Q"]["qj"]["hip"].as<double>();
  Q_(11, 11) = config["Q"]["qj"]["knee"].as<double>();
  Q_(12, 12) = config["Q"]["qj"]["abad"].as<double>();
  Q_(13, 13) = config["Q"]["qj"]["hip"].as<double>();
  Q_(14, 14) = config["Q"]["qj"]["knee"].as<double>();
  Q_(15, 15) = config["Q"]["qj"]["abad"].as<double>();
  Q_(16, 16) = config["Q"]["qj"]["hip"].as<double>();
  Q_(17, 17) = config["Q"]["qj"]["knee"].as<double>();
  Q_(18, 18) = config["Q"]["linvel"]["z"].as<double>();
  Q_(19, 19) = config["Q"]["linvel"]["y"].as<double>();
  Q_(20, 20) = config["Q"]["linvel"]["x"].as<double>();
  Q_(21, 21) = config["Q"]["zyxanglevel"]["x"].as<double>();
  Q_(22, 22) = config["Q"]["zyxanglevel"]["y"].as<double>();
  Q_(23, 23) = config["Q"]["zyxanglevel"]["z"].as<double>();
  Q_(24, 24) = config["Q"]["qdj"]["abad"].as<double>();
  Q_(25, 25) = config["Q"]["qdj"]["hip"].as<double>();
  Q_(26, 26) = config["Q"]["qdj"]["knee"].as<double>();
  Q_(27, 27) = config["Q"]["qdj"]["abad"].as<double>();
  Q_(28, 28) = config["Q"]["qdj"]["hip"].as<double>();
  Q_(29, 29) = config["Q"]["qdj"]["knee"].as<double>();
  Q_(30, 30) = config["Q"]["qdj"]["abad"].as<double>();
  Q_(31, 31) = config["Q"]["qdj"]["hip"].as<double>();
  Q_(32, 32) = config["Q"]["qdj"]["knee"].as<double>();
  Q_(33, 33) = config["Q"]["qdj"]["abad"].as<double>();
  Q_(34, 34) = config["Q"]["qdj"]["hip"].as<double>();
  Q_(35, 35) = config["Q"]["qdj"]["knee"].as<double>();
  Q_ = Q_ * dt;
  Qn_(0, 0) = config["Qn"]["pos"]["x"].as<double>();
  Qn_(1, 1) = config["Qn"]["pos"]["y"].as<double>();
  Qn_(2, 2) = config["Qn"]["pos"]["z"].as<double>();
  Qn_(3, 3) = config["Qn"]["zyxangle"]["z"].as<double>();
  Qn_(4, 4) = config["Qn"]["zyxangle"]["y"].as<double>();
  Qn_(5, 5) = config["Qn"]["zyxangle"]["x"].as<double>();
  Qn_(6, 6) = config["Qn"]["qj"]["abad"].as<double>();
  Qn_(7, 7) = config["Qn"]["qj"]["hip"].as<double>();
  Qn_(8, 8) = config["Qn"]["qj"]["knee"].as<double>();
  Qn_(9, 9) = config["Qn"]["qj"]["abad"].as<double>();
  Qn_(10, 10) = config["Qn"]["qj"]["hip"].as<double>();
  Qn_(11, 11) = config["Qn"]["qj"]["knee"].as<double>();
  Qn_(12, 12) = config["Qn"]["qj"]["abad"].as<double>();
  Qn_(13, 13) = config["Qn"]["qj"]["hip"].as<double>();
  Qn_(14, 14) = config["Qn"]["qj"]["knee"].as<double>();
  Qn_(15, 15) = config["Qn"]["qj"]["abad"].as<double>();
  Qn_(16, 16) = config["Qn"]["qj"]["hip"].as<double>();
  Qn_(17, 17) = config["Qn"]["qj"]["knee"].as<double>();
  Qn_(18, 18) = config["Qn"]["linvel"]["z"].as<double>();
  Qn_(19, 19) = config["Qn"]["linvel"]["y"].as<double>();
  Qn_(20, 20) = config["Qn"]["linvel"]["x"].as<double>();
  Qn_(21, 21) = config["Qn"]["zyxanglevel"]["x"].as<double>();
  Qn_(22, 22) = config["Qn"]["zyxanglevel"]["y"].as<double>();
  Qn_(23, 23) = config["Qn"]["zyxanglevel"]["z"].as<double>();
  Qn_(24, 24) = config["Qn"]["qdj"]["abad"].as<double>();
  Qn_(25, 25) = config["Qn"]["qdj"]["hip"].as<double>();
  Qn_(26, 26) = config["Qn"]["qdj"]["knee"].as<double>();
  Qn_(27, 27) = config["Qn"]["qdj"]["abad"].as<double>();
  Qn_(28, 28) = config["Qn"]["qdj"]["hip"].as<double>();
  Qn_(29, 29) = config["Qn"]["qdj"]["knee"].as<double>();
  Qn_(30, 30) = config["Qn"]["qdj"]["abad"].as<double>();
  Qn_(31, 31) = config["Qn"]["qdj"]["hip"].as<double>();
  Qn_(32, 32) = config["Qn"]["qdj"]["knee"].as<double>();
  Qn_(33, 33) = config["Qn"]["qdj"]["abad"].as<double>();
  Qn_(34, 34) = config["Qn"]["qdj"]["hip"].as<double>();
  Qn_(35, 35) = config["Qn"]["qdj"]["knee"].as<double>();
  R_(0, 0) = config["R"]["r1"].as<double>();
  R_(1, 1) = config["R"]["r2"].as<double>();
  R_(2, 2) = config["R"]["r3"].as<double>();
  R_(3, 3) = config["R"]["r4"].as<double>();
  R_(4, 4) = config["R"]["r5"].as<double>();
  R_(5, 5) = config["R"]["r6"].as<double>();
  R_(6, 6) = config["R"]["r7"].as<double>();
  R_(7, 7) = config["R"]["r8"].as<double>();
  R_(8, 8) = config["R"]["r9"].as<double>();
  R_(9, 9) = config["R"]["r10"].as<double>();
  R_(10, 10) = config["R"]["r11"].as<double>();
  R_(11, 11) = config["R"]["r12"].as<double>();
  R_ = R_ * dt;
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