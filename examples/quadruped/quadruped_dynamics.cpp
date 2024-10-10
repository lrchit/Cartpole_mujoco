
#include <pinocchio/fwd.hpp>  // always include it before any other header

#include <pinocchio/parsers/urdf.hpp>
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/parsers/sample-models.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/spatial/explog.hpp"
#include <pinocchio/algorithm/center-of-mass.hpp>

#include <quadruped_dynamics.h>

Quadruped_Dynamics::Quadruped_Dynamics(YAML::Node config, const pinocchio::ModelTpl<ocs2::scalar_t>& model, const std::vector<size_t>& footId)
    : pinocchioModel_(model),
      pinocchioData_(pinocchio::DataTpl<ocs2::scalar_t>(pinocchioModel_)),
      pinocchioModelCppAd_(model.template cast<ocs2::ad_scalar_t>()),
      pinocchioDataCppAd_(pinocchio::DataTpl<ocs2::ad_scalar_t>(pinocchioModelCppAd_)),
      footId_(footId) {
  dt_ = config["mpc"]["dt"].as<double>();
  param_.k_spring = config["dms"]["contactModelParam"]["k_spring"].as<double>();
  param_.d_damper = config["dms"]["contactModelParam"]["d_damper"].as<double>();
  param_.damper_smooth = config["dms"]["contactModelParam"]["damper_smooth"].as<double>();
  param_.spring_smooth = config["dms"]["contactModelParam"]["spring_smooth"].as<double>();
  param_.zOffset = config["dms"]["contactModelParam"]["zOffset"].as<double>();
  param_.contact_stiffness = config["dms"]["contactModelParam"]["contact_stiffness"].as<double>();
  param_.dissipation_velocity = config["dms"]["contactModelParam"]["dissipation_velocity"].as<double>();
  param_.smoothing_factor = config["dms"]["contactModelParam"]["smoothing_factor"].as<double>();
  param_.friction_coefficient = config["dms"]["contactModelParam"]["friction_coefficient"].as<double>();
  param_.stiction_velocity = config["dms"]["contactModelParam"]["stiction_velocity"].as<double>();
  param_.which_contact_model = config["dms"]["contactModelParam"]["which_contact_model"].as<int>();

  auto systemFlowMapFunc = [&](const ocs2::ad_vector_t& x, ocs2::ad_vector_t& y) {
    ocs2::ad_vector_t state = x.head(nx_);
    ocs2::ad_vector_t input = x.tail(nu_);
    y = quadruped_discrete_dynamics(pinocchioModelCppAd_, pinocchioDataCppAd_, state, input);
  };
  systemFlowMapCppAdInterfacePtr_.reset(new ocs2::CppAdInterface(systemFlowMapFunc, nx_ + nu_, "quadruped_dynamics_systemFlowMap", "../cppad_generated"));
  if (config["dms"]["recompileCppAdLibrary"].as<bool>()) {
    systemFlowMapCppAdInterfacePtr_->createModels(ocs2::CppAdInterface::ApproximationOrder::First, true);
  } else {
    systemFlowMapCppAdInterfacePtr_->loadModelsIfAvailable(ocs2::CppAdInterface::ApproximationOrder::First, true);
  }
}

Quadruped_Dynamics::~Quadruped_Dynamics() {}

ocs2::vector_t Quadruped_Dynamics::getValue(const ocs2::vector_t& x, const ocs2::vector_t& u) {
  // const ocs2::vector_t stateInput = (ocs2::vector_t(x.rows() + u.rows()) << x, u).finished();
  // return systemFlowMapCppAdInterfacePtr_->getFunctionValue(stateInput);
  pinocchio::forwardKinematics(pinocchioModel_, pinocchioData_, x.head(18), x.tail(18));
  pinocchio::updateFramePlacements(pinocchioModel_, pinocchioData_);
  // std::cerr << "a*dt = " << dt_ * ocs2::legged_robot::forwardDynamics(pinocchioModel_, pinocchioData_, x, u, param_, footId_).transpose() << std::endl;
  for (int leg = 0; leg < 4; ++leg) {
    // std::cerr << "x = " << pinocchioData_.oMf[footId_[leg]].translation().transpose() << std::endl;
    // std::cerr << "v = "
    //           << pinocchio::getFrameVelocity(pinocchioModel_, pinocchioData_, footId_[leg], pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED).linear().transpose()
    //           << std::endl;
    std::cerr << " force = " << computeEEForce(pinocchioModel_, pinocchioData_, param_, footId_[leg], x).transpose() << std::endl;
  }
  return quadruped_discrete_dynamics(pinocchioModel_, pinocchioData_, x, u);
}

std::pair<ocs2::matrix_t, ocs2::matrix_t> Quadruped_Dynamics::getFirstDerivatives(const ocs2::vector_t& x, const ocs2::vector_t& u) {
  const ocs2::vector_t stateInput = (ocs2::vector_t(x.rows() + u.rows()) << x, u).finished();
  const ocs2::matrix_t Jacobian = systemFlowMapCppAdInterfacePtr_->getJacobian(stateInput);
  if (Jacobian == ocs2::matrix_t::Zero(nx_, nx_ + nu_)) {
    std::cerr << "jacobian =\n" << Jacobian << std::endl;
    std::cerr << "x =\n" << x.transpose() << std::endl;
    std::cerr << "u =\n" << u.transpose() << std::endl;
  }
  return std::pair(Jacobian.leftCols(nx_), Jacobian.rightCols(nu_));
}

std::pair<ocs2::vector_t, ocs2::vector_t> Quadruped_Dynamics::solveQuasiStaticProblem(const ocs2::vector_t& x) {
  return std::pair(x, ocs2::legged_robot::weightCompensatingInput(pinocchioModel_, pinocchioData_, x, footId_));
}

template <typename SCALAR_T>
ocs2::vector_s_t<SCALAR_T> Quadruped_Dynamics::quadruped_discrete_dynamics(const pinocchio::ModelTpl<SCALAR_T>& pinocchioModel,
    const pinocchio::DataTpl<SCALAR_T>& pinocchioData,
    const ocs2::vector_s_t<SCALAR_T>& x,
    const ocs2::vector_s_t<SCALAR_T>& u) {
  const ocs2::vector_s_t<SCALAR_T> acceleration = ocs2::legged_robot::forwardDynamics(pinocchioModel, pinocchioData, x, u, param_, footId_);
  return (ocs2::vector_s_t<SCALAR_T>(nx_) << x.head(nq_) + SCALAR_T(dt_) * x.tail(nv_), x.tail(nv_) + SCALAR_T(dt_) * acceleration).finished();
}
template ocs2::vector_t Quadruped_Dynamics::quadruped_discrete_dynamics(const pinocchio::ModelTpl<ocs2::scalar_t>&,
    const pinocchio::DataTpl<ocs2::scalar_t>&,
    const ocs2::vector_t&,
    const ocs2::vector_t&);
template ocs2::ad_vector_t Quadruped_Dynamics::quadruped_discrete_dynamics(const pinocchio::ModelTpl<ocs2::ad_scalar_t>&,
    const pinocchio::DataTpl<ocs2::ad_scalar_t>&,
    const ocs2::ad_vector_t&,
    const ocs2::ad_vector_t&);