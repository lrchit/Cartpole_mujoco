

#pragma once

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <iostream>  // standard input/output
#include <vector>    // standard vector
#include <yaml-cpp/yaml.h>

#include "auto_diff/CppAdInterface.h"
#include <dynamics.h>

#include <ModelHelperFunctions.h>

class Quadruped_Dynamics : public Dynamics {
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Quadruped_Dynamics(YAML::Node config, const pinocchio::ModelTpl<ocs2::scalar_t>& model, const std::vector<size_t>& footId);
  ~Quadruped_Dynamics();

  virtual ocs2::vector_t getValue(const ocs2::vector_t& x, const ocs2::vector_t& u) override;
  virtual std::pair<ocs2::matrix_t, ocs2::matrix_t> getFirstDerivatives(const ocs2::vector_t& x, const ocs2::vector_t& u) override;

  virtual std::pair<ocs2::vector_t, ocs2::vector_t> solveQuasiStaticProblem(const ocs2::vector_t& x) override;

  private:
  // discrete
  template <typename SCALAR_T>
  ocs2::vector_s_t<SCALAR_T> quadruped_discrete_dynamics(const pinocchio::ModelTpl<SCALAR_T>& pinocchioModel,
      const pinocchio::DataTpl<SCALAR_T>& pinocchioData,
      const ocs2::vector_s_t<SCALAR_T>& x,
      const ocs2::vector_s_t<SCALAR_T>& u);

  int nx_ = 36;
  int nu_ = 12;
  int nq_ = 18;
  int nv_ = 18;
  double dt_;

  std::vector<size_t> footId_;
  ocs2::legged_robot::ContactModelParam param_;
  pinocchio::ModelTpl<ocs2::scalar_t> pinocchioModel_;
  pinocchio::DataTpl<ocs2::scalar_t> pinocchioData_;
  pinocchio::ModelTpl<ocs2::ad_scalar_t> pinocchioModelCppAd_;
  pinocchio::DataTpl<ocs2::ad_scalar_t> pinocchioDataCppAd_;

  std::shared_ptr<ocs2::CppAdInterface> systemFlowMapCppAdInterfacePtr_;  //!< CppAd code gen
};