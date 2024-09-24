
#pragma once

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <iostream>  // standard input/output
#include <vector>    // standard vector
#include <yaml-cpp/yaml.h>

#include "auto_diff/CppAdInterface.h"
#include <cost.h>

#include <ModelHelperFunctions.h>

class Quadruped_Cost : public Cost {
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Quadruped_Cost(YAML::Node config, const pinocchio::ModelTpl<ocs2::scalar_t>& model, const std::vector<size_t>& footId);
  ~Quadruped_Cost();

  virtual ocs2::scalar_t getValue(const ocs2::vector_t& x, const ocs2::vector_t& u, const ocs2::vector_t& xref) override;
  virtual ocs2::scalar_t getValue(const ocs2::vector_t& x, const ocs2::vector_t& xref) override;

  virtual std::pair<ocs2::vector_t, ocs2::vector_t> getFirstDerivatives(const ocs2::vector_t& x, const ocs2::vector_t& u, const ocs2::vector_t& xref) override;
  virtual std::pair<ocs2::vector_t, ocs2::vector_t> getFirstDerivatives(const ocs2::vector_t& x, const ocs2::vector_t& xref) override;

  virtual std::tuple<ocs2::matrix_t, ocs2::matrix_t, ocs2::matrix_t> getSecondDerivatives(const ocs2::vector_t& x,
      const ocs2::vector_t& u,
      const ocs2::vector_t& xref) override;
  virtual std::tuple<ocs2::matrix_t, ocs2::matrix_t, ocs2::matrix_t> getSecondDerivatives(const ocs2::vector_t& x, const ocs2::vector_t& xref) override;

  private:
  const int nx = 36;
  const int nu = 12;

  Eigen::Matrix<double, 36, 36> Q_, Qn_;
  Eigen::Matrix<double, 12, 12> R_;

  pinocchio::ModelTpl<ocs2::ad_scalar_t> pinocchioModel_;
  pinocchio::DataTpl<ocs2::ad_scalar_t> pinocchioData_;

  std::shared_ptr<ocs2::CppAdInterface> costFlowMapCppAdInterfacePtr_;  //!< CppAd code gen
};