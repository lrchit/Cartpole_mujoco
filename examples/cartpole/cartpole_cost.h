
#pragma once

#include <iostream>  // standard input/output
#include <vector>    // standard vector
#include <yaml-cpp/yaml.h>

#include "auto_diff/CppAdInterface.h"
#include <cost.h>

class Cartpole_Cost : public Cost {
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Cartpole_Cost(YAML::Node config);
  ~Cartpole_Cost();

  virtual ocs2::scalar_t getValue(const ocs2::vector_t& x, const ocs2::vector_t& u, const ocs2::vector_t& xgoal) override;
  virtual ocs2::scalar_t getValue(const ocs2::vector_t& x, const ocs2::vector_t& xgoal) override;

  virtual std::pair<ocs2::vector_t, ocs2::vector_t> getFirstDerivatives(const ocs2::vector_t& x, const ocs2::vector_t& u, const ocs2::vector_t& xgoal) override;
  virtual std::pair<ocs2::vector_t, ocs2::vector_t> getFirstDerivatives(const ocs2::vector_t& x, const ocs2::vector_t& xgoal) override;

  virtual std::tuple<ocs2::matrix_t, ocs2::matrix_t, ocs2::matrix_t> getSecondDerivatives(const ocs2::vector_t& x,
      const ocs2::vector_t& u,
      const ocs2::vector_t& xgoal) override;
  virtual std::tuple<ocs2::matrix_t, ocs2::matrix_t, ocs2::matrix_t> getSecondDerivatives(const ocs2::vector_t& x, const ocs2::vector_t& xgoal) override;

  private:
  const int nx = 4;
  const int nu = 1;

  Eigen::Matrix<double, 4, 4> Q_, Qn_;
  Eigen::Matrix<double, 1, 1> R_;

  ocs2::scalar_t dt_;
};