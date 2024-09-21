#pragma once

#include <tuple>

#include "auto_diff/CppAdInterface.h"

class Cost {
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Cost() {}
  ~Cost() = default;

  virtual ocs2::scalar_t getValue(const ocs2::vector_t& x, const ocs2::vector_t& u, const ocs2::vector_t& xgoal) = 0;
  virtual ocs2::scalar_t getValue(const ocs2::vector_t& x, const ocs2::vector_t& xgoal) = 0;

  virtual std::pair<ocs2::vector_t, ocs2::vector_t> getFirstDerivatives(const ocs2::vector_t& x, const ocs2::vector_t& u, const ocs2::vector_t& xgoal) = 0;
  virtual std::pair<ocs2::vector_t, ocs2::vector_t> getFirstDerivatives(const ocs2::vector_t& x, const ocs2::vector_t& xgoal) = 0;

  virtual std::tuple<ocs2::matrix_t, ocs2::matrix_t, ocs2::matrix_t> getSecondDerivatives(const ocs2::vector_t& x,
      const ocs2::vector_t& u,
      const ocs2::vector_t& xgoal) = 0;
  virtual std::tuple<ocs2::matrix_t, ocs2::matrix_t, ocs2::matrix_t> getSecondDerivatives(const ocs2::vector_t& x, const ocs2::vector_t& xgoal) = 0;

  protected:
  std::shared_ptr<ocs2::CppAdInterface> costFlowMapCppAdInterfacePtr_;  //!< CppAd code gen
};