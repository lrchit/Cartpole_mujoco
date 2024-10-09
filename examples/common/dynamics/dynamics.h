#pragma once

#include "auto_diff/CppAdInterface.h"

class Dynamics {
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Dynamics() {}
  ~Dynamics() = default;

  virtual ocs2::vector_t getValue(const ocs2::vector_t& x, const ocs2::vector_t& u) = 0;
  virtual std::pair<ocs2::matrix_t, ocs2::matrix_t> getFirstDerivatives(const ocs2::vector_t& x, const ocs2::vector_t& u) = 0;

  virtual std::pair<ocs2::vector_t, ocs2::vector_t> solveQuasiStaticProblem(const ocs2::vector_t& x) = 0;

  protected:
};