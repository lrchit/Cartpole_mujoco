#pragma once

#include "auto_diff/CppAdInterface.h"

class Dynamics {
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Dynamics() {}
  ~Dynamics() {}

  virtual ocs2::vector_t getValue(const ocs2::vector_t& x, const ocs2::vector_t& u) {
    const ocs2::vector_t stateInput = (ocs2::vector_t(x.rows() + u.rows()) << x, u).finished();
    return systemFlowMapCppAdInterfacePtr_->getFunctionValue(stateInput);
  }
  virtual ocs2::matrix_t getFirstDerivatives(const ocs2::vector_t& x, const ocs2::vector_t& u) {
    const ocs2::vector_t stateInput = (ocs2::vector_t(x.rows() + u.rows()) << x, u).finished();
    return systemFlowMapCppAdInterfacePtr_->getJacobian(stateInput);
  }

  virtual ocs2::vector_t getQuasiStaticInput(const ocs2::vector_t& x) = 0;

  virtual const int get_nx() { return nx; }
  virtual const int get_nu() { return nu; }

  protected:
  int nx;
  int nu;
  double dt;

  std::shared_ptr<ocs2::CppAdInterface> systemFlowMapCppAdInterfacePtr_;  //!< CppAd code gen
};