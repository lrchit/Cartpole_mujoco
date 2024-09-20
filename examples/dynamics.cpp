
#include <dynamics.h>

ocs2::vector_t Dynamics::getValue(const ocs2::vector_t& x, const ocs2::vector_t& u) {
  const ocs2::vector_t stateInput = (ocs2::vector_t(x.rows() + u.rows()) << x, u).finished();
  return systemFlowMapCppAdInterfacePtr_->getFunctionValue(stateInput);
}

ocs2::matrix_t Dynamics::getFirstDerivatives(const ocs2::vector_t& x, const ocs2::vector_t& u) {
  const ocs2::vector_t stateInput = (ocs2::vector_t(x.rows() + u.rows()) << x, u).finished();
  return systemFlowMapCppAdInterfacePtr_->getJacobian(stateInput);
}