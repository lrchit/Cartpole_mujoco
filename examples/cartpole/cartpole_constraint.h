
#pragma once

#include <constraint.h>
#include <BoxConstraint.h>

class Cartpole_Constraint : public BoxConstraint {
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Cartpole_Constraint(YAML::Node config) : BoxConstraint(4, 1) {
    ocs2::vector_t lbx(nx), ubx(nx), lbu(nu), ubu(nu);
    lbx = ocs2::vector_t{};
    ubx = ocs2::vector_t{};
    lbu << -100;
    ubu << 100;
    setBounds(lbx, ubx, lbu, ubu);
  }
  ~Cartpole_Constraint() = default;

  private:
  const int nx = 4;
  const int nu = 1;

  std::shared_ptr<ocs2::CppAdInterface> constraintFlowMapCppAdInterfacePtr_;  //!< CppAd code gen
};