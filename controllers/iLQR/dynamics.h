
#pragma once

#include <iostream>  // standard input/output
#include <vector>    // standard vector

#include "auto_diff/CppAdInterface.h"

#define pi 3.1416

class Cartpole_Dynamics {
  public:
  Cartpole_Dynamics(double _dt, double _m_cart, double _m_pole, double _l);
  ~Cartpole_Dynamics();

  ocs2::vector_t getValue(const ocs2::vector_t& x, const ocs2::vector_t& u);
  ocs2::matrix_t getFirstDerivatives(const ocs2::vector_t& x, const ocs2::vector_t& u);

  private:
  // dynamics
  template <typename T>
  ocs2::vector_s_t<T> cartpole_dynamics_model(const ocs2::vector_s_t<T>& x, const ocs2::vector_s_t<T>& u);

  // rollout
  template <typename T>
  ocs2::vector_s_t<T> cartpole_discrete_dynamics(const ocs2::vector_s_t<T>& x, const ocs2::vector_s_t<T>& u);

  const int nx = 4;
  const int nu = 1;
  double dt;
  double m_cart, m_pole;
  double l;
  double g;

  std::shared_ptr<ocs2::CppAdInterface> systemFlowMapCppAdInterfacePtr_;  //!< CppAd code gen
};