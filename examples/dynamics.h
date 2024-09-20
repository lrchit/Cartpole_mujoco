#pragma once

#include "auto_diff/CppAdInterface.h"

class Dynamics {
  public:
  Dynamics(){};
  ~Dynamics(){};

  virtual ocs2::vector_t getValue(const ocs2::vector_t& x, const ocs2::vector_t& u);
  virtual ocs2::matrix_t getFirstDerivatives(const ocs2::vector_t& x, const ocs2::vector_t& u);

  virtual ocs2::vector_t getQuasiStaticInput(const ocs2::vector_t& x) = 0;

  virtual ocs2::matrix_t getRunningStateCostMatrix() { return Q; }
  virtual ocs2::matrix_t getRunningInputCostMatrix() { return R; }
  virtual ocs2::matrix_t getTerminalStateCostMatrix() { return Qn; }

  virtual const int get_nx() { return nx; }
  virtual const int get_nu() { return nu; }

  protected:
  int nx;
  int nu;
  double dt;
  double m_cart, m_pole;
  double l;
  double g;

  ocs2::matrix_t Q, Qn;
  ocs2::matrix_t R;

  std::shared_ptr<ocs2::CppAdInterface> systemFlowMapCppAdInterfacePtr_;  //!< CppAd code gen
};