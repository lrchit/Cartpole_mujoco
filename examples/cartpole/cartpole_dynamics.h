
#pragma once

#include <iostream>  // standard input/output
#include <vector>    // standard vector
#include <yaml-cpp/yaml.h>

#include "auto_diff/CppAdInterface.h"
#include <dynamics.h>

class Cartpole_Dynamics : public Dynamics {
  public:
  Cartpole_Dynamics(double _dt, YAML::Node config);
  ~Cartpole_Dynamics();

  virtual ocs2::vector_t getQuasiStaticInput(const ocs2::vector_t& x) override { return ocs2::vector_t::Zero(nu); }

  private:
  // dynamics
  template <typename T>
  ocs2::vector_s_t<T> cartpole_dynamics_model(const ocs2::vector_s_t<T>& x, const ocs2::vector_s_t<T>& u);
  template <typename T>
  ocs2::vector_s_t<T> cartpole_discrete_dynamics(const ocs2::vector_s_t<T>& x, const ocs2::vector_s_t<T>& u);

  double m_cart, m_pole;
  double l;
  double g;
};