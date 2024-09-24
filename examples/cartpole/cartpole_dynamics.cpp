
#include <cartpole_dynamics.h>

Cartpole_Dynamics::Cartpole_Dynamics(YAML::Node config) {
  m_cart = config["m_cart"].as<double>();
  m_pole = config["m_pole"].as<double>();
  l = config["l"].as<double>() / 2;
  g = 9.81;
  nx = 4;
  nu = 1;

  dt = config["dt"].as<double>();

  auto systemFlowMapFunc = [&](const ocs2::ad_vector_t& x, ocs2::ad_vector_t& y) {
    ocs2::ad_vector_t state = x.head(4);
    ocs2::ad_vector_t input = x.tail(1);
    y = cartpole_discrete_dynamics<ocs2::ad_scalar_t>(state, input);
  };
  systemFlowMapCppAdInterfacePtr_.reset(new ocs2::CppAdInterface(systemFlowMapFunc, 5, "cartpole_dynamics_systemFlowMap", "../cppad_generated"));
  if (true) {
    systemFlowMapCppAdInterfacePtr_->createModels(ocs2::CppAdInterface::ApproximationOrder::First, true);
  } else {
    systemFlowMapCppAdInterfacePtr_->loadModelsIfAvailable(ocs2::CppAdInterface::ApproximationOrder::First, true);
  }
}

Cartpole_Dynamics::~Cartpole_Dynamics() {}

ocs2::vector_t Cartpole_Dynamics::getValue(const ocs2::vector_t& x, const ocs2::vector_t& u) {
  const ocs2::vector_t stateInput = (ocs2::vector_t(x.rows() + u.rows()) << x, u).finished();
  return systemFlowMapCppAdInterfacePtr_->getFunctionValue(stateInput);
}

std::pair<ocs2::matrix_t, ocs2::matrix_t> Cartpole_Dynamics::getFirstDerivatives(const ocs2::vector_t& x, const ocs2::vector_t& u) {
  const ocs2::vector_t stateInput = (ocs2::vector_t(x.rows() + u.rows()) << x, u).finished();
  const ocs2::matrix_t Jacobian = systemFlowMapCppAdInterfacePtr_->getJacobian(stateInput);
  if (Jacobian == ocs2::matrix_t::Zero(nx, nx + nu)) {
    std::cerr << "jacobian =\n" << Jacobian << std::endl;
    std::cerr << "x =\n" << x.transpose() << std::endl;
    std::cerr << "u =\n" << u.transpose() << std::endl;
  }
  return std::pair(Jacobian.leftCols(nx), Jacobian.rightCols(nu));
}

template <typename T>
ocs2::vector_s_t<T> Cartpole_Dynamics::cartpole_dynamics_model(const ocs2::vector_s_t<T>& x, const ocs2::vector_s_t<T>& u) {
  ocs2::vector_s_t<T> dx(nx);  // 状态变量的变化率
  // x_cart的动力学方程
  dx[0] = x[2];
  // theta的动力学方程
  dx[1] = x[3];

  T s1, c1;
  s1 = sin(x[1]);
  c1 = cos(x[1]);

  // x_cart_dot的动力学方程
  dx[2] =
      (u[0] + m_pole * T(l) * s1 * x[3] * x[3] - T(3.0) * T(m_pole) * g * s1 * c1 / T(4.0)) / (T(m_cart) + T(m_pole) - T(3.0) * T(m_pole) * c1 * c1 / T(4.0));
  // std::cout << "m_pole = " << m_pole << std::endl;
  // std::cout << "m_cart = " << m_cart << std::endl;
  // std::cout << "l = " << l << std::endl;
  // theta_dot的动力学方程
  dx[3] = T(3.0) * (T(g) * s1 - dx[2] * c1) / T(4.0) / T(l);

  return dx;
}

template <typename T>
ocs2::vector_s_t<T> Cartpole_Dynamics::cartpole_discrete_dynamics(const ocs2::vector_s_t<T>& x, const ocs2::vector_s_t<T>& u) {
  ocs2::vector_s_t<T> k1, k2, k3, k4, x_next;
  // 计算k1
  k1 = cartpole_dynamics_model<T>(x, u) * T(dt);
  // 计算k2
  k2 = cartpole_dynamics_model<T>(x + T(0.5) * k1, u) * T(dt);
  // 计算k3
  k3 = cartpole_dynamics_model<T>(x + T(0.5) * k2, u) * T(dt);
  // 计算k4
  k4 = cartpole_dynamics_model<T>(x + k3, u) * T(dt);

  // 更新状态
  x_next = x + T(1.0 / 6.0) * (k1 + T(2.0) * k2 + T(2.0) * k3 + k4);

  return x_next;
}

template ocs2::vector_t Cartpole_Dynamics::cartpole_dynamics_model(const ocs2::vector_t&, const ocs2::vector_t&);
template ocs2::ad_vector_t Cartpole_Dynamics::cartpole_dynamics_model(const ocs2::ad_vector_t&, const ocs2::ad_vector_t&);
template ocs2::vector_t Cartpole_Dynamics::cartpole_discrete_dynamics(const ocs2::vector_t&, const ocs2::vector_t&);
template ocs2::ad_vector_t Cartpole_Dynamics::cartpole_discrete_dynamics(const ocs2::ad_vector_t&, const ocs2::ad_vector_t&);
