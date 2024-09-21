
#include <dynamics.h>

Cartpole_Dynamics::Cartpole_Dynamics(double _dt, double _m_cart, double _m_pole, double _l)
{
  dt = _dt;
  m_cart = _m_cart;
  m_pole = _m_pole;
  l = _l;
  g = 9.81;

  Matrix<AD<double>, Dynamic, 1> x_ad(nx + nu);
  // declare independent variables and start recording operation sequence
  CppAD::Independent(x_ad);
  // 创建AD函数对象，表示离散化后的动力学方程
  Matrix<AD<double>, Dynamic, 1> x_next_ad = cartpole_dynamics_integrate<AD<double>>(x_ad.head(nx), x_ad.tail(nu));

  // dynamcis
  f = CppAD::ADFun<double>(x_ad, x_next_ad);

  // n by n identity matrix sparsity
  sparse_rc<s_vector> pattern_in;
  pattern_in.resize(nx + nu, nx + nu, nx + nu);
  for (size_t k = 0; k < nx + nu; k++)
    pattern_in.set(k, k, k);
  // sparsity for J(x)
  bool transpose = false;
  bool dependency = false;
  bool internal_bool = true;
  f.for_jac_sparsity(pattern_in, transpose, dependency, internal_bool, pattern_jac);
  // compute entire forward mode Jacobian to initialize "work"
  subset = sparse_rcv<s_vector, d_vector>(pattern_jac);
  coloring = "cppad";
  group_max = 100;
  Matrix<double, Dynamic, 1> x_cur(nx + nu);
  x_cur << 1, 1, 1, 1, 1;
  f.sparse_jac_for(group_max, x_cur, subset, pattern_jac, coloring, work);
}

Cartpole_Dynamics::~Cartpole_Dynamics(){};

// compute jacobian
Matrix<double, 4, 5> Cartpole_Dynamics::get_dynamics_jacobian(const Matrix<double, Dynamic, 1>& x,
    const Matrix<double, Dynamic, 1>& u)
{
  Matrix<double, 4, 5> Jacobian;
  Matrix<double, Dynamic, 1> x_cur(nx + nu);
  for (int i = 0; i < nx; ++i)
  {
    x_cur[i] = x[i];
  }
  for (int i = nx; i < nx + nu; ++i)
  {
    x_cur[i] = u[i - nx];
  }

  // std::chrono::time_point<std::chrono::system_clock> t_start =
  //     std::chrono::system_clock::now();

  f.sparse_jac_for(group_max, x_cur, subset, pattern_jac, coloring, work);
  Jacobian.setZero();
  for (int i = 0; i < pattern_jac.row().size(); ++i)
  {
    Jacobian(pattern_jac.row()[i], pattern_jac.col()[i]) = subset.val()[i];
  }

  // std::chrono::time_point<std::chrono::system_clock> t_end =
  //     std::chrono::system_clock::now();
  // double time_record =
  //     std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
  //         .count();
  // std::cout << "controller_time: " << time_record << "\n";

  return Jacobian;
}

template <typename T>
Matrix<T, Dynamic, 1> Cartpole_Dynamics::cartpole_dynamics_model(const Matrix<T, Dynamic, 1>& x,
    const Matrix<T, Dynamic, 1>& u)
{
  Matrix<T, Dynamic, 1> dx(nx);  // 状态变量的变化率
  // x_cart的动力学方程
  dx[0] = x[2];
  // theta的动力学方程
  dx[1] = x[3];

  T s1, c1;
  s1 = sin(x[1]);
  c1 = cos(x[1]);

  // x_cart_dot的动力学方程
  dx[2] = (u[0] + m_pole * l * s1 * x[3] * x[3] - 3 * m_pole * g * s1 * c1 / 4) /
          (m_cart + m_pole - 3 * m_pole * c1 * c1 / 4);
  // std::cout << "m_pole = " << m_pole << std::endl;
  // std::cout << "m_cart = " << m_cart << std::endl;
  // std::cout << "l = " << l << std::endl;
  // theta_dot的动力学方程
  dx[3] = 3 * (g * s1 - dx[2] * c1) / 4 / l;

  return dx;
}

template <typename T>
Matrix<T, Dynamic, 1> Cartpole_Dynamics::cartpole_dynamics_integrate(const Matrix<T, Dynamic, 1>& x,
    const Matrix<T, Dynamic, 1>& u)
{
  Matrix<T, Dynamic, 1> k1, k2, k3, k4, x_next;
  // 计算k1
  k1 = cartpole_dynamics_model<T>(x, u) * dt;
  // 计算k2
  k2 = cartpole_dynamics_model<T>(x + 0.5 * k1, u) * dt;
  // 计算k3
  k3 = cartpole_dynamics_model<T>(x + 0.5 * k2, u) * dt;
  // 计算k4
  k4 = cartpole_dynamics_model<T>(x + k3, u) * dt;

  // 更新状态
  x_next = x + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

  return x_next.segment(0, 4);
}

template Matrix<double, Dynamic, 1> Cartpole_Dynamics::cartpole_dynamics_model(const Matrix<double, Dynamic, 1>&,
    const Matrix<double, Dynamic, 1>&);
template Matrix<AD<double>, Dynamic, 1> Cartpole_Dynamics::cartpole_dynamics_model(
    const Matrix<AD<double>, Dynamic, 1>&,
    const Matrix<AD<double>, Dynamic, 1>&);
template Matrix<double, Dynamic, 1> Cartpole_Dynamics::cartpole_dynamics_integrate(const Matrix<double, Dynamic, 1>&,
    const Matrix<double, Dynamic, 1>&);
template Matrix<AD<double>, Dynamic, 1> Cartpole_Dynamics::cartpole_dynamics_integrate(
    const Matrix<AD<double>, Dynamic, 1>&,
    const Matrix<AD<double>, Dynamic, 1>&);
