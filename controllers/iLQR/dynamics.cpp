
#include <dynamics.h>

Cartpole_Dynamics::Cartpole_Dynamics(double _dt, double _m_cart, double _m_pole,
                                     double _l) {

  dt = _dt;
  nx = 4;
  nu = 1;
  m_cart = _m_cart;
  m_pole = _m_pole;
  l = _l;
  g = 9.81;

  vector<AD<double>> x_ad(nx + nu);
  // declare independent variables and start recording operation sequence
  CppAD::Independent(x_ad);
  // 创建AD函数对象，表示离散化后的动力学方程
  vector<AD<double>> x_next_ad = cartpole_dynamics_discrete(x_ad);

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
  f.for_jac_sparsity(pattern_in, transpose, dependency, internal_bool,
                     pattern_jac);
  // compute entire forward mode Jacobian to initialize "work"
  subset = sparse_rcv<s_vector, d_vector>(pattern_jac);
  coloring = "cppad";
  group_max = 100;
  vector<double> x_cur = {1, 1, 1, 1, 1};
  f.sparse_jac_for(group_max, x_cur, subset, pattern_jac, coloring, work);
}

Cartpole_Dynamics::~Cartpole_Dynamics(){};

// continuous dynamics
vector<AD<double>>
Cartpole_Dynamics::cartpole_dynamics_continuous(const vector<AD<double>> &x) {
  vector<AD<double>> dx(nx); // 状态变量的变化率
  // x_cart的动力学方程
  dx[0] = x[2];
  // theta的动力学方程
  dx[1] = x[3];

  AD<double> s1, c1;
  s1 = sin(x[1]);
  c1 = cos(x[1]);

  // x_cart_dot的动力学方程
  dx[2] =
      (x[4] + m_pole * l * s1 * x[3] * x[3] - 3 * m_pole * g * s1 * c1 / 4) /
      (m_cart + m_pole - 3 * m_pole * c1 * c1 / 4);
  // theta_dot的动力学方程
  dx[3] = 3 * (g * s1 - dx[2] * c1) / 4 / l;

  return dx;
}
// discretizised dynamics
vector<AD<double>>
Cartpole_Dynamics::cartpole_dynamics_discrete(const vector<AD<double>> &x) {
  vector<AD<double>> k1(nx), k2(nx), k3(nx), k4(nx);
  vector<AD<double>> x_next(nx);

  vector<AD<double>> x_temp(nx + nu);
  for (int i = 0; i < nu; ++i) {
    x_temp[nx + i] = x[nx + i];
  }
  // 计算k1
  k1 = cartpole_dynamics_continuous(x);
  // 计算k2
  for (int i = 0; i < nx; ++i) {
    x_temp[i] = x[i] + 0.5 * k1[i] * dt;
  }
  k2 = cartpole_dynamics_continuous(x_temp);
  // 计算k3
  for (int i = 0; i < nx; ++i) {
    x_temp[i] = x[i] + 0.5 * k2[i] * dt;
  }
  k3 = cartpole_dynamics_continuous(x_temp);
  // 计算k4
  for (int i = 0; i < nx; ++i) {
    x_temp[i] = x[i] + k3[i] * dt;
  }
  k4 = cartpole_dynamics_continuous(x_temp);

  // 更新状态
  for (int i = 0; i < nx; ++i) {
    x_next[i] =
        x[i] + (1.0 / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) * dt;
  }

  return x_next;
}
// compute jacobian
Matrix<double, 4, 5> Cartpole_Dynamics::jacobian_cartpole_dynamics_discrete(
    const Vector<double, 4> &_x, const double &_u) {

  Matrix<double, 4, 5> Jacobian;
  vector<double> x_cur(nx + nu);
  for (int i = 0; i < nx; ++i) {
    x_cur[i] = _x[i];
  }
  x_cur[nx] = _u;

  // std::chrono::time_point<std::chrono::system_clock> t_start =
  //     std::chrono::system_clock::now();

  f.sparse_jac_for(group_max, x_cur, subset, pattern_jac, coloring, work);
  Jacobian.setZero();
  for (int i = 0; i < pattern_jac.row().size(); ++i) {
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

// rollout dynamics
Vector<double, 5>
Cartpole_Dynamics::cartpole_dynamics_model(const Vector<double, 5> &x) {
  Vector<double, 5> dx; // 状态变量的变化率
  // x_cart的动力学方程
  dx[0] = x[2];
  // theta的动力学方程
  dx[1] = x[3];

  double s1, c1;
  s1 = sin(x[1]);
  c1 = cos(x[1]);

  // x_cart_dot的动力学方程
  dx[2] =
      (x[4] + m_pole * l * s1 * x[3] * x[3] - 3 * m_pole * g * s1 * c1 / 4) /
      (m_cart + m_pole - 3 * m_pole * c1 * c1 / 4);
  // std::cout << "m_pole = " << m_pole << std::endl;
  // std::cout << "m_cart = " << m_cart << std::endl;
  // std::cout << "l = " << l << std::endl;
  // theta_dot的动力学方程
  dx[3] = 3 * (g * s1 - dx[2] * c1) / 4 / l;
  dx[4] = 0;

  return dx;
}
Vector<double, 4>
Cartpole_Dynamics::cartpole_dynamics_rollout(const Vector<double, 4> &_x,
                                             const double &_u) {
  Vector<double, 5> x, k1, k2, k3, k4, x_next;
  x.segment(0, 4) = _x;
  x(4) = _u;
  // 计算k1
  k1 = cartpole_dynamics_model(x) * dt;
  // 计算k2
  k2 = cartpole_dynamics_model(x + 0.5 * k1) * dt;
  // 计算k3
  k3 = cartpole_dynamics_model(x + 0.5 * k2) * dt;
  // 计算k4
  k4 = cartpole_dynamics_model(x + k3) * dt;

  // 更新状态
  x_next = x + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

  return x_next.segment(0, 4);
}
