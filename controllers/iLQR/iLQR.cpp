
#include <iLQR.h>

Cartpole_iLQR::Cartpole_iLQR(std::string yaml_name)
{
  nx = 4;
  nu = 1;

  Q.setZero(nx, nx);
  Qn.setZero(nx, nx);

  YAML::Node config = YAML::LoadFile(yaml_name);
  // Initialize weight matrix
  Q(0, 0) = config["Q"]["q1"].as<double>();
  Q(1, 1) = config["Q"]["q2"].as<double>();
  Q(2, 2) = config["Q"]["q3"].as<double>();
  Q(3, 3) = config["Q"]["q4"].as<double>();
  Qn(0, 0) = config["Qn"]["q1"].as<double>();
  Qn(1, 1) = config["Qn"]["q2"].as<double>();
  Qn(2, 2) = config["Qn"]["q3"].as<double>();
  Qn(3, 3) = config["Qn"]["q4"].as<double>();
  R = config["R"]["r1"].as<double>();

  dt = config["dt"].as<double>();
  step = config["step"].as<double>();
  Tfinal = config["Tfinal"].as<double>();
  Nt = (int)(Tfinal / dt) + 1;

  m_cart = config["m_cart"].as<double>();
  m_pole = config["m_pole"].as<double>();
  l = config["l"].as<double>();

  cartpole_dynamics = new Cartpole_Dynamics(dt, m_cart, m_pole, l / 2);

  p.resize(Nt);
  P.resize(Nt);
  d.resize(Nt - 1);
  K.resize(Nt - 1);

  // Initial guess
  xtraj.setZero(nx, Nt);
  utraj.setZero(Nt - 1);
  xgoal << 0, 0, 0, 0;
}

Cartpole_iLQR::~Cartpole_iLQR(){};

// cost function
double Cartpole_iLQR::stage_cost(const Matrix<double, 4, 1>& x, const double& u)
{
  return 0.5 * (x - xgoal).transpose() * Q * (x - xgoal) + 0.5 * R * u * u;
}
double Cartpole_iLQR::terminal_cost(const Matrix<double, 4, 1>& x)
{
  return 0.5 * (x - xgoal).transpose() * Qn * (x - xgoal);
}
double Cartpole_iLQR::cost(const Matrix<double, 4, Dynamic>& _xtraj, const Matrix<double, 1, Dynamic>& _utraj)
{
  double J = 0.0;
  for (int k = 0; k < (Nt - 1); ++k)
    J += stage_cost(_xtraj.col(k), _utraj(k));
  J += terminal_cost(_xtraj.col(Nt - 1));
  return J;
}

bool Cartpole_iLQR::isPositiveDefinite(const MatrixXd& M)
{
  // 对于小尺寸矩阵，可以直接计算特征值
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M);
  const auto& eigenvalues = es.eigenvalues();
  // 检查所有特征值是否大于零
  for (int i = 0; i < eigenvalues.size(); ++i)
  {
    if (eigenvalues[i] <= 0)
    {  // 如果有任何一个特征值小于等于零
      return false;
    }
  }
  return true;  // 所有特征值都大于零，则矩阵是正定的
}

double Cartpole_iLQR::backward_pass()
{
  double delta_J = 0.0;
  p[Nt - 1] = Qn * (xtraj.col(Nt - 1) - xgoal);
  P[Nt - 1] = Qn;

  Matrix<double, 4, 5> jacobian;
  Matrix<double, 4, 1> q;
  double r;
  Matrix<double, 4, 4> A;
  Matrix<double, 4, 1> B;
  Matrix<double, 4, 1> gx;
  double gu;
  Matrix<double, 4, 4> Gxx;
  double Guu;
  Matrix<double, 4, 1> Gxu;
  Matrix<double, 1, 4> Gux;
  MatrixXd G(5, 5);
  Matrix4d I;
  I.setIdentity();

  for (int k = (Nt - 2); k > -1; --k)
  {
    // Calculate derivatives
    q = Q * (xtraj.col(k) - xgoal);
    r = R * utraj[k];

    // df/dx for A and df/du for B
    jacobian = cartpole_dynamics->get_dynamics_jacobian(xtraj.col(k), utraj.col(k));
    A = jacobian.block(0, 0, 4, 4);
    B = jacobian.block(0, 4, 4, 1);

    gx = q + A.transpose() * p[k + 1];
    gu = r + B.transpose() * p[k + 1];

    // iLQR (Gauss-Newton) version
    Gxx = Q + A.transpose() * P[k + 1] * A;
    Guu = R + B.transpose() * P[k + 1] * B;
    Gxu = A.transpose() * P[k + 1] * B;
    Gux = B.transpose() * P[k + 1] * A;

    // regularization
    double beta = 0.1;

    G.block(0, 0, 4, 4) = Gxx;
    G.block(0, 4, 4, 1) = Gxu;
    G.block(4, 0, 1, 4) = Gux;
    G(4, 4) = Guu;

    while (!isPositiveDefinite(G))
    {
      Gxx += A.transpose() * beta * I * A;
      Guu += B.transpose() * beta * I * B;
      Gxu += A.transpose() * beta * I * B;
      Gux += B.transpose() * beta * I * A;
      beta *= 2;

      G.block(0, 0, 4, 4) = Gxx;
      G.block(0, 4, 4, 1) = Gxu;
      G.block(4, 0, 1, 4) = Gux;
      G(4, 4) = Guu;
    }

    d[k] = gu / Guu;
    K[k] = Gux / Guu;

    p[k] = gx - K[k].transpose() * gu + K[k].transpose() * Guu * d[k] - Gxu * d[k];
    P[k] = Gxx + K[k].transpose() * Guu * K[k] - Gxu * K[k] - K[k].transpose() * Gux;

    delta_J += gu * d[k];
  }

  return delta_J;
}

double Cartpole_iLQR::vector_max(const vector<double>& v)
{
  double v_max = 0;
  for (int i = 0; i < v.size(); ++i)
  {
    if (v_max < fabs(v[i]))
      v_max = fabs(v[i]);
  }
  return v_max;
}

void Cartpole_iLQR::iLQR_algorithm(const Matrix<double, 4, 1>& xcur, const double& ucur)
{
  // std::chrono::time_point<std::chrono::system_clock> t_start =
  //     std::chrono::system_clock::now();

  for (int i = 0; i < Nt - 1; ++i)
  {
    xtraj.col(i) = xcur;
    utraj(i) = ucur;
  }
  xtraj.col(Nt - 1) = xcur;

  // initialize p and d
  for (int i = 0; i < Nt - 1; ++i)
  {
    p[i].setOnes();
    d[i] = 1;
  }
  p[Nt - 1].setOnes();

  // Initial Rollout
  for (int k = 0; k < (Nt - 1); ++k)
  {
    xtraj.col(k + 1) = cartpole_dynamics->cartpole_dynamics_integrate<double>(xtraj.col(k), utraj.col(k));
  }
  double J = cost(xtraj, utraj);

  // DDP Algorithm
  double delta_J = 1.0;

  Matrix<double, 4, Dynamic> xn(nx, Nt);
  Matrix<double, 1, Dynamic> un(Nt - 1);

  int iter = 0;
  while (vector_max(d) > 1e-3)
  {
    iter++;
    // Backward Pass
    delta_J = backward_pass();

    // Forward rollout with line search
    xn.col(0) = xtraj.col(0);
    double alpha = 1.0;

    for (int k = 0; k < (Nt - 1); ++k)
    {
      un[k] = utraj[k] - alpha * d[k] - K[k] * (xn.col(k) - xtraj.col(k));
      xn.col(k + 1) = cartpole_dynamics->cartpole_dynamics_integrate<double>(xn.col(k), un.col(k));
    }
    double Jn = cost(xn, un);

    // line search
    while (Jn > (J - 1e-2 * alpha * delta_J))
    {
      alpha = 0.5 * alpha;
      for (int k = 0; k < (Nt - 1); ++k)
      {
        un[k] = utraj[k] - alpha * d[k] - K[k] * (xn.col(k) - xtraj.col(k));
        xn.col(k + 1) = cartpole_dynamics->cartpole_dynamics_integrate<double>(xn.col(k), un.col(k));
      }
      Jn = cost(xn, un);
    }

    J = Jn;
    xtraj = xn;
    utraj = un;
    Jtraj.push_back(J);
  }
  // std::chrono::time_point<std::chrono::system_clock> t_end =
  //     std::chrono::system_clock::now();
  // double time_record =
  //     std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start)
  //         .count();
  // std::cout << "cal_time: " << time_record / 1000 << "\n";
  // std::cout << "iLQR completed! iter = " << iter << std::endl;
}

// 最优状态和控制轨迹
void Cartpole_iLQR::traj_plot()
{
  // 绘制J
  std::vector<double> J_length;
  for (size_t i = 0; i < Jtraj.size(); ++i)
  {
    J_length.push_back(i);
  }
  plt::named_plot("cost", J_length, Jtraj, "-y");
  plt::legend();
  plt::xlabel("iter");
  plt::ylabel("J");
  plt::title("cost traj for Cart-Pole Problem");
  plt::show();
  // clear
  Jtraj.clear();

  // 绘制状态轨迹
  std::vector<double> time_series(Nt - 1);
  for (size_t i = 0; i < Nt - 1; ++i)
  {
    time_series[i] = i * dt;
  }
  std::vector<std::string> line_type = {"-b", "-k", "-r", "-g"};
  std::vector<std::string> plot_name = {"pos", "theta", "vel", "omega"};
  std::vector<double> state_series(Nt - 1);
  for (int k = 0; k < nx; ++k)
  {
    for (size_t i = 0; i < Nt - 1; ++i)
    {
      state_series[i] = xtraj(k, i);
    }
    plt::named_plot(plot_name[k], time_series, state_series, line_type[k]);
  }
  plt::legend();
  plt::xlabel("Time");
  plt::ylabel("Value");
  plt::title("Optimal State Trajectories for Cart-Pole Problem");
  plt::show();

  // 绘制控制轨迹
  std::vector<double> control_series(Nt - 1);
  for (int i = 0; i < Nt - 1; ++i)
  {
    control_series[i] = utraj(i);
  }
  plt::named_plot("control", time_series, control_series, "r--");
  plt::legend();
  plt::xlabel("Time");
  plt::ylabel("Value");
  plt::title("Optimal Control Trajectories for Cart-Pole Problem");
  plt::show();
}

void Cartpole_iLQR::get_control(mjData* d)
{
  int waiting_time = 100;
  static int counter = 0;
  static int index = 0;

  if (counter < waiting_time)
  {
    counter++;
  }
  else if ((counter - waiting_time) % (int)(step / 0.002) == 0)
  {
    // std::cout << "********** iLQR *********" << std::endl;
    Matrix<double, 4, 1> _xcur =
        Matrix<double, 4, 1>(d->sensordata[0], d->sensordata[1], d->sensordata[2], d->sensordata[3]);
    double _ucur = 1e-4;
    iLQR_algorithm(_xcur, _ucur);
    index = 0;
    counter++;
  }
  else
  {
    // 设置控制力
    d->ctrl[0] = fmin(fmax(utraj[index], -100), 100);
    d->ctrl[1] = 0;  // pole没有直接控制
    counter++;
    if (counter % (int)(dt / 0.002) == 0)
    {
      index++;
    }
  }

  // // plot
  // if (counter < waiting_time) {
  //   counter++;
  // } else if (counter == waiting_time) {
  //   Matrix<double,4,1> _xcur = Matrix<double,4,1>(d->sensordata[0], d->sensordata[1],
  //                             d->sensordata[2], d->sensordata[3]);
  //   double _ucur = 1e-4;
  //   iLQR_algorithm(_xcur, _ucur);
  //   std::cout << "xcur = " << _xcur.transpose() << "\n ";
  //   traj_plot();
  //   counter++;
  // } else if ((counter - waiting_time) % (int)(Tfinal / 0.002) != 0) {
  //   // 设置控制力
  //   d->ctrl[0] = fmin(fmax(utraj[index], -100), 100);
  //   d->ctrl[1] = 0; // pole没有直接控制
  //   counter++;
  //   if (counter % (int)(dt / 0.002) == 0) {
  //     std::cout << "utraj_" << index << " = " << utraj[index] << "\n ";
  //     index++;
  //   }
  // } else {
  //   pd_controller(d);
  // }
}
