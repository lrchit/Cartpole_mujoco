
#include <iLQR.h>

Cartpole_iLQR::Cartpole_iLQR(std::string yaml_name) {
  nx = 4;
  nu = 1;

  Q.setZero(nx, nx);
  Qn.setZero(nx, nx);
  R.setZero(nu, nu);

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
  R(0, 0) = config["R"]["r1"].as<double>();

  dt = config["dt"].as<double>();
  step = config["step"].as<double>();
  Tfinal = config["Tfinal"].as<double>();
  Nt = (int)(Tfinal / dt) + 1;

  m_cart = config["m_cart"].as<double>();
  m_pole = config["m_pole"].as<double>();
  l = config["l"].as<double>();

  sigma = config["sigma"].as<double>();
  beta = config["beta"].as<double>();

  verbose_cal_time = config["verbose_cal_time"].as<bool>();

  Cartpole_Dynamics* cartpole_dynamics;
  cartpole_dynamics = new Cartpole_Dynamics(dt, m_cart, m_pole, l / 2);
  std::shared_ptr<ocs2::CppAdInterface> systemFlowMapCppAdInterfacePtr;
  auto systemFlowMapFunc = [&](const ocs2::ad_vector_t& x, ocs2::ad_vector_t& y) {
    ocs2::ad_vector_t state = x.head(4);
    ocs2::ad_vector_t input = x.tail(1);
    y = cartpole_dynamics->cartpole_dynamics_integrate<ocs2::ad_scalar_t>(state, input);
  };
  systemFlowMapCppAdInterfacePtr.reset(new ocs2::CppAdInterface(systemFlowMapFunc, 5, "cartpole_dynamics_systemFlowMap", "../cppad_generated"));
  if (true) {
    systemFlowMapCppAdInterfacePtr->createModels(ocs2::CppAdInterface::ApproximationOrder::First, true);
  } else {
    systemFlowMapCppAdInterfacePtr->loadModelsIfAvailable(ocs2::CppAdInterface::ApproximationOrder::First, true);
  }
  for (int i = 0; i < Nt; ++i) {
    systemFlowMapCppAdInterfacePtr_.push_back(systemFlowMapCppAdInterfacePtr);
  }

  p.resize(Nt);
  P.resize(Nt);
  d.resize(Nt - 1);
  K.resize(Nt - 1);
  q.resize(Nt - 1);
  r.resize(Nt - 1);
  A.resize(Nt - 1);
  B.resize(Nt - 1);
  xtraj.resize(Nt);
  utraj.resize(Nt - 1);
  for (int i = 0; i < Nt - 1; ++i) {
    p[i].setZero(nx, nu);
    P[i].setZero(nx, nx);
    d[i].setZero(nu, nu);
    K[i].setZero(nu, nx);
    xtraj[i].setZero(nx);
    utraj[i].setZero(nu);
  }
  p[Nt - 1].setZero(nx, nu);
  P[Nt - 1].setZero(nx, nx);
  xtraj[Nt - 1].setZero(nx);

  // Initial guess
  xgoal.setZero(nx);
}

Cartpole_iLQR::~Cartpole_iLQR() {
  double derivativeTimeTotal = 0;
  double backwardPassTimeTotal = 0;
  double lineSeachTimeTotal = 0;
  for (int i = 0; i < derivativeTime_.size() - 1; ++i) {
    derivativeTimeTotal += derivativeTime_[i];
    backwardPassTimeTotal += backwardPassTime_[i];
    lineSeachTimeTotal += lineSeachTime_[i];
  }
  double totalTimeAverage = (derivativeTimeTotal + backwardPassTimeTotal + lineSeachTimeTotal) / derivativeTime_.size();
  if (verbose_cal_time) {
    std::cout << "################################################" << std::endl;
    std::cout << "  Average Time    =     " << totalTimeAverage << " ms " << std::endl;
    std::cout << "  derivative      =     " << derivativeTimeTotal / derivativeTime_.size() << " ms  "
              << derivativeTimeTotal / derivativeTime_.size() / totalTimeAverage * 100 << "%" << std::endl;
    std::cout << "  backward pass   =     " << backwardPassTimeTotal / derivativeTime_.size() << " ms  "
              << backwardPassTimeTotal / derivativeTime_.size() / totalTimeAverage * 100 << "%" << std::endl;
    std::cout << "  line search     =     " << lineSeachTimeTotal / derivativeTime_.size() << " ms  "
              << lineSeachTimeTotal / derivativeTime_.size() / totalTimeAverage * 100 << "%" << std::endl;
    std::cout << "################################################" << std::endl;
  }
}

double Cartpole_iLQR::vector_max(const std::vector<ocs2::vector_t>& v) {
  double v_max = 0;
  for (int i = 0; i < v.size(); ++i) {
    if (v_max < v[i].cwiseAbs().maxCoeff())
      v_max = v[i].cwiseAbs().maxCoeff();
  }
  return v_max;
}

bool Cartpole_iLQR::isPositiveDefinite(const ocs2::matrix_t& M) {
  // 对于小尺寸矩阵，可以直接计算特征值
  Eigen::SelfAdjointEigenSolver<ocs2::matrix_t> es(M);
  const auto& eigenvalues = es.eigenvalues();
  // 检查所有特征值是否大于零
  for (int i = 0; i < eigenvalues.size(); ++i) {
    if (eigenvalues[i] <= 0) {  // 如果有任何一个特征值小于等于零
      return false;
    }
  }
  return true;  // 所有特征值都大于零，则矩阵是正定的
}

// cost function
double Cartpole_iLQR::cost(const std::vector<ocs2::vector_t>& _xtraj, const std::vector<ocs2::vector_t>& _utraj) {
  double J = 0.0;
  for (int k = 0; k < (Nt - 1); ++k)
    J += (0.5 * (_xtraj[k] - xgoal).transpose() * Q * (_xtraj[k] - xgoal) + 0.5 * _utraj[k].transpose() * R * _utraj[k]).value();
  J += (0.5 * (_xtraj[Nt - 1] - xgoal).transpose() * Qn * (_xtraj[Nt - 1] - xgoal)).value();
  return J;
}

void Cartpole_iLQR::calDerivatives() {
  // #pragma omp parallel for num_threads(4)
  for (int k = (Nt - 2); k > -1; --k) {
    // Calculate derivatives
    q[k].noalias() = Q * (xtraj[k] - xgoal);
    r[k].noalias() = R * utraj[k];

    // df/dx for A and df/du for B
    const ocs2::vector_t stateInput = (ocs2::vector_t(xtraj[k].rows() + utraj[k].rows()) << xtraj[k], utraj[k]).finished();
    const ocs2::matrix_t jacobian = systemFlowMapCppAdInterfacePtr_[k]->getJacobian(stateInput);

    // df/dx for A and df/du for B
    A[k].noalias() = jacobian.block(0, 0, 4, 4);
    B[k].noalias() = jacobian.block(0, 4, 4, 1);
  }
}

double Cartpole_iLQR::backward_pass() {
  double delta_J = 0.0;
  p[Nt - 1].noalias() = Qn * (xtraj[Nt - 1] - xgoal);
  P[Nt - 1].noalias() = Qn;

  ocs2::vector_t gx;
  ocs2::vector_t gu;
  ocs2::matrix_t Gxx;
  ocs2::vector_t Guu;
  ocs2::vector_t Gxu;
  ocs2::matrix_t Gux;
  ocs2::matrix_t G(nx + nu, nx + nu);
  ocs2::matrix_t tempAP;
  ocs2::matrix_t tempBP;
  ocs2::matrix_t tempKGuu;
  for (int k = (Nt - 2); k > -1; --k) {
    gx.noalias() = (q[k] + A[k].transpose() * p[k + 1]).eval();
    gu.noalias() = (r[k] + B[k].transpose() * p[k + 1]).eval();

    // iLQR (Gauss-Newton) version
    tempAP.noalias() = (A[k].transpose() * P[k + 1]).eval();
    tempBP.noalias() = (B[k].transpose() * P[k + 1]).eval();
    Gxx.noalias() = (Q + tempAP * A[k]).eval();
    Guu.noalias() = (R + tempBP * B[k]).eval();
    Gxu.noalias() = (tempAP * B[k]).eval();
    Gux.noalias() = (tempBP * A[k]).eval();

    G.block(0, 0, 4, 4).noalias() = Gxx;
    G.block(0, 4, 4, 1).noalias() = Gxu;
    G.block(4, 0, 1, 4).noalias() = Gux;
    G.block(4, 4, 1, 1).noalias() = Guu;

    d[k].noalias() = (Guu.inverse() * gu).eval();
    K[k].noalias() = (Guu.inverse() * Gux).eval();

    tempKGuu.noalias() = (K[k].transpose() * Guu).eval();
    p[k].noalias() = (gx - K[k].transpose() * gu + tempKGuu * d[k] - Gxu * d[k]).eval();
    P[k].noalias() = (Gxx + tempKGuu * K[k] - Gxu * K[k] - K[k].transpose() * Gux).eval();

    delta_J += (gu.transpose() * d[k]).value();
  }

  return delta_J;
}

// line search
double Cartpole_iLQR::line_search(double delta_J, double J) {
  double alpha = 1.0;
  double Jn = 1.0e+9;
  std::vector<ocs2::vector_t> xn(Nt);
  std::vector<ocs2::vector_t> un(Nt - 1);
  xn[0] = xtraj[0];

  while (Jn > (J - beta * alpha * delta_J)) {
    for (int k = 0; k < (Nt - 1); ++k) {
      un[k] = utraj[k] - alpha * d[k] - K[k] * (xn[k] - xtraj[k]);
      const ocs2::vector_t stateInput = (ocs2::vector_t(xn[k].rows() + un[k].rows()) << xn[k], un[k]).finished();
      xn[k + 1] = systemFlowMapCppAdInterfacePtr_[k]->getFunctionValue(stateInput);
    }
    alpha = sigma * alpha;
    Jn = cost(xn, un);
  }
  xtraj = xn;
  utraj = un;

  return Jn;
}

void Cartpole_iLQR::solve() {
  int iter = 0;
  double J = cost(xtraj, utraj);
  double delta_J = 1.0e+9;
  std::vector<ocs2::scalar_t> derivativeTime;
  std::vector<ocs2::scalar_t> backwardPassTime;
  std::vector<ocs2::scalar_t> lineSeachTime;
  while (delta_J > 1e-2) {
    iter++;
    // compute derivatives
    std::chrono::time_point<std::chrono::system_clock> derivative_time_record_start = std::chrono::system_clock::now();
    calDerivatives();
    std::chrono::time_point<std::chrono::system_clock> derivative_time_record_end = std::chrono::system_clock::now();
    derivativeTime.push_back(
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(derivative_time_record_end - derivative_time_record_start).count()) / 1.0e3);

    // Backward Pass
    std::chrono::time_point<std::chrono::system_clock> backward_pass_start = std::chrono::system_clock::now();
    delta_J = backward_pass();
    std::chrono::time_point<std::chrono::system_clock> backward_pass_end = std::chrono::system_clock::now();
    backwardPassTime.push_back(
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(backward_pass_end - backward_pass_start).count()) / 1.0e3);

    // Forward rollout with line search
    std::chrono::time_point<std::chrono::system_clock> line_search_start = std::chrono::system_clock::now();
    J = line_search(delta_J, J);
    std::chrono::time_point<std::chrono::system_clock> line_search_end = std::chrono::system_clock::now();
    lineSeachTime.push_back(static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(line_search_end - line_search_start).count()) / 1.0e3);

    Jtraj.push_back(J);
  }

  double derivativeTimeTotal = 0;
  double backwardPassTimeTotal = 0;
  double lineSeachTimeTotal = 0;
  for (int i = 0; i < iter - 1; ++i) {
    derivativeTimeTotal += derivativeTime[i];
    backwardPassTimeTotal += backwardPassTime[i];
    lineSeachTimeTotal += lineSeachTime[i];
  }
  derivativeTime_.push_back(derivativeTimeTotal);
  backwardPassTime_.push_back(backwardPassTimeTotal);
  lineSeachTime_.push_back(lineSeachTimeTotal);
}

void Cartpole_iLQR::reset_solver(const ocs2::vector_t& xcur) {
  if (first_run) {
    // Initial Rollout
    xtraj[0] = xcur;
    for (int k = 0; k < (Nt - 1); ++k) {
      utraj[k] = ocs2::vector_t::Zero(nu);
      const ocs2::vector_t stateInput = (ocs2::vector_t(xtraj[k].rows() + utraj[k].rows()) << xtraj[k], utraj[k]).finished();
      xtraj[k + 1] = systemFlowMapCppAdInterfacePtr_[k]->getFunctionValue(stateInput);
    }
    // first_run = false;
  } else {
    for (int k = 0; k < (Nt - 2); ++k) {
      xtraj[k] = xtraj[k + 1];
      utraj[k] = utraj[k + 1];
    }
    xtraj[0] = xcur;
  }

  // initialize K and d
  for (int i = 0; i < Nt - 1; ++i) {
    d[i].setOnes();
    K[i].setZero();
  }
}

void Cartpole_iLQR::iLQR_algorithm(const ocs2::vector_t& xcur) {
  // reset
  reset_solver(xcur);

  // DDP Algorithm
  solve();
}

// 最优状态和控制轨迹
void Cartpole_iLQR::traj_plot() {
  // 绘制J
  std::vector<double> J_length;
  for (size_t i = 0; i < Jtraj.size(); ++i) {
    J_length.push_back(i);
  }
  plt::named_plot("cost", J_length, Jtraj, "-y");
  plt::legend();
  plt::xlabel("iter");
  plt::ylabel("J");
  plt::title("cost traj for Cart-Pole Problem");
  plt::show();
}

void Cartpole_iLQR::get_control(mjData* d) {
  int waiting_time = 100;
  static int counter = 0;
  static int index = 0;

  if (counter < waiting_time) {
    counter++;
  } else if ((counter - waiting_time) % (int)(step / 0.002) == 0) {
    // std::cout << "********** iLQR *********" << std::endl;
    ocs2::vector_t _xcur(4);
    _xcur << d->sensordata[0], d->sensordata[1], d->sensordata[2], d->sensordata[3];
    iLQR_algorithm(_xcur);
    index = 0;
    counter++;
  } else {
    // 设置控制力
    d->ctrl[0] = fmin(fmax(utraj[index].value(), -100), 100);
    d->ctrl[1] = 0;  // pole没有直接控制
    counter++;
    if (counter % (int)(dt / 0.002) == 0) {
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
