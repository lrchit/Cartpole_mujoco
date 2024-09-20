
#include <iLQR.h>

iLQR_Solver::iLQR_Solver(YAML::Node config, std::shared_ptr<Dynamics> dynamics_model) {
  dt = config["dt"].as<double>();
  double Tfinal = config["Tfinal"].as<double>();
  Nt = (int)(Tfinal / dt) + 1;

  // dynamics
  for (int i = 0; i < Nt - 1; ++i) {
    dynamics_model_.push_back(dynamics_model);
  }

  nx = dynamics_model->get_nx();
  nu = dynamics_model->get_nu();
  Q = dynamics_model->getRunningStateCostMatrix();
  Qn = dynamics_model->getTerminalStateCostMatrix();
  R = dynamics_model->getRunningInputCostMatrix();

  sigma = config["sigma"].as<double>();
  beta = config["beta"].as<double>();
  tolerance = config["tolerance"].as<double>();

  verbose_cal_time = config["verbose_cal_time"].as<bool>();

  p.resize(Nt);
  P.resize(Nt);
  d.resize(Nt - 1);
  K.resize(Nt - 1);
  derivatives.lx.resize(Nt);
  derivatives.lu.resize(Nt - 1);
  derivatives.lxx.resize(Nt);
  derivatives.lux.resize(Nt - 1);
  derivatives.luu.resize(Nt - 1);
  derivatives.fx.resize(Nt - 1);
  derivatives.fu.resize(Nt - 1);
  xtraj.resize(Nt);
  utraj.resize(Nt - 1);
  xgoal.resize(Nt);
  for (int i = 0; i < Nt - 1; ++i) {
    p[i].setZero(nx, nu);
    P[i].setZero(nx, nx);
    d[i].setZero(nu, nu);
    K[i].setZero(nu, nx);
    xtraj[i].setZero(nx);
    utraj[i].setZero(nu);
    xgoal[i].setZero(nx);
  }
  p[Nt - 1].setZero(nx, nu);
  P[Nt - 1].setZero(nx, nx);
  xtraj[Nt - 1].setZero(nx);
  xgoal[Nt - 1].setZero(nx);
}

iLQR_Solver::~iLQR_Solver() {
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

// cost derivatives.function
double iLQR_Solver::calCost(const std::vector<ocs2::vector_t>& _xtraj, const std::vector<ocs2::vector_t>& _utraj) {
  double J = 0.0;
  for (int k = 0; k < (Nt - 1); ++k)
    J += (0.5 * (_xtraj[k] - xgoal[k]).transpose() * Q * (_xtraj[k] - xgoal[k]) + 0.5 * _utraj[k].transpose() * R * _utraj[k]).value();
  J += (0.5 * (_xtraj[Nt - 1] - xgoal[Nt - 1]).transpose() * Qn * (_xtraj[Nt - 1] - xgoal[Nt - 1])).value();
  return J;
}

void iLQR_Solver::calDerivatives() {
  // #pragma omp parallel for num_threads(4)
  for (int k = (Nt - 2); k > -1; --k) {
    // Calculate derivatives
    derivatives.lx[k].noalias() = Q * (xtraj[k] - xgoal[k]);
    derivatives.lu[k].noalias() = R * utraj[k];
    derivatives.lxx[k].noalias() = Q;
    derivatives.lux[k].noalias() = ocs2::matrix_t::Zero(nu, nx);
    derivatives.luu[k].noalias() = R;

    // df/dx for A and df/du for B
    const ocs2::matrix_t jacobian = dynamics_model_[k]->getFirstDerivatives(xtraj[k], utraj[k]);
    derivatives.fx[k].noalias() = jacobian.leftCols(nx);
    derivatives.fu[k].noalias() = jacobian.rightCols(nu);
  }
  derivatives.lx[Nt - 1].noalias() = Qn * (xtraj[Nt - 1] - xgoal[Nt - 1]);
  derivatives.lxx[Nt - 1].noalias() = Qn;
}

double iLQR_Solver::backward_pass() {
  double delta_J = 0.0;
  p[Nt - 1].noalias() = derivatives.lx[Nt - 1];
  P[Nt - 1].noalias() = derivatives.lxx[Nt - 1];

  for (int k = (Nt - 2); k > -1; --k) {
    ddp_matrix.Qx.noalias() = (derivatives.lx[k] + derivatives.fx[k].transpose() * p[k + 1]).eval();
    ddp_matrix.Qu.noalias() = (derivatives.lu[k] + derivatives.fu[k].transpose() * p[k + 1]).eval();

    // iLQR (Gauss-Newton) version
    ddp_matrix.Qxx.noalias() = (derivatives.lxx[k] + derivatives.fx[k].transpose() * P[k + 1] * derivatives.fx[k]).eval();
    ddp_matrix.Quu.noalias() = (derivatives.luu[k] + derivatives.fu[k].transpose() * P[k + 1] * derivatives.fu[k]).eval();
    ddp_matrix.Quu_inverse = ddp_matrix.Quu.ldlt().solve(ocs2::matrix_t::Identity(nu, nu));
    ddp_matrix.Qux.noalias() = derivatives.lux[k] + (derivatives.fu[k].transpose() * P[k + 1] * derivatives.fx[k]).eval();
    ddp_matrix.Qxu.noalias() = ddp_matrix.Qux.transpose().eval();

    d[k].noalias() = (ddp_matrix.Quu_inverse * ddp_matrix.Qu).eval();
    K[k].noalias() = (ddp_matrix.Quu_inverse * ddp_matrix.Qux).eval();

    p[k].noalias() = (ddp_matrix.Qx - ddp_matrix.Qxu * d[k]).eval();
    P[k].noalias() = (ddp_matrix.Qxx - ddp_matrix.Qxu * K[k]).eval();

    delta_J += (ddp_matrix.Qu.transpose() * d[k]).value();
  }

  return delta_J;
}

// line search
double iLQR_Solver::line_search(double delta_J, double J) {
  double alpha = 1.0;
  double Jn = 1.0e+9;
  std::vector<ocs2::vector_t> xn(Nt);
  std::vector<ocs2::vector_t> un(Nt - 1);
  xn[0] = xtraj[0];

  while (Jn > (J - beta * alpha * delta_J)) {
    assert(alpha >= 1e-8; "line search failed to find a feasible rollout");
    for (int k = 0; k < (Nt - 1); ++k) {
      un[k] = utraj[k] - alpha * d[k] - K[k] * (xn[k] - xtraj[k]);
      xn[k + 1] = dynamics_model_[k]->getValue(xn[k], un[k]);
    }
    alpha = sigma * alpha;
    Jn = calCost(xn, un);
  }
  xtraj = xn;
  utraj = un;

  return Jn;
}

void iLQR_Solver::solve() {
  int iter = 0;
  double J = calCost(xtraj, utraj);
  double delta_J = 1.0e+9;
  std::vector<ocs2::scalar_t> derivativeTime;
  std::vector<ocs2::scalar_t> backwardPassTime;
  std::vector<ocs2::scalar_t> lineSeachTime;
  while (delta_J > tolerance) {
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

void iLQR_Solver::reset_solver(const ocs2::vector_t& xcur, const std::vector<ocs2::vector_t>& x_goal) {
  // Initial Rollout
  xtraj[0] = xcur;
  for (int k = 0; k < (Nt - 1); ++k) {
    utraj[k] = dynamics_model_[k]->getQuasiStaticInput(xcur);
    xtraj[k + 1] = dynamics_model_[k]->getValue(xtraj[k], utraj[k]);
  }

  // initialize K and d
  for (int i = 0; i < Nt - 1; ++i) {
    d[i].setOnes();
    K[i].setZero();
  }

  // reference
  set_reference(x_goal);
}

std::vector<ocs2::vector_t> iLQR_Solver::iLQR_algorithm(const ocs2::vector_t& xcur, const std::vector<ocs2::vector_t>& x_goal) {
  // reset
  reset_solver(xcur, x_goal);

  // DDP Algorithm
  solve();

  return utraj;
}

// 最优状态和控制轨迹
void iLQR_Solver::traj_plot() {
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
