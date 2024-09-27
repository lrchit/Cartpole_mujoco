
#include <iLQR.h>

iLQR_Solver::iLQR_Solver(YAML::Node config, std::shared_ptr<Dynamics> dynamics_model, std::shared_ptr<Cost> cost, const ocs2::matrix_t Kguess) {
  Nt = config["horizon"].as<int>() + 1;
  nx = config["nx"].as<double>();
  nu = config["nu"].as<double>();

  max_iteration = config["max_iteration"].as<double>();

  // dynamics and cost
  for (int i = 0; i < Nt; ++i) {
    dynamics_model_.push_back(dynamics_model);
    cost_.push_back(cost);
  }

  // feedback gain for initial guess
  K_guess = Kguess;

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
  xref.resize(Nt);
  for (int k = 0; k < Nt - 1; ++k) {
    p[k].setZero(nx, nu);
    P[k].setZero(nx, nx);
    d[k].setZero(nu, nu);
    K[k].setZero(nu, nx);
    xtraj[k].setZero(nx);
    utraj[k].setZero(nu);
    xref[k].setZero(nx);
  }
  p[Nt - 1].setZero(nx, nu);
  P[Nt - 1].setZero(nx, nx);
  xtraj[Nt - 1].setZero(nx);
  xref[Nt - 1].setZero(nx);
}

iLQR_Solver::~iLQR_Solver() {
  double derivativeTimeTotal = 0;
  double backwardPassTimeTotal = 0;
  double lineSeachTimeTotal = 0;
  for (int k = 0; k < derivativeTime_.size() - 1; ++k) {
    derivativeTimeTotal += derivativeTime_[k];
    backwardPassTimeTotal += backwardPassTime_[k];
    lineSeachTimeTotal += lineSeachTime_[k];
  }
  double totalTimeAverage = (derivativeTimeTotal + backwardPassTimeTotal + lineSeachTimeTotal) / derivativeTime_.size();
  if (verbose_cal_time) {
    std::cout << "################################################" << std::endl;
    std::cout << "  Average Time    =     " << totalTimeAverage << " ms " << std::endl;
    std::cout << "  derivation      =     " << derivativeTimeTotal / derivativeTime_.size() << " ms  "
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
  for (int k = 0; k < (Nt - 1); ++k) {
    J += cost_[k]->getValue(_xtraj[k], _utraj[k], xref[k]);
  }
  J += cost_[Nt - 1]->getValue(_xtraj[Nt - 1], xref[Nt - 1]);
  return J;
}

void iLQR_Solver::calDerivatives() {
  // #pragma omp parallel for num_threads(4)
  for (int k = (Nt - 2); k > -1; --k) {
    // Calculate derivatives
    std::tie(derivatives.lx[k], derivatives.lu[k]) = cost_[k]->getFirstDerivatives(xtraj[k], utraj[k], xref[k]);
    std::tie(derivatives.lxx[k], derivatives.lux[k], derivatives.luu[k]) = cost_[k]->getSecondDerivatives(xtraj[k], utraj[k], xref[k]);

    // df/dx for A and df/du for B
    std::tie(derivatives.fx[k], derivatives.fu[k]) = dynamics_model_[k]->getFirstDerivatives(xtraj[k], utraj[k]);
  }
  std::tie(derivatives.lx[Nt - 1], std::ignore) = cost_[Nt - 1]->getFirstDerivatives(xtraj[Nt - 1], xref[Nt - 1]);
  std::tie(derivatives.lxx[Nt - 1], std::ignore, std::ignore) = cost_[Nt - 1]->getSecondDerivatives(xtraj[Nt - 1], xref[Nt - 1]);
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
    ddp_matrix.Quu_inverse.noalias() = ddp_matrix.Quu.ldlt().solve(ocs2::matrix_t::Identity(nu, nu));
    ddp_matrix.Qux.noalias() = derivatives.lux[k] + (derivatives.fu[k].transpose() * P[k + 1] * derivatives.fx[k]).eval();
    ddp_matrix.Qxu.noalias() = ddp_matrix.Qux.transpose().eval();

    d[k].noalias() = (-ddp_matrix.Quu_inverse * ddp_matrix.Qu).eval();
    K[k].noalias() = (-ddp_matrix.Quu_inverse * ddp_matrix.Qux).eval();

    p[k].noalias() = (ddp_matrix.Qx + ddp_matrix.Qxu * d[k]).eval();
    P[k].noalias() = (ddp_matrix.Qxx + ddp_matrix.Qxu * K[k]).eval();

    delta_J -= 0.5 * (ddp_matrix.Qu.transpose() * d[k]).value();
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

  while (Jn > (J - beta * alpha * (1 - alpha / 2) * delta_J)) {
    if (alpha <= 1e-8) {
      std::cerr << "delta_J = " << delta_J << std::endl;
      std::cerr << "J = " << J << std::endl;
      throw std::runtime_error("line search failed to find a feasible rollout!");
    }
    for (int k = 0; k < (Nt - 1); ++k) {
      un[k] = utraj[k] + alpha * d[k] + K[k] * (xn[k] - xtraj[k]);
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
  while (delta_J > tolerance && iter < max_iteration) {
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
  for (int k = 0; k < iter - 1; ++k) {
    derivativeTimeTotal += derivativeTime[k];
    backwardPassTimeTotal += backwardPassTime[k];
    lineSeachTimeTotal += lineSeachTime[k];
  }
  derivativeTime_.push_back(derivativeTimeTotal);
  backwardPassTime_.push_back(backwardPassTimeTotal);
  lineSeachTime_.push_back(lineSeachTimeTotal);
}

void iLQR_Solver::launch_controller(const ocs2::vector_t& xcur, const std::vector<ocs2::vector_t>& x_ref) {
  // initial guess
  xtraj[0] = xcur;
  for (int k = 0; k < (Nt - 1); ++k) {
    utraj[k] = dynamics_model_[k]->getQuasiStaticInput(xtraj[k]);
    // utraj[k] += K_guess * (x_ref[k + 1] - xtraj[k]);
    xtraj[k + 1] = dynamics_model_[k]->getValue(xtraj[k], utraj[k]);
  }

  // reference
  xref = x_ref;

  // DDP Algorithm
  solve();
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
