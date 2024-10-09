
#include "hpipm_interface.h"

HpipmInterface::HpipmInterface(YAML::Node config) {
  nx_ = config["nx"].as<int>();
  nu_ = config["nu"].as<int>();
  horizon_ = config["horizon"].as<int>() + 1;

  nx = new int[horizon_];    // number of states
  nu = new int[horizon_];    // number of inputs
  nbx = new int[horizon_];   // number of state bounds
  nbu = new int[horizon_];   // number of input bounds
  ng = new int[horizon_];    // number of polytopic constras
  nsbx = new int[horizon_];  // number of slack variables on state
  nsbu = new int[horizon_];  // number of slack variables on input
  nsg = new int[horizon_];   // number of slack variables on polytopic constraints

  hA = new double*[horizon_];
  hB = new double*[horizon_];
  hb = new double*[horizon_];
  for (int k = 0; k < horizon_ - 1; ++k) {
    nx[k] = nx_;
    nu[k] = nu_;

    hb[k] = new double[nx_];
    std::fill(hb[k], hb[k] + nx_, 0.0);
  }
  nx[0] = 0;
  nx[horizon_ - 1] = nx_;
  nu[horizon_ - 1] = 0;

  hQ = new double*[horizon_];
  hS = new double*[horizon_];
  hR = new double*[horizon_];
  hq = new double*[horizon_];
  hr = new double*[horizon_];

  hlg = new double*[horizon_];
  hug = new double*[horizon_];
  hD = new double*[horizon_];
  hC = new double*[horizon_];

  hidxbx = new int*[horizon_];
  hlbx = new double*[horizon_];
  hubx = new double*[horizon_];
  hidxbu = new int*[horizon_];
  hlbu = new double*[horizon_];
  hubu = new double*[horizon_];

  hZl = new double*[horizon_];
  hZu = new double*[horizon_];
  hzl = new double*[horizon_];
  hzu = new double*[horizon_];

  hlls = new double*[horizon_];
  hlus = new double*[horizon_];

  hidxs = new int*[horizon_];

  hpipmWrappers.reset(new HpipmWrappers(config["dms"]["use_partial_condensed"].as<bool>()));

  delta_xtraj.resize(horizon_, ocs2::vector_t::Zero(nx_));
  delta_utraj.resize(horizon_ - 1, ocs2::vector_t::Zero(nu_));
}

void HpipmInterface::setCosts(int stage, ocs2::vector_t& q, ocs2::vector_t& r, ocs2::matrix_t& Q, ocs2::matrix_t& S, ocs2::matrix_t& R) {
  hQ[stage] = Q.data();
  hq[stage] = q.data();

  if (stage < horizon_) {
    hR[stage] = R.data();
    hS[stage] = S.data();
    hr[stage] = r.data();
  }
}

void HpipmInterface::setDynamics(int stage, ocs2::matrix_t& A, ocs2::matrix_t& B) {
  hA[stage] = A.data();
  hB[stage] = B.data();
}

// no bounds supported
void HpipmInterface::setBounds() {
  for (int k = 0; k < horizon_; k++) {
    nbx[k] = 0;
    hidxbx[k] = nullptr;
    hlbx[k] = nullptr;
    hubx[k] = nullptr;

    nbu[k] = 0;
    hidxbu[k] = nullptr;
    hlbu[k] = nullptr;
    hubu[k] = nullptr;
  }
}

void HpipmInterface::setBounds(int stage,
    ocs2::vector_t& lbx,
    ocs2::vector_t& ubx,
    Eigen::Matrix<int, Eigen::Dynamic, 1>& idxbx,
    ocs2::vector_t& lbu,
    ocs2::vector_t& ubu,
    Eigen::Matrix<int, Eigen::Dynamic, 1>& idxbu) {
  nbx[stage] = idxbx.cols();
  hidxbx[stage] = idxbx.data();
  hlbx[stage] = lbx.data();
  hubx[stage] = ubx.data();

  nbu[stage] = idxbu.cols();
  hidxbu[stage] = idxbu.data();
  hlbu[stage] = lbu.data();
  hubu[stage] = ubu.data();

  if (stage < horizon_ - 1) {
    nbu[stage] = idxbu.cols();
    hidxbu[stage] = idxbu.data();
    hlbu[stage] = lbu.data();
    hubu[stage] = ubu.data();
  } else {
    nbu[stage] = 0;
    hidxbu[stage] = nullptr;
    hlbu[stage] = nullptr;
    hubu[stage] = nullptr;
  }
}

// no polytopic constraint supported
void HpipmInterface::setPolytopicConstraints() {
  for (int k = 0; k < horizon_; k++) {
    ng[k] = 0;
    hC[k] = nullptr;
    hD[k] = nullptr;
    hlg[k] = nullptr;
    hug[k] = nullptr;
  }
}

// no soft constraint supported
void HpipmInterface::setSoftConstraints() {
  for (int k = 0; k < horizon_; k++) {
    nsbx[k] = 0;
    nsbu[k] = 0;
    nsg[k] = 0;

    hidxs[k] = nullptr;
    hlls[k] = nullptr;
    hlus[k] = nullptr;

    hZl[k] = nullptr;
    hZu[k] = nullptr;
    hzl[k] = nullptr;
    hzu[k] = nullptr;
  }
}

void HpipmInterface::solve() {
  // ocp qp dim
  hpipmWrappers->resetDim(horizon_);
  d_ocp_qp_dim_set_all(nx, nu, nbx, nbu, ng, nsbx, nsbu, nsg, &hpipmWrappers->dim);

  // ocp qp
  hpipmWrappers->resetQp();
  d_ocp_qp_set_all(
      hA, hB, hb, hQ, hS, hR, hq, hr, hidxbx, hlbx, hubx, hidxbu, hlbu, hubu, hC, hD, hlg, hug, hZl, hZu, hzl, hzu, hidxs, hlls, hlus, &hpipmWrappers->qp);

  // ocp qp ipm arg
  hpipmWrappers->resetArg();
  enum hpipm_mode mode = hpipm_mode::BALANCE;  // BALANCE, ROBUST, SPEED_ABS
  int iter_max = 30;
  double alpha_min = 1.0e-08;
  double mu0 = 1.0e+02;
  double tol_stat = 1.0e-4;
  double tol_eq = 1.0e-4;
  double tol_ineq = 1.0e-4;
  double tol_comp = 1.0e-4;
  double reg_prim = 1.0e-12;
  int warm_start = 0;
  int pred_corr = 1;
  int ric_alg = 0;
  int split_step = 1;
  d_ocp_qp_ipm_arg_set_default(mode, &hpipmWrappers->arg);
  d_ocp_qp_ipm_arg_set_iter_max(&iter_max, &hpipmWrappers->arg);
  d_ocp_qp_ipm_arg_set_alpha_min(&alpha_min, &hpipmWrappers->arg);
  d_ocp_qp_ipm_arg_set_mu0(&mu0, &hpipmWrappers->arg);
  d_ocp_qp_ipm_arg_set_tol_stat(&tol_stat, &hpipmWrappers->arg);
  d_ocp_qp_ipm_arg_set_tol_eq(&tol_eq, &hpipmWrappers->arg);
  d_ocp_qp_ipm_arg_set_tol_ineq(&tol_ineq, &hpipmWrappers->arg);
  d_ocp_qp_ipm_arg_set_tol_comp(&tol_comp, &hpipmWrappers->arg);
  d_ocp_qp_ipm_arg_set_reg_prim(&reg_prim, &hpipmWrappers->arg);
  d_ocp_qp_ipm_arg_set_warm_start(&warm_start, &hpipmWrappers->arg);
  d_ocp_qp_ipm_arg_set_pred_corr(&pred_corr, &hpipmWrappers->arg);
  d_ocp_qp_ipm_arg_set_ric_alg(&ric_alg, &hpipmWrappers->arg);
  d_ocp_qp_ipm_arg_set_split_step(&split_step, &hpipmWrappers->arg);

  // ocp qp ipm ws
  hpipmWrappers->resetWorkSpace();

  // ocp qp sol
  hpipmWrappers->resetQpSol();
  if (warm_start) {
    for (int k = 0; k < horizon_ - 1; ++k) {
      d_ocp_qp_sol_set_x(k + 1, delta_xtraj[k + 1].data(), &hpipmWrappers->qp_sol);
      d_ocp_qp_sol_set_u(k, delta_utraj[k].data(), &hpipmWrappers->qp_sol);
    }
  }

  d_ocp_qp_ipm_solve(&hpipmWrappers->qp, &hpipmWrappers->qp_sol, &hpipmWrappers->arg, &hpipmWrappers->workspace);
  // printf("exitflag %d\n", workspace.status);
  // printf("ipm iter = %d\n", workspace.iter);

  // extract and print solution
  for (int k = 0; k < horizon_ - 1; k++) {
    d_ocp_qp_sol_get_x(k, &hpipmWrappers->qp_sol, delta_xtraj[k + 1].data());
    d_ocp_qp_sol_get_u(k, &hpipmWrappers->qp_sol, delta_utraj[k].data());
  }
}
