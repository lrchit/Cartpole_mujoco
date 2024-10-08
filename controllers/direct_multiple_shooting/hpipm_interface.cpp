
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

  initialStateBoundIndex = new int[nx_];
  for (int i = 0; i < nx_; ++i) {
    initialStateBoundIndex[i] = i;
  }

  delta_xtraj.resize(horizon_, ocs2::vector_t::Zero(nx_));
  delta_utraj.resize(horizon_ - 1, ocs2::vector_t::Zero(nu_));
}

void HpipmInterface::setCosts(ocs2::vector_t& q, ocs2::vector_t& r, ocs2::matrix_t& Q, ocs2::matrix_t& S, ocs2::matrix_t& R, int index) {
  hQ[index] = Q.data();
  hq[index] = q.data();

  if (index < horizon_) {
    hR[index] = R.data();
    hS[index] = S.data();
    hr[index] = r.data();
  }
}

void HpipmInterface::setDynamics(ocs2::matrix_t& A, ocs2::matrix_t& B, ocs2::vector_t& b, int index) {
  hA[index] = A.data();
  hB[index] = B.data();
  hb[index] = b.data();
}

// no bounds supported
void HpipmInterface::setBounds() {
  for (int k = 0; k < horizon_; k++) {
    nbu[k] = 0;
    hidxbu[k] = nullptr;
    hlbu[k] = nullptr;
    hubu[k] = nullptr;

    nbx[k] = 0;
    hidxbx[k] = nullptr;
    hlbx[k] = nullptr;
    hubx[k] = nullptr;
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
  hpipm_size_t dim_size = d_ocp_qp_dim_memsize(horizon_ - 1);
  void* dim_mem = malloc(dim_size);
  struct d_ocp_qp_dim dim;
  d_ocp_qp_dim_create(horizon_ - 1, &dim, dim_mem);
  d_ocp_qp_dim_set_all(nx, nu, nbx, nbu, ng, nsbx, nsbu, nsg, &dim);

  // ocp qp
  hpipm_size_t qp_size = d_ocp_qp_memsize(&dim);
  void* qp_mem = malloc(qp_size);
  struct d_ocp_qp qp;
  d_ocp_qp_create(&dim, &qp, qp_mem);
  d_ocp_qp_set_all(hA, hB, hb, hQ, hS, hR, hq, hr, hidxbx, hlbx, hubx, hidxbu, hlbu, hubu, hC, hD, hlg, hug, hZl, hZu, hzl, hzu, hidxs, hlls, hlus, &qp);

  // ocp qp ipm arg
  hpipm_size_t ipm_arg_size = d_ocp_qp_ipm_arg_memsize(&dim);
  void* ipm_arg_mem = malloc(ipm_arg_size);
  struct d_ocp_qp_ipm_arg arg;
  d_ocp_qp_ipm_arg_create(&dim, &arg, ipm_arg_mem);
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
  d_ocp_qp_ipm_arg_set_default(mode, &arg);
  d_ocp_qp_ipm_arg_set_iter_max(&iter_max, &arg);
  d_ocp_qp_ipm_arg_set_alpha_min(&alpha_min, &arg);
  d_ocp_qp_ipm_arg_set_mu0(&mu0, &arg);
  d_ocp_qp_ipm_arg_set_tol_stat(&tol_stat, &arg);
  d_ocp_qp_ipm_arg_set_tol_eq(&tol_eq, &arg);
  d_ocp_qp_ipm_arg_set_tol_ineq(&tol_ineq, &arg);
  d_ocp_qp_ipm_arg_set_tol_comp(&tol_comp, &arg);
  d_ocp_qp_ipm_arg_set_reg_prim(&reg_prim, &arg);
  d_ocp_qp_ipm_arg_set_warm_start(&warm_start, &arg);
  d_ocp_qp_ipm_arg_set_pred_corr(&pred_corr, &arg);
  d_ocp_qp_ipm_arg_set_ric_alg(&ric_alg, &arg);
  d_ocp_qp_ipm_arg_set_split_step(&split_step, &arg);

  // ocp qp ipm ws
  hpipm_size_t ipm_size = d_ocp_qp_ipm_ws_memsize(&dim, &arg);
  void* ipm_mem = malloc(ipm_size);
  struct d_ocp_qp_ipm_ws workspace;
  d_ocp_qp_ipm_ws_create(&dim, &arg, &workspace, ipm_mem);

  // ocp qp sol
  hpipm_size_t qp_sol_size = d_ocp_qp_sol_memsize(&dim);
  void* qp_sol_mem = malloc(qp_sol_size);
  struct d_ocp_qp_sol qp_sol;
  d_ocp_qp_sol_create(&dim, &qp_sol, qp_sol_mem);
  if (warm_start) {
    for (int k = 0; k < horizon_ - 1; ++k) {
      d_ocp_qp_sol_set_x(k + 1, delta_xtraj[k + 1].data(), &qp_sol);
      d_ocp_qp_sol_set_u(k, delta_utraj[k].data(), &qp_sol);
    }
  }

  d_ocp_qp_ipm_solve(&qp, &qp_sol, &arg, &workspace);
  // printf("exitflag %d\n", workspace.status);
  // printf("ipm iter = %d\n", workspace.iter);

  // extract and print solution
  for (int k = 0; k < horizon_ - 1; k++) {
    d_ocp_qp_sol_get_x(k, &qp_sol, delta_xtraj[k + 1].data());
    d_ocp_qp_sol_get_u(k, &qp_sol, delta_utraj[k].data());
  }

  free(dim_mem);
  free(qp_mem);
  free(qp_sol_mem);
  free(ipm_arg_mem);
  free(ipm_mem);
}
