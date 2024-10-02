
#pragma once

#include <stdlib.h>
#include <iostream>
#include <array>
#include <vector>
#include <Eigen/Dense>

#include "hpipm_d_ocp_qp_ipm.h"
#include "hpipm_d_ocp_qp_dim.h"
#include "hpipm_d_ocp_qp.h"
#include "hpipm_d_ocp_qp_sol.h"
#include "hpipm_timing.h"

#include <yaml-cpp/yaml.h>
#include <Types.h>

static constexpr double INF = 1E5;

struct HpipmBounds {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  std::vector<int> idx_u;
  std::vector<int> idx_x;
  std::vector<int> idx_s;
  std::vector<double> u_lower;
  std::vector<double> u_upper;
  std::vector<double> x_lower;
  std::vector<double> x_upper;
};

class HpipmInterface {
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  HpipmInterface(YAML::Node config);
  ~HpipmInterface() = default;

  void solve(std::vector<ocs2::vector_t>& xtraj, std::vector<ocs2::vector_t>& utraj);

  void setInitialState(const ocs2::vector_t& xcur);
  void setDynamics(const ocs2::matrix_t& A, const ocs2::matrix_t& B, const int index);
  void setCosts(const ocs2::vector_t& df_dx,
      const ocs2::vector_t& df_du,
      const ocs2::matrix_t& df_dxx,
      const ocs2::matrix_t& df_dux,
      const ocs2::matrix_t& df_duu,
      const int index);

  // these are not support currently
  void setBounds();
  void setPolytopicConstraints();
  void setSoftConstraints();

  private:
  int* nx;    // number of states
  int* nu;    // number of inputs
  int* nbx;   // number of state bounds
  int* nbu;   // number of input bounds
  int* ng;    // number of polytopic constraint*s
  int* nsbx;  // number of slack variables on state
  int* nsbu;  // number of slack variables on input
  int* nsg;   // number of slack variables on polytopic constraints

  // LTV dynamics
  // xk1 = Ak*xk + Bk*uk + bk
  double** hA;  // hA[k] = Ak
  double** hB;  // hB[k] = Bk
  double** hb;  // hb[k] = bk

  // Cost (without soft constraints)
  // min(x,u) sum(0<=k<=N) 0.5 * [xk;uk]^T * [Qk, Sk; Sk^T, Rk] * [xk;uk] + [qk; rk]^T * [xk;uk]
  double** hQ;
  double** hS;
  double** hR;
  double** hq;
  double** hr;

  // Polytopic constraints
  // g_(lower, k) <= Dk*xk + Ck*uk <= g_(upper, k)
  double** hlg;
  double** hug;
  double** hD;
  double** hC;

  // General bounds
  // x_(lower, k) <= xk <= x_(upper, k)
  // hbxid can be used to select bounds on a subset of states
  int** hidxbx;
  double** hlbx;
  double** hubx;
  // u_(lower, k) <= uk <= u_(upper, k)
  int** hidxbu;
  double** hlbu;
  double** hubu;

  // Cost (only soft constraints)
  // s_(lower, k) -> slack variable of lower polytopic constraint (3) + lower bounds
  // s_(upper, k) -> slack variable of upper polytopic constraint (4) + upper bounds
  // min(x,u) sum(0<=k<=N) 0.5 * [s_lower, k; s_upper, k]^T * [Z_lower, k, 0; 0, Z_upper, k] * [s_lower, k; s_upper, k]
  //                      + [z_lower, k; z_upper, k]^T * [s_lower, k; s_upper, k]
  double** hZl;
  double** hZu;
  double** hzl;
  double** hzu;

  // Bounds of soft constraint multipliers
  double** hlls;
  double** hlus;
  // index of the bounds and constraints that are softened
  // order is not really clear
  int** hidxs;

  int nx_;
  int nu_;
  int horizon_;

  int* initialStateBoundIndex;
};
