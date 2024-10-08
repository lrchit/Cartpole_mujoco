
#pragma once

#include <iostream>

#include "hpipm_d_ocp_qp_ipm.h"
#include "hpipm_d_ocp_qp_dim.h"
#include "hpipm_d_ocp_qp.h"
#include "hpipm_d_ocp_qp_sol.h"
#include "hpipm_timing.h"

struct HpipmWrappers {
  HpipmWrappers(bool use_partial_cond);
  void resetDim(int horizon);
  void resetQp();
  void resetArg();
  void resetWorkSpace();
  void resetQpSol();

  d_ocp_qp_dim dim;
  hpipm_size_t dim_memsize;
  void* dim_mem;
  int horizon_;

  d_ocp_qp qp;
  hpipm_size_t qp_memsize;
  void* qp_mem;

  d_ocp_qp_ipm_arg arg;
  hpipm_size_t arg_memsize;
  void* arg_mem;

  d_ocp_qp_ipm_ws workspace;
  hpipm_size_t workspace_memsize;
  void* workspace_mem;

  d_ocp_qp_sol qp_sol;
  hpipm_size_t qp_sol_memsize;
  void* qp_sol_mem;

  bool use_partial_cond_;
};