
#include <hpipm_wrapper.h>

HpipmWrappers::HpipmWrappers(bool use_partial_cond)
    : dim(),
      dim_mem(nullptr),
      dim_memsize(0),
      horizon_(0),
      qp(),
      qp_mem(nullptr),
      qp_memsize(0),
      arg(),
      arg_mem(nullptr),
      arg_memsize(0),
      workspace(),
      workspace_mem(nullptr),
      workspace_memsize(0),
      qp_sol(),
      qp_sol_mem(nullptr),
      qp_sol_memsize(0),
      use_partial_cond_(use_partial_cond) {}

void HpipmWrappers::resetDim(int horizon) {
  hpipm_size_t new_memsize = d_ocp_qp_dim_memsize(horizon - 1);
  if (dim_mem != nullptr && new_memsize > dim_memsize) {
    free(dim_mem);
    dim_mem = nullptr;
  }
  dim_memsize = std::max(dim_memsize, new_memsize);
  if (dim_mem == nullptr) {
    dim_mem = malloc(dim_memsize);
  }
  if (horizon_ != horizon) {
    horizon_ = horizon;
    d_ocp_qp_dim_create(horizon - 1, &dim, dim_mem);
  }
}

void HpipmWrappers::resetQp() {
  const hpipm_size_t new_memsize = d_ocp_qp_memsize(&dim);
  if (qp_mem != nullptr && new_memsize > qp_memsize) {
    free(qp_mem);
    qp_mem = nullptr;
  }
  qp_memsize = std::max(qp_memsize, new_memsize);
  if (qp_mem == nullptr) {
    qp_mem = malloc(qp_memsize);
  }
  d_ocp_qp_create(&dim, &qp, qp_mem);
}

void HpipmWrappers::resetArg() {
  const hpipm_size_t new_memsize = d_ocp_qp_ipm_arg_memsize(&dim);
  if (arg_mem && new_memsize > arg_memsize) {
    free(arg_mem);
    arg_mem = nullptr;
  }
  arg_memsize = std::max(arg_memsize, new_memsize);
  if (!arg_mem) {
    arg_mem = malloc(arg_memsize);
    d_ocp_qp_ipm_arg_create(&dim, &arg, arg_mem);
  }
}

void HpipmWrappers::resetWorkSpace() {
  const hpipm_size_t new_memsize = d_ocp_qp_ipm_ws_memsize(&dim, &arg);
  if (workspace_mem != nullptr && new_memsize > workspace_memsize) {
    free(workspace_mem);
    workspace_mem = nullptr;
  }
  workspace_memsize = std::max(workspace_memsize, new_memsize);
  if (workspace_mem == nullptr) {
    workspace_mem = malloc(workspace_memsize);
  }
  d_ocp_qp_ipm_ws_create(&dim, &arg, &workspace, workspace_mem);
}

void HpipmWrappers::resetQpSol() {
  const hpipm_size_t new_memsize = d_ocp_qp_sol_memsize(&dim);
  if (qp_sol_mem != nullptr && new_memsize > qp_sol_memsize) {
    free(qp_sol_mem);
    qp_sol_mem = nullptr;
  }
  qp_sol_memsize = std::max(qp_sol_memsize, new_memsize);
  if (qp_sol_mem == nullptr) {
    qp_sol_mem = malloc(qp_sol_memsize);
  }
  d_ocp_qp_sol_create(&dim, &qp_sol, qp_sol_mem);
}