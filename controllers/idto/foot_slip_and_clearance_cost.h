#pragma once

#include <vector>

#include <Types.h>

namespace idto {
namespace optimizer {

template <typename T>
struct FootSlipAndClearanceCost {
  FootSlipAndClearanceCost(const int num_steps, const int nv, const int nq) {
    dcost_dq.assign(num_steps + 1, ocs2::vector_s_t<T>::Zero(nq));
    dcost_dv.assign(num_steps + 1, ocs2::vector_s_t<T>::Zero(nv));
    dcost_dqdq.assign(num_steps + 1, ocs2::matrix_s_t<T>::Zero(nq, nq));
    dcost_dvdv.assign(num_steps + 1, ocs2::matrix_s_t<T>::Zero(nv, nv));
    dcost_dqdv.assign(num_steps + 1, ocs2::matrix_s_t<T>::Zero(nq, nv));
  }

  // gradient
  std::vector<ocs2::vector_s_t<T>> dcost_dq;
  std::vector<ocs2::vector_s_t<T>> dcost_dv;

  // Hessian
  std::vector<ocs2::matrix_s_t<T>> dcost_dqdq;
  std::vector<ocs2::matrix_s_t<T>> dcost_dvdv;
  std::vector<ocs2::matrix_s_t<T>> dcost_dvdv;
};

template struct FootSlipAndClearanceCost<double>;

}  // namespace optimizer
}  // namespace idto
