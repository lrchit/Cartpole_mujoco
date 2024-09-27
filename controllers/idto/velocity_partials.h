#pragma once

#include <vector>

#include <Types.h>

namespace idto {
namespace optimizer {

/**
 * Struct storing gradients of generalized velocities (v) with respect to
 * generalized positions (q).
 *
 * TODO(vincekurtz): extend to quaternion DoFs, where these quantities are
 * different for each timestep, and include a factor of N+(q).
 */
template <typename T>
struct VelocityPartials {
  VelocityPartials(const int num_steps, const int nv, const int nq) {
    dvt_dqt.assign(num_steps + 1, ocs2::matrix_s_t<T>::Zero(nv, nq));
    dvt_dqm.assign(num_steps + 1, ocs2::matrix_s_t<T>::Zero(nv, nq));

    // Derivatives w.r.t. q(-1) are undefined
    dvt_dqm[0].setConstant(nv, nq, NAN);
  }
  // Partials of v_t w.r.t. q_t at each time step:
  //
  //    [d(v_0)/d(q_0), d(v_1)/d(q_1), ... , d(v_{num_steps})/d(q_{num_steps}) ]
  //
  std::vector<ocs2::matrix_s_t<T>> dvt_dqt;

  // Partials of v_t w.r.t. q_{t-1} at each time step:
  //
  //    [NaN, d(v_1)/d(q_0), ... , d(v_{num_steps})/d(q_{num_steps-1}) ]
  //
  std::vector<ocs2::matrix_s_t<T>> dvt_dqm;
};

template struct VelocityPartials<double>;

}  // namespace optimizer
}  // namespace idto
