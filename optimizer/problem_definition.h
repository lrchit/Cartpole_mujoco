#pragma once

#include <vector>

#include <Types.h>

namespace idto {
namespace optimizer {

/**
 * A struct for specifying the optimization problem
 *
 *    min x_err(T)'*Qf*x_err(T) + dt*sum{ x_err(t)'*Q*x_err(t) + u(t)'*R*u(t) }
 *    s.t. x(0) = x0
 *         multibody dynamics with contact
 *
 *  where x(t) = [q(t); v(t)], x_err(t) = x(t)-x_nom(t), and
 *  Q = [ Qq  0  ]
 *      [ 0   Qv ].
 */
struct ProblemDefinition {
  // Time horizon (number of steps) for the optimization problem
  int num_steps;

  // Initial generalized positions
  ocs2::vector_t q_init;

  // Initial generalized velocities
  ocs2::vector_t v_init;

  // Running cost coefficients for generalized positions
  // N.B. these weights are per unit of time
  // TODO(vincekurtz): consider storing these as ocs2::vector_t, assuming they are
  // diagonal, and using X.asDiagonal() when multiplying.
  ocs2::matrix_t Qq;

  // Running cost coefficients for generalized velocities
  // N.B. these weights are per unit of time
  ocs2::matrix_t Qv;

  // Terminal cost coefficients for generalized positions
  ocs2::matrix_t Qf_q;

  // Terminal cost coefficients for generalized velocities
  ocs2::matrix_t Qf_v;

  // Control cost coefficients
  // N.B. these weights are per unit of time
  ocs2::matrix_t R;

  // Target generalized positions at each time step
  std::vector<ocs2::vector_t> q_nom;

  // Target generalized velocities at each time step
  std::vector<ocs2::vector_t> v_nom;
};

}  // namespace optimizer
}  // namespace idto
