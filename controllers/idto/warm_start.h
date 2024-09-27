#pragma once

#include <pinocchio/fwd.hpp>  // always include it before any other header

#include <string>
#include <vector>

#include "trajectory_optimizer_state.h"

#include <Types.h>

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

namespace idto {
namespace optimizer {

/**
 * A container for holding variables that are re-used between MPC solves. Using
 * this container allows us to perform a full warm-start (including trust region
 * radius) for MPC, and allows us to avoid expensive state allocations between
 * MPC re-solves.
 */
class WarmStart {
  public:
  /**
   * The constructor allocates state variables.
   *
   * @param num_steps Number of steps in the trajectory optimization problem
   * @param num_eq_constraints Number of equality constraints
   * @param q_guess Initial guess of the sequence of generalized positions
   * @param Delta0 Initial trust region radius
   */
  WarmStart(const int num_steps,
      const pinocchio::ModelTpl<double>& model,
      const pinocchio::DataTpl<double>& data,
      const int num_eq_constraints,
      const std::vector<ocs2::vector_t> q_guess,
      const double Delta0)
      : state(num_steps, model, data, num_eq_constraints), scratch_state(num_steps, model, data, num_eq_constraints), Delta(Delta0) {
    // Set the initial guess
    state.set_q(q_guess);

    // Make sure the update vector is the right size
    const int num_vars = model.nq + (num_steps + 1);
    dq.resize(num_vars);
    dqH.resize(num_vars);
  }

  /**
   * Set the initial guess of the sequence of generalized positions.
   *
   * @param q_guess Initial guess of the sequence of generalized positions
   */
  void set_q(const std::vector<ocs2::vector_t>& q_guess) { state.set_q(q_guess); }

  /**
   * Get the initial guess of the sequence of generalized positions.
   */
  const std::vector<ocs2::vector_t>& get_q() const { return state.q(); }

  // A state variable to store q and everything that is computed from q
  TrajectoryOptimizerState<double> state;

  // A separate state variable for computations like L(q + dq)
  TrajectoryOptimizerState<double> scratch_state;

  // Trust region size
  double Delta;

  // The update vector q_{k+1} = q_k + dq
  ocs2::vector_t dq;

  // The full Newton step H * dqH = -g
  ocs2::vector_t dqH;
};

}  // namespace optimizer
}  // namespace idto
