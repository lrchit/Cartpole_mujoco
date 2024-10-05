#pragma once

#include <pinocchio/fwd.hpp>  // always include it before any other header

#include <memory>
#include <utility>
#include <vector>

#include "foot_slip_and_clearance_cost.h"
#include "inverse_dynamics_partials.h"
#include "penta_diagonal_matrix.h"
#include "trajectory_optimizer_workspace.h"
#include "velocity_partials.h"

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

#include <Types.h>

namespace idto {
namespace optimizer {

using internal::PentaDiagonalMatrix;

template <typename T>
struct Context {
  ~Context() {}
  Context(const int nv, const int nq) {
    q_.setZero(nq);
    v_.setZero(nv);
  }

  ocs2::vector_s_t<T> q_;
  ocs2::vector_s_t<T> v_;
};

/**
 * Struct for holding quantities that are computed from the optimizer state (q),
 * such as generalized velocities (v), accelerations (a), and forces (tau), as
 * well as various derivatives.
 *
 * We separate the trajectory data (v, a, tau) and derivative data (v_partials,
 * id_partials), since we often need to just use the trajectory data, for
 * instance to compute the total cost, without performing expensive updates of
 * the derivative data.
 */
template <typename T>
struct TrajectoryOptimizerCache {
  ~TrajectoryOptimizerCache() {}

  /**
   * Construct a cache for the trajectory optimizer. This is stored inside a
   * TrajectoryOptimizerState, and stores just about anything that you could
   * compute from the decision variables q. That includes the cost, the
   * gradient, the Hessian, derivatives, etc.
   *
   * @param num_steps number of time steps in the optimization problem
   * @param nv number of velocities for the plant
   * @param nq number of positions for the plant
   * @param num_eq_constraints number of equality constraints
   */
  TrajectoryOptimizerCache(const int num_steps, const int nv, const int nq, const int num_eq_constraints)
      : derivatives_data(num_steps, nv, nq),
        gradient((num_steps + 1) * nq),
        hessian(num_steps + 1, nq),
        scaled_hessian(num_steps + 1, nq),
        scaled_gradient((num_steps + 1) * nq),
        scale_factors((num_steps + 1) * nq),
        constraint_violation(num_eq_constraints),
        constraint_jacobian(num_eq_constraints, (num_steps + 1) * nq),
        lagrange_multipliers(num_eq_constraints),
        merit_gradient((num_steps + 1) * nq) {
    trajectory_data.v.assign(num_steps + 1, ocs2::vector_s_t<T>::Zero(nv));
    trajectory_data.a.assign(num_steps, ocs2::vector_s_t<T>::Zero(nv));
    inverse_dynamics_cache.tau.assign(num_steps, ocs2::vector_s_t<T>::Zero(nv));
    foot_slip_and_clearance_cost_cache.cost = 0;
    N_plus.assign(num_steps + 1, ocs2::matrix_s_t<T>::Zero(nv, nq));
    scale_factors.setConstant(1.0);
    constraint_jacobian.setZero();
    lagrange_multipliers.setZero();
  }

  TrajectoryOptimizerCache(const int num_steps, const pinocchio::ModelTpl<T>& model, const pinocchio::DataTpl<T>& data, const int num_eq_constraints)
      : TrajectoryOptimizerCache(num_steps, model.nv, model.nq, num_eq_constraints) {
    context_cache = std::make_unique<ContextCache>(num_steps, model, data);
  }

  // Cache state at each time step t as a Context so that multibody quantities
  // that the plant caches can be reused.
  struct ContextCache {
    ~ContextCache() {}

    // Creates a ContextCache for a problem with num_steps
    ContextCache(int num_steps, const pinocchio::ModelTpl<T>& model, const pinocchio::DataTpl<T>& data) {
      plant_contexts.reserve(num_steps + 1);
      for (int t = 0; t <= num_steps; ++t) {
        auto context_t = new Context<T>(model.nv, model.nq);
        plant_contexts.push_back(context_t);
      }
      up_to_date = false;
    }

    // std::vector of references to the MultibodyPlant contexts in
    // digram_contexts, of size num_steps+1.
    std::vector<Context<T>*> plant_contexts;

    bool up_to_date{false};
  };
  std::unique_ptr<ContextCache> context_cache;

  // Data used to compute the cost L(q)
  struct TrajectoryData {
    // Generalized velocities at each timestep
    // [v(0), v(1), ..., v(num_steps)]
    std::vector<ocs2::vector_s_t<T>> v;

    // Generalized accelerations at each timestep
    // [a(0), a(1), ..., a(num_steps-1)]
    std::vector<ocs2::vector_s_t<T>> a;

    bool up_to_date{false};
  } trajectory_data;

  struct InverseDynamicsCache {
    // Generalized forces at each timestep
    // [tau(0), tau(1), ..., tau(num_steps-1)]
    std::vector<ocs2::vector_s_t<T>> tau;

    bool up_to_date{false};
  } inverse_dynamics_cache;

  struct FootSlipAndClearanceCostCache {
    // Generalized forces at each timestep
    // [tau(0), tau(1), ..., tau(num_steps-1)]
    T cost;

    bool up_to_date{false};
  } foot_slip_and_clearance_cost_cache;

  // The mapping from qdot to v, v = N+(q)*qdot, at each time step
  std::vector<ocs2::matrix_s_t<T>> N_plus;
  bool n_plus_up_to_date{false};

  // Data used to construct the gradient ∇L and Hessian ∇²L approximation
  struct DerivativesData {
    DerivativesData(const int num_steps, const int nv, const int nq)
        : v_partials(num_steps, nv, nq), id_partials(num_steps, nv, nq), footSlipAndClearanceCost_Partials(num_steps, nv, nq) {}

    // Storage for dv(t)/dq(t) and dv(t)/dq(t-1)
    VelocityPartials<T> v_partials;

    // Storage for dtau(t)/dq(t-1), dtau(t)/dq(t), and dtau(t)/dq(t+1)
    InverseDynamicsPartials<T> id_partials;

    FootSlipAndClearanceCostPartials<T> footSlipAndClearanceCost_Partials;

    bool up_to_date{false};
  } derivatives_data;

  // The total cost L(q)
  T cost;
  bool cost_up_to_date{false};

  // The gradient of the unconstrained cost ∇L
  ocs2::vector_s_t<T> gradient;
  bool gradient_up_to_date{false};

  // Our Hessian approximation of the unconstrained cost ∇²L
  PentaDiagonalMatrix<T> hessian;
  bool hessian_up_to_date{false};

  // The scaled version of the Hessian, H̃ = DHD
  PentaDiagonalMatrix<T> scaled_hessian;
  bool scaled_hessian_up_to_date{false};

  // The scaled version of the gradient, g̃ = Dg
  ocs2::vector_s_t<T> scaled_gradient;
  bool scaled_gradient_up_to_date{false};

  // Vector of scaling factors D = 1/sqrt(diag(D))
  ocs2::vector_s_t<T> scale_factors;
  bool scale_factors_up_to_date{false};

  // Vector of equality constraint violations for the constraint h(q) = 0
  ocs2::vector_s_t<T> constraint_violation;
  bool constraint_violation_up_to_date{false};

  // Jacobian of equality constraints J = ∂h(q)/∂q
  ocs2::matrix_s_t<T> constraint_jacobian;
  bool constraint_jacobian_up_to_date{false};

  // Lagrange multipliers λ for the equality constraints h(q) = 0
  ocs2::vector_s_t<T> lagrange_multipliers;
  bool lagrange_multipliers_up_to_date{false};

  // Merit function ϕ = L + hᵀλ
  T merit;
  bool merit_up_to_date{false};

  // Gradient of the merit function g̃ = g + Jᵀλ
  ocs2::vector_s_t<T> merit_gradient;
  bool merit_gradient_up_to_date{false};
};

/**
 * Struct for storing the "state" of the trajectory optimizer.
 *
 * The only actual state is the sequence of generalized positions q at each
 * timestep. This class stores that directly, but also a "cache" of other values
 * computed from q, such as generalized velocities and forces at each timesteps,
 * relevant dynamics partials, etc.
 */
template <typename T>
class TrajectoryOptimizerState {
  public:
  // Not copyable.
  TrajectoryOptimizerState(const TrajectoryOptimizerState<T>&) = delete;
  void operator=(const TrajectoryOptimizerState<T>&) = delete;

  // We do allow to move it.
  TrajectoryOptimizerState(TrajectoryOptimizerState<T>&&) = default;
  TrajectoryOptimizerState<T>& operator=(TrajectoryOptimizerState<T>&&) = default;

  /**
   * Constructor which allocates things of the proper sizes.
   *
   * @param num_steps number of timesteps in the optimization problem
   * @param plant multibody plant system model
   * @param num_eq_constraints number of equality constraints
   */
  TrajectoryOptimizerState(const int num_steps, const pinocchio::ModelTpl<T>& model, const pinocchio::DataTpl<T>& data, const int num_eq_constraints)
      : workspace(num_steps, model, data), num_steps_(num_steps), nq_(model.nq), cache_(num_steps, model, data, num_eq_constraints) {
    const int nq = model.nq;
    q_.assign(num_steps + 1, ocs2::vector_s_t<T>::Zero(nq));
    per_timestep_workspace.assign(num_steps + 1, TrajectoryOptimizerWorkspace<T>(num_steps, model, data));
  }

  /** Getter for the sequence of generalized positions. */
  const std::vector<ocs2::vector_s_t<T>>& q() const { return q_; }

  /** Mutable reference to the sequence of generalized positions.
   @warning This method invalidates the cache. However be careful about holding
   onto the returned reference for too long, since updates to the values stored
   in this state through this reference will not cause cache invalidation.
   Consider using set_q() whenever possible instead. */
  std::vector<ocs2::vector_s_t<T>>& mutable_q() {
    invalidate_cache();
    return q_;
  }

  /**
   * Setter for the sequence of generalized positions. Invalidates the cache.
   *
   * @param q sequence of generalized positions at each time step
   */
  void set_q(const std::vector<ocs2::vector_s_t<T>>& q) {
    q_ = q;
    invalidate_cache();
  }

  /**
   * Update the sequence of generalized positions, q, by adding
   *
   *    q = q + dq,
   *
   * where dq is a large vector which stacks changes in each q[t].
   *
   * @param dq vector of changes in generalized positions
   */
  void AddToQ(const ocs2::vector_s_t<T>& dq) {
    assert(dq.size() == nq_ * (num_steps_ + 1));
    for (int t = 0; t <= num_steps_; ++t) {
      q_[t] += dq.segment(t * nq_, nq_);
    }
    invalidate_cache();
  }

  // Norm of the q vector for all trajectories.
  T norm() const {
    using std::sqrt;
    T squared_norm = 0.0;
    for (int t = 0; t <= num_steps_; ++t) {
      squared_norm += q_[t].squaredNorm();
    }
    return sqrt(squared_norm);
  }

  /**
   * Getter for the cache, containing other values computed from q, such as
   * generalized velocities, forces, and various dynamics derivatives.
   *
   * @return const TrajectoryOptimizerCache& cache
   */
  const TrajectoryOptimizerCache<T>& cache() const { return cache_; }

  /**
   * Get a mutable copy of the cache, containing other values computed from q,
   * such as generalized velocities, forces, and various dynamics derivatives.
   *
   * @return TrajectoryOptimizerCache&
   */
  TrajectoryOptimizerCache<T>& mutable_cache() const { return cache_; }

  /**
   * Scratch space for intermediate computations, to avoid expensive
   * allocations.
   */
  mutable TrajectoryOptimizerWorkspace<T> workspace;

  /**
   * Scratch space for intermediate computations, one per timestep to enable
   * parallelization.
   */
  mutable std::vector<TrajectoryOptimizerWorkspace<T>> per_timestep_workspace;

  private:
  // Number of timesteps in the optimization problem
  const int num_steps_;

  // Number of multibody positions for this system
  const int nq_;

  // Sequence of generalized velocities at each timestep,
  // [q(0), q(1), ..., q(num_steps)]
  // TODO(vincekurtz): consider storing as a single ocs2::vector_s_t<T> for better memory
  // layout.
  std::vector<ocs2::vector_s_t<T>> q_;

  // Storage for all other quantities that are computed from q, and are useful
  // for our calculations
  mutable TrajectoryOptimizerCache<T> cache_;

  // Set all the cache invalidation flags to false
  void invalidate_cache() {
    cache_.trajectory_data.up_to_date = false;
    cache_.inverse_dynamics_cache.up_to_date = false;
    cache_.foot_slip_and_clearance_cost_cache.up_to_date = false;
    cache_.derivatives_data.up_to_date = false;
    cache_.cost_up_to_date = false;
    cache_.gradient_up_to_date = false;
    cache_.hessian_up_to_date = false;
    if (cache_.context_cache)
      cache_.context_cache->up_to_date = false;
    cache_.n_plus_up_to_date = false;
    cache_.scaled_hessian_up_to_date = false;
    cache_.scaled_gradient_up_to_date = false;
    cache_.scale_factors_up_to_date = false;
    cache_.constraint_violation_up_to_date = false;
    cache_.constraint_jacobian_up_to_date = false;
    cache_.lagrange_multipliers_up_to_date = false;
    cache_.merit_up_to_date = false;
    cache_.merit_gradient_up_to_date = false;
  }
};

template struct ::idto::optimizer::Context<double>;
template struct ::idto::optimizer::TrajectoryOptimizerCache<double>;
template class ::idto::optimizer::TrajectoryOptimizerState<double>;

}  // namespace optimizer
}  // namespace idto
