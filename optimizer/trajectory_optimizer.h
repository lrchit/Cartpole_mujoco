#pragma once

#include <pinocchio/fwd.hpp>  // always include it before any other header

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "inverse_dynamics_partials.h"
#include "penta_diagonal_matrix.h"
#include "problem_definition.h"
#include "solver_parameters.h"
#include "trajectory_optimizer_solution.h"
#include "trajectory_optimizer_state.h"
#include "trajectory_optimizer_workspace.h"
#include "velocity_partials.h"
#include "warm_start.h"

#include <Types.h>

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

namespace idto {
namespace optimizer {

using internal::PentaDiagonalMatrix;

template <typename T>
class TrajectoryOptimizer {
  public:
  /**
   * Construct a new Trajectory Optimizer object.
   *
   * @param plant A model of the system that we're trying to find an optimal
   *              trajectory for.
   * @param prob Problem definition, including cost, initial and target states,
   *             etc.
   * @param params solver parameters, including max iterations, linesearch
   *               method, etc.
   */
  TrajectoryOptimizer(const pinocchio::ModelTpl<T>& model,
      const pinocchio::DataTpl<T>& data,
      const ProblemDefinition& prob,
      const double time_step,
      std::vector<pinocchio::FrameIndex> footId,
      const SolverParameters& params = SolverParameters{});

  /**
   * Convienience function to get the timestep of this optimization problem.
   *
   * @return double dt, the time step for this optimization problem
   */
  double time_step() const { return time_step_; }

  /**
   * Convienience function to get the time horizon (T) of this optimization
   * problem.
   *
   * @return int the number of time steps in the optimal trajectory.
   */
  int num_steps() const { return prob_.num_steps; }

  /**
   * Return indices of the unactuated degrees of freedom in the model.
   *
   * @return const std::vector<int>& indices for the unactuated DoFs
   */
  const std::vector<int>& unactuated_dofs() const { return unactuated_dofs_; }

  /**
   * Convienience function to get the number of equality constraints (i.e.,
   * torques on unactuated DoFs at each time step)
   *
   * @return int the number of equality constraints
   */
  int num_equality_constraints() const { return unactuated_dofs().size() * num_steps(); }

  /**
   * Convienience function to get a const reference to the solver parameters.
  */
  const SolverParameters& params() const { return params_; }

  /**
   * Convienience function to get a const reference to the problem definition.
  */
  const ProblemDefinition& prob() const { return prob_; }

  /**
   * Create a state object which contains the decision variables (generalized
   * positions at each timestep), along with a cache of other things that are
   * computed from positions, such as velocities, accelerations, forces, and
   * various derivatives.
   *
   * @return TrajectoryOptimizerState
   */
  TrajectoryOptimizerState<T> CreateState() const {
    // INSTRUMENT_FUNCTION("Creates state object with caching.");
    return TrajectoryOptimizerState<T>(num_steps(), model_, data_, num_equality_constraints());
  }

  /**
   * Compute the gradient of the unconstrained cost L(q).
   *
   * @param state optimizer state, including q, v, tau, gradients, etc.
   * @param g a single ocs2::vector_t containing the partials of L w.r.t. each
   *          decision variable (q_t[i]).
   */
  void CalcGradient(const TrajectoryOptimizerState<T>& state, ocs2::vector_s_t<T>* g) const;

  /**
   * Compute the Hessian of the unconstrained cost L(q) as a sparse
   * penta-diagonal matrix.
   *
   * @param state optimizer state, including q, v, tau, gradients, etc.
   * @param H a PentaDiagonalMatrix containing the second-order derivatives of
   *          the total cost L(q). This matrix is composed of (num_steps+1 x
   *          num_steps+1) blocks of size (nq x nq) each.
   */
  void CalcHessian(const TrajectoryOptimizerState<T>& state, PentaDiagonalMatrix<T>* H) const;

  /**
   * Solve the optimization from the given initial guess, which may or may not
   * be dynamically feasible.
   *
   * @param q_guess a sequence of generalized positions corresponding to the
   * initial guess
   * @param solution a container for the optimal solution, including velocities
   * and torques
   * @param stats a container for other timing and iteration-specific
   * data regarding the solve process.
   * @return SolverFlag
   */
  SolverFlag Solve(const std::vector<ocs2::vector_s_t<T>>& q_guess,
      TrajectoryOptimizerSolution<T>* solution,
      TrajectoryOptimizerStats<T>* stats,
      ConvergenceReason* reason = nullptr) const;

  /**
   * Solve the optimization with a full warm-start, including both an initial
   * guess and optimizer parameters like the trust region radius.
   *
   * @note this is only used for the trust-region method
   *
   * @param warm_start Container for the initial guess, optimizer state, etc.
   * @param solution Optimal solution, including velocities and torques
   * @param stats timing and other iteration-specific statistics
   * @param reason convergence reason, if applicable
   * @return SolverFlag
   */
  SolverFlag SolveFromWarmStart(WarmStart* warm_start,
      TrajectoryOptimizerSolution<T>* solution,
      TrajectoryOptimizerStats<T>* stats,
      ConvergenceReason* reason = nullptr) const;

  // The following evaluator functions get data from the state's cache, and
  // update it if necessary.

  /**
   * Evaluate generalized velocities
   *
   *    v_t = (q_t - q_{t-1}) / dt
   *
   * at each timestep t, t = [0, ..., num_steps()],
   *
   * where v_0 is fixed by the initial condition.
   *
   * @param state optimizer state
   * @return const std::vector<ocs2::vector_s_t<T>>& v_t
   */
  const std::vector<ocs2::vector_s_t<T>>& EvalV(const TrajectoryOptimizerState<T>& state) const;

  /**
   * Evaluate generalized accelerations
   *
   *    a_t = (v_{t+1} - v_t) / dt
   *
   * at each timestep t, t = [0, ..., num_steps()-1].
   *
   * @param state optimizer state
   * @return const std::vector<ocs2::vector_s_t<T>>& a_t
   */
  const std::vector<ocs2::vector_s_t<T>>& EvalA(const TrajectoryOptimizerState<T>& state) const;

  /**
   * Evaluate generalized forces
   *
   *    τ_t = ID(q_{t+1}, v_{t+1}, a_t) - J(q_{t+1})'γ(q_{t+1},v_{t+1})
   *
   * at each timestep t, t = [0, ..., num_steps()-1].
   *
   * @param state optimizer state
   * @return const std::vector<ocs2::vector_s_t<T>>& τ_t
   */
  const std::vector<ocs2::vector_s_t<T>>& EvalTau(const TrajectoryOptimizerState<T>& state) const;

  /**
   * Evaluate partial derivatives of velocites with respect to positions at each
   * time step.
   *
   * @param state optimizer state
   * @return const VelocityPartials<T>& container for ∂v/∂q
   */
  const VelocityPartials<T>& EvalVelocityPartials(const TrajectoryOptimizerState<T>& state) const;

  /**
   * Evaluate the mapping from qdot to v, v = N+(q)*qdot, at each time step.
   *
   * @param state optimizer state
   * @return const std::vector<ocs2::matrix_s_t<T>>& N+(q_t) for each time step t.
   */
  const std::vector<ocs2::matrix_s_t<T>>& EvalNplus(const TrajectoryOptimizerState<T>& state) const;

  /**
   * Evaluate partial derivatives of generalized forces with respect to
   * positions at each time step.
   *
   * @param state optimizer state
   * @return const InverseDynamicsPartials<T>& container for ∂τ/∂q
   */
  const InverseDynamicsPartials<T>& EvalInverseDynamicsPartials(const TrajectoryOptimizerState<T>& state) const;

  /**
   * Evaluate the total (unconstrained) cost of the optimization problem,
   *
   *     L(q) = x_err(T)'*Qf*x_err(T)
   *                + dt*sum_{t=0}^{T-1} x_err(t)'*Q*x_err(t) + u(t)'*R*u(t),
   *
   * where:
   *      x_err(t) = x(t) - x_nom is the state error,
   *      T = num_steps() is the time horizon of the optimization problem,
   *      x(t) = [q(t); v(t)] is the system state at time t,
   *      u(t) are control inputs, and we assume (for now) that u(t) = tau(t),
   *      Q{f} = diag([Qq{f}, Qv{f}]) are a block diagonal PSD state-error
   *       weighting matrices,
   *      R is a PSD control weighting matrix.
   *
   * A cached version of this cost is stored in the state. If the cache is up to
   * date, simply return that cost.
   *
   * @param state optimizer state
   * @return const double, total cost
   */
  const T EvalCost(const TrajectoryOptimizerState<T>& state) const;

  /**
   * Evaluate the Hessian of the unconstrained cost L(q) as a sparse
   * penta-diagonal matrix.
   *
   * @param state optimizer state, including q, v, tau, gradients, etc.
   * @return const PentaDiagonalMatrix<T>& the second-order derivatives of
   *          the total cost L(q). This matrix is composed of (num_steps+1 x
   *          num_steps+1) blocks of size (nq x nq) each.
   */
  const PentaDiagonalMatrix<T>& EvalHessian(const TrajectoryOptimizerState<T>& state) const;

  /**
   * Evaluate a scaled version of the Hessian, given by
   *
   *     H̃ = DHD,
   *
   * where H is the original Hessian and D is a diagonal scaling matrix.
   *
   * @note if params_.scaling = false, this returns the ordinary Hessian H.
   *
   * @param state optimizer state, including q, v, tau, gradients, etc.
   * @return const PentaDiagonalMatrix<T>& the scaled Hessian
   */
  const PentaDiagonalMatrix<T>& EvalScaledHessian(const TrajectoryOptimizerState<T>& state) const;

  /**
   * Evaluate a scaled version of the gradient, given by
   *
   *     g̃ = Dg,
   *
   * where g is the original gradient and D is a diagonal scaling matrix.
   *
   * @note if params_.scaling = false, this returns the ordinary gradient g.
   *
   * @param state optimizer state, including q, v, tau, gradients, etc.
   * @return const ocs2::vector_s_t<T>& the scaled gradient
   */
  const ocs2::vector_s_t<T>& EvalScaledGradient(const TrajectoryOptimizerState<T>& state) const;

  /**
   * Evaluate a vector of scaling factors based on the diagonal of the Hessian.
   *
   * @param state the optimizer state
   * @return const ocs2::vector_s_t<T>& the scaling vector D
   */
  const ocs2::vector_s_t<T>& EvalScaleFactors(const TrajectoryOptimizerState<T>& state) const;

  /**
   * Evaluate the vector of violations of equality constrants h(q) = 0.
   *
   * Currently, these equality constraints consist of torques on unactuated
   * degrees of freedom.
   *
   * @param state the optimizer state
   * @return const ocs2::vector_s_t<T>& violations h(q)
   */
  const ocs2::vector_s_t<T>& EvalEqualityConstraintViolations(const TrajectoryOptimizerState<T>& state) const;

  /**
   * Evaluate the Jacobian J = ∂h(q)/∂q of the equality constraints h(q) = 0.
   *
   * @note if scaling is enabled this returns a scaled version J*D, where D is a
   * diagonal scaling matrix.
   *
   * @param state the optimizer state
   * @return const ocs2::matrix_s_t<T>& the Jacobian of equality constraints
   */
  const ocs2::matrix_s_t<T>& EvalEqualityConstraintJacobian(const TrajectoryOptimizerState<T>& state) const;

  /**
   * Evaluate the lagrange multipliers λ for the equality constraints h(q) = 0.
   *
   * These are given by
   *
   *    λ = (J H⁻¹ Jᵀ)⁻¹ (h − J H⁻¹ g),
   *
   * or equivalently, the solution of the KKT conditions
   *
   *    [H Jᵀ][Δq] = [-g]
   *    [J 0 ][ λ]   [-h]
   *
   * where H is the unconstrained Hessian, J is the equality constraint
   * jacobian, and g is the unconstrained gradient.
   *
   * @param state the optimizer state
   * @return const ocs2::vector_s_t<T>& the lagrange multipliers
   */
  const ocs2::vector_s_t<T>& EvalLagrangeMultipliers(const TrajectoryOptimizerState<T>& state) const;

  /**
   * Evaluate the (augmented-lagrangian-inspired) merit function
   *
   *    ϕ(q) = L(q) + h(q)ᵀλ
   *
   * for constrained optimization. If equality constraints are turned off, this
   * simply returns the unconstrained cost L(q).
   *
   * @param state the optimizer state
   * @return const T the merit function ϕ(q)
   */
  const T EvalMeritFunction(const TrajectoryOptimizerState<T>& state) const;

  /**
   * Evaluate the gradient of the merit function ϕ(q):
   *
   *    g̃ = g + Jᵀλ,
   *
   * under the assumption that the lagrange multipliers λ are constant. If
   * equality constraints are turned off, this simply returns the regular
   * gradient g.
   *
   * @note if scaling is enabled this uses scaled versions of g and J.
   *
   * @param state the optimizer state
   * @return const ocs2::vector_s_t<T>& the gradient of the merit function g̃
   */
  const ocs2::vector_s_t<T>& EvalMeritFunctionGradient(const TrajectoryOptimizerState<T>& state) const;

  /**
   * Evaluate the gradient of the unconstrained cost L(q).
   *
   * @param state optimizer state, including q, v, tau, gradients, etc.
   * @return const ocs2::vector_s_t<T>& a single vector containing the partials of L
   * w.r.t. each decision variable (q_t[i]).
   */
  const ocs2::vector_s_t<T>& EvalGradient(const TrajectoryOptimizerState<T>& state) const;

  /**
   * Evaluate a system context for the plant at the given time step.
   *
   * @param state optimizer state
   * @param t time step
   * @return const Context<T>& context for the plant at time t
   */
  const Context<T>& EvalPlantContext(const TrajectoryOptimizerState<T>& state, int t) const;

  /**
   * Overwrite the initial conditions x0 = [q0, v0] stored in the solver
   * parameters. This is particularly useful when re-solving the optimization
   * problem for MPC.
   *
   * @param q_init Initial generalized positions
   * @param v_init Initial generalized velocities
   */
  void ResetInitialConditions(const ocs2::vector_t& q_init, const ocs2::vector_t& v_init) {
    assert(q_init.size() == model_.nq);
    assert(v_init.size() == model_.nv);
    assert(params_.q_nom_relative_to_q_init.size() == model_.nq);
    prob_.q_init = q_init;
    prob_.v_init = v_init;
  }

  /**
   * Overwrite the nominal trajectory q_nom stored in the solver parameters.
   * This is particularly useful when re-solving the optimization problem for
   * MPC.
   *
   * @param q_nom Nominal trajectory for the generalized positions
   */
  void UpdateNominalTrajectory(const std::vector<ocs2::vector_t>& q_nom, const std::vector<ocs2::vector_t>& v_nom) {
    assert(static_cast<int>(q_nom.size()) == num_steps() + 1);
    assert(static_cast<int>(q_nom[0].size()) == model_.nq);
    prob_.q_nom = q_nom;
    prob_.v_nom = v_nom;
  }

  private:
  /**
   * Solve the optimization problem from the given initial guess using a
   * linesearch strategy.
   *
   * @param q_guess a sequence of generalized positions corresponding to the
   * initial guess
   * @param solution a container for the optimal solution, including velocities
   * and torques
   * @param stats a container for other timing and iteration-specific
   * data regarding the solve process.
   * @return SolverFlag
   */
  SolverFlag SolveWithLinesearch(const std::vector<ocs2::vector_s_t<T>>& q_guess,
      TrajectoryOptimizerSolution<T>* solution,
      TrajectoryOptimizerStats<T>* stats) const;

  /**
   * Solve the optimization problem from the given initial guess using a trust
   * region strategy.
   *
   * @param q_guess a sequence of generalized positions corresponding to the
   * initial guess
   * @param solution a container for the optimal solution, including velocities
   * and torques
   * @param stats a container for other timing and iteration-specific
   * data regarding the solve process.
   * @return SolverFlag
   */
  SolverFlag SolveWithTrustRegion(const std::vector<ocs2::vector_s_t<T>>& q_guess,
      TrajectoryOptimizerSolution<T>* solution,
      TrajectoryOptimizerStats<T>* stats,
      ConvergenceReason* reason) const;

  /**
   * Return a mutable system context for the plant at the given time step.
   *
   * @param state optimizer state
   * @param t time step
   * @return Context<T>& context for the plant at time t
   */
  Context<T>& GetMutablePlantContext(const TrajectoryOptimizerState<T>& state, int t) const;

  /**
   * Update the system context for the plant at each time step to store q and v
   * from the state.
   *
   * @param state optimizer state containing q and v
   * @param cache context cache containing a plant context for each timestep
   */
  void CalcContextCache(const TrajectoryOptimizerState<T>& state, typename TrajectoryOptimizerCache<T>::ContextCache* cache) const;

  /**
   * Compute all of the "trajectory data" (velocities v, accelerations a,
   * torques tau) in the state's cache to correspond to the state's generalized
   * positions q.
   *
   * @param state optimizer state to update.
   */
  void CalcCacheTrajectoryData(const TrajectoryOptimizerState<T>& state) const;

  void CalcInverseDynamicsCache(const TrajectoryOptimizerState<T>& state, typename TrajectoryOptimizerCache<T>::InverseDynamicsCache* cache) const;

  /**
   * Compute all of the "derivatives data" (dv/dq, dtau/dq) stored in the
   * state's cache to correspond to the state's generalized positions q.
   *
   * @param state optimizer state to update.
   */
  void CalcCacheDerivativesData(const TrajectoryOptimizerState<T>& state) const;

  /**
   * Return the total (unconstrained) cost of the optimization problem,
   *
   *     L(q) = x_err(T)'*Qf*x_err(T)
   *                + dt*sum_{t=0}^{T-1} x_err(t)'*Q*x_err(t) + u(t)'*R*u(t),
   *
   * where:
   *      x_err(t) = x(t) - x_nom is the state error,
   *      T = num_steps() is the time horizon of the optimization problem,
   *      x(t) = [q(t); v(t)] is the system state at time t,
   *      u(t) are control inputs, and we assume (for now) that u(t) = tau(t),
   *      Q{f} = diag([Qq{f}, Qv{f}]) are a block diagonal PSD state-error
   *       weighting matrices,
   *      R is a PSD control weighting matrix.
   *
   * A cached version of this cost is stored in the state. If the cache is up to
   * date, simply return that cost.
   *
   * @param state optimizer state
   * @return double, total cost
   */
  T CalcCost(const TrajectoryOptimizerState<T>& state) const;

  /**
   * Compute the total cost of the unconstrained problem.
   *
   * @param q sequence of generalized positions
   * @param v sequence of generalized velocities (consistent with q)
   * @param tau sequence of generalized forces (consistent with q and v)
   * @param workspace scratch space for intermediate computations
   * @return double, total cost
   */
  T CalcCost(const std::vector<ocs2::vector_s_t<T>>& q,
      const std::vector<ocs2::vector_s_t<T>>& v,
      const std::vector<ocs2::vector_s_t<T>>& tau,
      TrajectoryOptimizerWorkspace<T>* workspace) const;

  /**
   * Compute a sequence of generalized velocities v from a sequence of
   * generalized positions, where
   *
   *     v_t = N+(q_t) * (q_t - q_{t-1}) / dt            (1)
   *
   * v and q are each vectors of length num_steps+1,
   *
   *     v = [v(0), v(1), v(2), ..., v(num_steps)],
   *     q = [q(0), q(1), q(2), ..., q(num_steps)].
   *
   * Note that v0 = v_init is defined by the initial state of the optimization
   * problem, rather than Equation (1) above.
   *
   * @param q sequence of generalized positions
   * @param Nplus the mapping from qdot to v, N+(q_t).
   * @param v sequence of generalized velocities
   */
  void CalcVelocities(const std::vector<ocs2::vector_s_t<T>>& q, const std::vector<ocs2::matrix_s_t<T>>& Nplus, std::vector<ocs2::vector_s_t<T>>* v) const;

  /**
   * Compute a sequence of generalized accelerations a from a sequence of
   * generalized velocities,
   *
   *    a_t = (v_{t+1} - v_{t})/dt,
   *
   * where v is of length (num_steps+1) and a is of length num_steps:
   *
   *     v = [v(0), v(1), v(2), ..., v(num_steps)],
   *     a = [a(0), a(1), a(2), ..., a(num_steps-1)].
   *
   * @param v sequence of generalized velocities
   * @param a sequence of generalized accelerations
   */
  void CalcAccelerations(const std::vector<ocs2::vector_s_t<T>>& v, std::vector<ocs2::vector_s_t<T>>* a) const;

  /**
   * Compute a sequence of generalized forces t from sequences of generalized
   * accelerations, velocities, and positions, where generalized forces are
   * defined by the inverse dynamics,
   *
   *    tau_t = M*(v_{t+1}-v_t})/dt + D*v_{t+1} - k(q_t,v_t)
   *                               - (1/dt) *J'*gamma(v_{t+1},q_t).
   *
   * Note that q and v have length num_steps+1,
   *
   *  q = [q(0), q(1), ..., q(num_steps)],
   *  v = [v(0), v(1), ..., v(num_steps)],
   *
   * while a and tau have length num_steps,
   *
   *  a = [a(0), a(1), ..., a(num_steps-1)],
   *  tau = [tau(0), tau(1), ..., tau(num_steps-1)],
   *
   * i.e., tau(t) takes us us from t to t+1.
   *
   * @param state state variable storing a context for each timestep. This
   * context in turn stores q(t) and v(t) for each timestep.
   * @param a sequence of generalized accelerations
   * @param tau sequence of generalized forces
   */
  void CalcInverseDynamics(const TrajectoryOptimizerState<T>& state, const std::vector<ocs2::vector_s_t<T>>& a, std::vector<ocs2::vector_s_t<T>>* tau) const;

  /**
   * Helper function for computing the inverse dynamics
   *
   *  tau = ID(a, v, q, f_ext)
   *
   * at a single timestep.
   *
   * @param context system context storing q and v
   * @param a generalized acceleration
   * @param workspace scratch space for intermediate computations
   * @param tau generalized forces
   */
  void CalcInverseDynamicsSingleTimeStep(const Context<T>& context,
      const ocs2::vector_s_t<T>& a,
      TrajectoryOptimizerWorkspace<T>* workspace,
      ocs2::vector_s_t<T>* tau) const;

  /**
   * Calculate the force contribution from contacts for each body, and add them
   * into the given MultibodyForces object.
   *
   * @param context system context storing q and v
   * @param forces total forces applied to the plant, which we will add into.
   */
  void CalcContactForceContribution(const Context<T>& context, pinocchio::container::aligned_vector<pinocchio::ForceTpl<T>>* fext) const;

  /**
   * Calculate the contact force through soft contact model
   *
   * @param vn normal vel in world frame
   * @param vt tangent vel in world frame
   * @param nhat unit normal vector
   * @return contact forces.
   */
  const ocs2::vector3_s_t<T> computeContactForceThroughSoftModel(const T distance, const ocs2::vector3_s_t<T> v, const ocs2::vector3_s_t<T> nhat) const;

  /**
   * Calculate the contact force through spring-damper contact model
   *
   * @param vn normal vel in world frame
   * @param vt tangent vel in world frame
   * @param nhat unit normal vector
   * @return contact forces.
   */
  const ocs2::vector3_s_t<T> computeContactForceThroughSpringDamperModel(const T distance, const ocs2::vector3_s_t<T> v, const ocs2::vector3_s_t<T> nhat) const;

  /**
   * Compute the mapping from qdot to v, v = N+(q)*qdot, at each time step.
   *
   * @param state optimizer state
   * @param N_plus vector containing N+(q_t) for each time step t.
   */
  void CalcNplus(const TrajectoryOptimizerState<T>& state, std::vector<ocs2::matrix_s_t<T>>* N_plus) const;

  /**
   * Compute partial derivatives of the generalized velocities
   *
   *    v_t = N+(q_t) * (q_t - q_{t-1}) / dt
   *
   * and store them in the given VelocityPartials struct.
   *
   * @param q sequence of generalized positions
   * @param v_partials struct for holding dv/dq
   */
  void CalcVelocityPartials(const TrajectoryOptimizerState<T>& state, VelocityPartials<T>* v_partials) const;

  /**
   * Compute partial derivatives of the inverse dynamics
   *
   *    tau_t = ID(q_{t-1}, q_t, q_{t+1})
   *
   * and store them in the given InverseDynamicsPartials struct.
   *
   * @param state state variable containing q for each timestep
   * @param id_partials struct for holding dtau/dq
   */
  void CalcInverseDynamicsPartials(const TrajectoryOptimizerState<T>& state, InverseDynamicsPartials<T>* id_partials) const;

  /**
   * Compute the linesearch parameter alpha given a linesearch direction
   * dq. In other words, approximately solve the optimization problem
   *
   *      min_{alpha} L(q + alpha*dq).
   *
   * @param state the optimizer state containing q and everything that we
   *              compute from q
   * @param dq search direction, stacked as one large vector
   * @param scratch_state scratch state variable used for computing L(q +
   *                      alpha*dq)
   * @return double, the linesearch parameter alpha
   * @return int, the number of linesearch iterations
   */
  std::tuple<double, int> Linesearch(const TrajectoryOptimizerState<T>& state, const ocs2::vector_s_t<T>& dq, TrajectoryOptimizerState<T>* scratch_state) const;

  /**
   * Simple backtracking linesearch strategy to find alpha that satisfies
   *
   *    L(q + alpha*dq) < L(q) + c*g'*dq
   *
   * and is (approximately) a local minimizer of L(q + alpha*dq).
   */
  std::tuple<double, int> BacktrackingLinesearch(const TrajectoryOptimizerState<T>& state,
      const ocs2::vector_s_t<T>& dq,
      TrajectoryOptimizerState<T>* scratch_state) const;

  /**
   * Simple backtracking linesearch strategy to find alpha that satisfies
   *
   *    L(q + alpha*dq) < L(q) + c*g'*dq
   */
  std::tuple<double, int> ArmijoLinesearch(const TrajectoryOptimizerState<T>& state,
      const ocs2::vector_s_t<T>& dq,
      TrajectoryOptimizerState<T>* scratch_state) const;

  /**
   * Compute the trust ratio
   *
   *           L(q) - L(q + dq)
   *    rho =  ----------------
   *             m(0) - m(dq)
   *
   * which compares the actual reduction in cost to the reduction in cost
   * predicted by the quadratic model
   *
   *    m(dq) = L + g'*dq + 1/2 dq'*H*dq
   *
   * @param state optimizer state containing q and everything computed from q
   * @param dq change in q, stacked in one large vector
   * @param scratch_state scratch state variable used to compute L(q+dq)
   * @return T, the trust region ratio
   */
  T CalcTrustRatio(const TrajectoryOptimizerState<T>& state, const ocs2::vector_s_t<T>& dq, TrajectoryOptimizerState<T>* scratch_state) const;

  /**
   * Compute the dogleg step δq, which approximates the solution to the
   * trust-region sub-problem
   *
   *   min_{δq} L(q) + g(q)'*δq + 1/2 δq'*H(q)*δq
   *   s.t.     ‖ δq ‖ <= Δ
   *
   * @param state the optimizer state, containing q and the ability to compute
   * g(q) and H(q)
   * @param Delta the trust region size
   * @param dq the dogleg step (change in decision variables)
   * @param dqH the Newton step
   * @return true if the step intersects the trust region
   * @return false if the step is in the interior of the trust region
   */
  bool CalcDoglegPoint(const TrajectoryOptimizerState<T>& state, const double Delta, ocs2::vector_s_t<T>* dq, ocs2::vector_s_t<T>* dqH) const;

  /**
   * Solve the scalar quadratic equation
   *
   *    a x² + b x + c = 0
   *
   * for the positive root. This problem arises from finding the intersection
   * between the trust region and the second leg of the dogleg path. Provided we
   * have properly checked that the trust region does intersect this second
   * leg, this quadratic equation has some special properties:
   *
   *     - a is strictly positive
   *     - there is exactly one positive root
   *     - this positive root is in (0,1)
   *
   * @param a the first coefficient
   * @param b the second coefficient
   * @param c the third coefficient
   * @return T the positive root
   */
  T SolveDoglegQuadratic(const T& a, const T& b, const T& c) const;

  /**
   * Helper to solve the system H⋅x = b with a solver specified in
   * SolverParameters::LinearSolverType.
   *
   * @param H A block penta-diagonal matrix H
   * @param b The vector b. Overwritten with x on output.
   */
  void SolveLinearSystemInPlace(const PentaDiagonalMatrix<T>& H, ocs2::vector_s_t<T>* b) const;

  ConvergenceReason VerifyConvergenceCriteria(const TrajectoryOptimizerState<T>& state, const T& previous_cost, const ocs2::vector_s_t<T>& dq) const;

  /**
   * Compute the scaled version of the Hessian, H̃ = DHD.
   *
   * @param state the optimizer state
   * @param Htilde the scaled Hessian H̃
   */
  void CalcScaledHessian(const TrajectoryOptimizerState<T>& state, PentaDiagonalMatrix<T>* Htilde) const;

  /**
   * Compute the scaled version of the gradient, g̃ = Dg.
   *
   * @param state the optimizer state
   * @param gtilde the scaled gradient g̃
   */
  void CalcScaledGradient(const TrajectoryOptimizerState<T>& state, ocs2::vector_s_t<T>* gtilde) const;

  /**
   * Compute the vector of scaling factors D based on the diagonal of the
   * Hessian.
   *
   * @param state the optimizer state
   * @param D the vector of scale factors D
   */
  void CalcScaleFactors(const TrajectoryOptimizerState<T>& state, ocs2::vector_s_t<T>* D) const;

  /**
   * Compute a vector of equality constrant h(q) = 0 violations.
   *
   * Currently, these equality constraints consist of torques on unactuated
   * degrees of freedom.
   *
   * @param state the optimizer state
   * @param violations vector of constraint violiations h
   */
  void CalcEqualityConstraintViolations(const TrajectoryOptimizerState<T>& state, ocs2::vector_s_t<T>* violations) const;

  /**
   * Compute the Jacobian J = ∂h(q)/∂q of the equality constraints h(q) = 0.
   *
   * @param state the optimizer state
   * @param J the constraint jacobian ∂h(q)/∂q
   */
  void CalcEqualityConstraintJacobian(const TrajectoryOptimizerState<T>& state, ocs2::matrix_s_t<T>* J) const;

  /**
   * Compute the lagrange multipliers λ for the equality constraints h(q) = 0.
   *
   * @param state the optimizer state
   * @param lambda the lagrange multipliers
   */
  void CalcLagrangeMultipliers(const TrajectoryOptimizerState<T>& state, ocs2::vector_s_t<T>* lambda) const;

  /**
   * Compute the (augmented-lagrangian-inspired) merit function ϕ(q) = L(q) +
   * h(q)ᵀλ.
   *
   * @param state the optimizer state
   * @param merit the merit function
   */
  void CalcMeritFunction(const TrajectoryOptimizerState<T>& state, T* merit) const;

  /**
   * Compute the gradient of the merit function g̃ = g + Jᵀλ.
   *
   * @param state the optimizer state
   * @param g_tilde the gradient of the merit function g̃
   */
  void CalcMeritFunctionGradient(const TrajectoryOptimizerState<T>& state, ocs2::vector_s_t<T>* g_tilde) const;

  pinocchio::ModelTpl<T> model_;
  pinocchio::DataTpl<T> data_;

  // Stores the problem definition, including cost, time horizon, initial state,
  // target state, etc.
  ProblemDefinition prob_;

  // discrete time step
  double time_step_;

  // Indices of unactuated degrees of freedom
  std::vector<int> unactuated_dofs_;

  // Various parameters
  const SolverParameters params_;

  // contact points
  std::vector<pinocchio::FrameIndex> footId_;
};

// Declare template specializations
template <>
SolverFlag TrajectoryOptimizer<double>::SolveWithLinesearch(const std::vector<ocs2::vector_t>&,
    TrajectoryOptimizerSolution<double>*,
    TrajectoryOptimizerStats<double>*) const;

template <>
SolverFlag TrajectoryOptimizer<double>::SolveWithTrustRegion(const std::vector<ocs2::vector_t>&,
    TrajectoryOptimizerSolution<double>*,
    TrajectoryOptimizerStats<double>*,
    ConvergenceReason*) const;

template <>
SolverFlag TrajectoryOptimizer<double>::SolveFromWarmStart(WarmStart*,
    TrajectoryOptimizerSolution<double>*,
    TrajectoryOptimizerStats<double>*,
    ConvergenceReason*) const;

}  // namespace optimizer
}  // namespace idto

#include <trajectory_optimizer_impl.h>
