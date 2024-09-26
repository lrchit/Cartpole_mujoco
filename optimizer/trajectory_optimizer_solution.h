#pragma once

#include <fstream>
#include <string>
#include <vector>

#include <Types.h>

namespace idto {
namespace optimizer {

// Status indicator for the overall success of our trajectory optimization.
enum SolverFlag {
  kSuccess,
  kLinesearchMaxIters,
  kFactorizationFailed,
  kMaxIterationsReached,
};

// Enum to indicate convergence reasons. Several convergence criteria can be
// satisfied at a time.
enum ConvergenceReason : int {
  // Bitmask-able values so they can be OR'd together.
  kNoConvergenceCriteriaSatisfied = 0b000,
  kCostReductionCriterionSatisfied = 0b001,
  kGradientCriterionSatisfied = 0b010,
  kSateCriterionSatisfied = 0b100
};

std::string DecodeConvergenceReasons(ConvergenceReason reason);

/**
 * A container for the optimal solution, including generalized positions,
 * velocities, and forces.
 *
 * TODO(vincekurtz): consider holding control inputs u rather than generalized
 * forces tau (tau = B*u)
 */
template <typename T>
struct TrajectoryOptimizerSolution {
  // Optimal sequence of generalized positions at each timestep
  std::vector<ocs2::vector_s_t<T>> q;

  // Optimal sequence of generalized velocities at each timestep
  std::vector<ocs2::vector_s_t<T>> v;

  // Optimal sequence of generalized forces at each timestep
  std::vector<ocs2::vector_s_t<T>> tau;
};

/**
 * A container for data about the solve process
 */
template <typename T>
struct TrajectoryOptimizerStats {
  ConvergenceReason convergence_reason{ConvergenceReason::kNoConvergenceCriteriaSatisfied};

  // Total solve time
  double solve_time;

  // Time for each iteration
  std::vector<double> iteration_times;

  // Cost at each iteration
  std::vector<T> iteration_costs;

  // Number of linesearch iterations, or number of times the trust-region was
  // modified, for each outer iteration
  std::vector<int> linesearch_iterations;

  // Linsearch parameter alpha for each iteration
  std::vector<double> linesearch_alphas;

  // Trust region radius Δ for each iteration
  std::vector<T> trust_region_radii;

  // Norm of the gradient at each iteration
  std::vector<T> gradient_norms;

  // Norm of the state q
  std::vector<T> q_norms;

  // Norm of the search direction at each iteration
  std::vector<T> dq_norms;

  // Norm of the unconstrained search direction at each iteration
  std::vector<T> dqH_norms;

  // Trust ratio (L(q) - L(q+dq)) / (m(q) - m(q+dq)), where m is a
  // quadratic model of the cost
  std::vector<T> trust_ratios;

  // Cost gradient along dq.
  std::vector<T> dL_dqs;

  // Norm of the equality constraint violations h(q) = 0
  std::vector<T> h_norms;

  // Merit function (measure of cost and constraint satisfaction)
  std::vector<T> merits;

  /**
   * Add the data from one iteration to the stored lists
   *
   * @param iter_time compute time for this iteration
   * @param iter_cost cost at this iteration
   * @param linesearch_iters number of linesearch or trust-region iterations
   * @param alpha linesearch parameter
   * @param delta trust region raidus
   * @param dq_norm norm of the linesearch direction Δq
   * @param dqH_norm norm of the unconstrained newton step
   * @param trust_ratio trust ratio: actual cost reduction / expected cost
   * reduction
   * @param grad_norm norm of the gradient
   * @param dL_dq cost gradient along the step Δq
   * @param h_norm norm of the equality constraint violations
   * @param merit merit function at this iteration
   */
  void push_data(double iter_time,
      T iter_cost,
      int linesearch_iters,
      double alpha,
      double delta,
      T q_norm,
      T dq_norm,
      T dqH_norm,
      T trust_ratio,
      T grad_norm,
      T dL_dq,
      T h_norm,
      T merit) {
    iteration_times.push_back(iter_time);
    iteration_costs.push_back(iter_cost);
    linesearch_iterations.push_back(linesearch_iters);
    linesearch_alphas.push_back(alpha);
    trust_region_radii.push_back(delta);
    q_norms.push_back(q_norm);
    dq_norms.push_back(dq_norm);
    dqH_norms.push_back(dqH_norm);
    trust_ratios.push_back(trust_ratio);
    gradient_norms.push_back(grad_norm);
    dL_dqs.push_back(dL_dq);
    h_norms.push_back(h_norm);
    merits.push_back(merit);
  }

  /**
   * Check if the stored lists of data are all empty
   */
  bool is_empty() const {
    return ((iteration_times.size() == 0) && (iteration_costs.size() == 0) && (linesearch_iterations.size() == 0) && (linesearch_alphas.size() == 0) &&
            (trust_region_radii.size() == 0) && (q_norms.size() == 0) && (dq_norms.size() == 0) && (dqH_norms.size() == 0) && (trust_ratios.size() == 0) &&
            (gradient_norms.size() == 0) && (dL_dqs.size() == 0) && (h_norms.size() == 0) && (merits.size() == 0));
  }
};

}  // namespace optimizer
}  // namespace idto
