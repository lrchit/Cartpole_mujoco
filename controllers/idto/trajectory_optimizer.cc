#include "trajectory_optimizer.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "penta_diagonal_solver.h"

#define PRINT_VAR(a) std::cout << #a ": " << a << std::endl;
#define PRINT_VARn(a) std::cout << #a ":\n" << a << std::endl;

namespace idto {
namespace optimizer {

using internal::PentaDiagonalFactorization;
using internal::PentaDiagonalFactorizationStatus;

template <typename T>
TrajectoryOptimizer<T>::TrajectoryOptimizer(const pinocchio::ModelTpl<T>& model,
    const pinocchio::DataTpl<T>& data,
    const ProblemDefinition& prob,
    const double time_step,
    std::vector<pinocchio::FrameIndex> footId,
    const SolverParameters& params)
    : model_(model), data_(data), prob_(prob), time_step_(time_step), footId_(footId), params_(params) {
  // Define unactuated degrees of freedom
  for (int i = 6; i < model_.nv; ++i) {
    unactuated_dofs_.push_back(i);
  }

  // Must have a target position and velocity specified for each time step
  assert(static_cast<int>(prob.q_nom.size()) == (num_steps() + 1));
  assert(static_cast<int>(prob.v_nom.size()) == (num_steps() + 1));

  // Target positions and velocities must be the right size
  for (int t = 0; t <= num_steps(); ++t) {
    assert(prob.q_nom[t].size() == model_.nq);
    assert(prob.v_nom[t].size() == model_.nv);
  }
}

template <typename T>
void TrajectoryOptimizer<T>::CalcGradient(const TrajectoryOptimizerState<T>& state, ocs2::vector_s_t<T>* g) const {
  //   INSTRUMENT_FUNCTION("Assembly of the gradient.");
  const double dt = time_step();
  const int nq = model_.nq;

  const std::vector<ocs2::vector_s_t<T>>& q = state.q();
  const std::vector<ocs2::vector_s_t<T>>& v = EvalV(state);
  const std::vector<ocs2::vector_s_t<T>>& tau = EvalTau(state);

  const VelocityPartials<T>& v_partials = EvalVelocityPartials(state);
  const InverseDynamicsPartials<T>& id_partials = EvalInverseDynamicsPartials(state);
  const std::vector<ocs2::matrix_s_t<T>>& dvt_dqt = v_partials.dvt_dqt;
  const std::vector<ocs2::matrix_s_t<T>>& dvt_dqm = v_partials.dvt_dqm;
  const std::vector<ocs2::matrix_s_t<T>>& dtau_dqp = id_partials.dtau_dqp;
  const std::vector<ocs2::matrix_s_t<T>>& dtau_dqt = id_partials.dtau_dqt;
  const std::vector<ocs2::matrix_s_t<T>>& dtau_dqm = id_partials.dtau_dqm;

  // Set first block of g (derivatives w.r.t. q_0) to zero, since q0 = q_init
  // are constant.
  g->topRows(model_.nq).setZero();

  for (int t = 1; t < num_steps(); ++t) {
    auto gt = g->segment(t * nq, nq);

    // Contribution from position cost
    gt = (q[t] - prob_.q_nom[t]).transpose() * 2 * prob_.Qq * dt;

    // Contribution from velocity cost
    gt += (v[t] - prob_.v_nom[t]).transpose() * 2 * prob_.Qv * dt * dvt_dqt[t];
    if (t == num_steps() - 1) {
      // The terminal cost needs to be handled differently
      gt += (v[t + 1] - prob_.v_nom[t + 1]).transpose() * 2 * prob_.Qf_v * dvt_dqm[t + 1];
    } else {
      gt += (v[t + 1] - prob_.v_nom[t + 1]).transpose() * 2 * prob_.Qv * dt * dvt_dqm[t + 1];
    }

    // Contribution from control cost
    gt += tau[t - 1].transpose() * 2 * (prob_.R + prob_.dSymmetricControlCost_dtaudtau) * dt * dtau_dqp[t - 1];
    gt += tau[t].transpose() * 2 * (prob_.R + prob_.dSymmetricControlCost_dtaudtau) * dt * dtau_dqt[t];
    if (t != num_steps() - 1) {
      // There is no constrol input at the final timestep
      gt += tau[t + 1].transpose() * 2 * (prob_.R + prob_.dSymmetricControlCost_dtaudtau) * dt * dtau_dqm[t + 1];
    }
  }

  // Last step is different, because there is terminal cost and v[t+1] doesn't
  // exist
  auto gT = g->tail(nq);
  gT = tau[num_steps() - 1].transpose() * 2 * (prob_.R + prob_.dSymmetricControlCost_dtaudtau) * dt * dtau_dqp[num_steps() - 1];
  gT += (q[num_steps()] - prob_.q_nom[num_steps()]).transpose() * 2 * prob_.Qf_q;
  gT += (v[num_steps()] - prob_.v_nom[num_steps()]).transpose() * 2 * prob_.Qf_v * dvt_dqt[num_steps()];
}

template <typename T>
const ocs2::vector_s_t<T>& TrajectoryOptimizer<T>::EvalGradient(const TrajectoryOptimizerState<T>& state) const {
  if (!state.cache().gradient_up_to_date) {
    CalcGradient(state, &state.mutable_cache().gradient);
    state.mutable_cache().gradient_up_to_date = true;
  }
  return state.cache().gradient;
}

template <typename T>
void TrajectoryOptimizer<T>::CalcHessian(const TrajectoryOptimizerState<T>& state, PentaDiagonalMatrix<T>* H) const {
  assert(H->is_symmetric());
  assert(H->block_rows() == num_steps() + 1);
  assert(H->block_size() == model_.nq);
  //   INSTRUMENT_FUNCTION("Assembly of the Hessian.");

  // Some convienient aliases
  const double dt = time_step();
  const ocs2::matrix_s_t<T> Qq = 2 * prob_.Qq * dt;
  const ocs2::matrix_s_t<T> Qv = 2 * prob_.Qv * dt;
  const ocs2::matrix_s_t<T> R = 2 * (prob_.R + prob_.dSymmetricControlCost_dtaudtau) * dt;
  const ocs2::matrix_s_t<T> Qf_q = 2 * prob_.Qf_q;
  const ocs2::matrix_s_t<T> Qf_v = 2 * prob_.Qf_v;

  const VelocityPartials<T>& v_partials = EvalVelocityPartials(state);
  const InverseDynamicsPartials<T>& id_partials = EvalInverseDynamicsPartials(state);
  const std::vector<ocs2::matrix_s_t<T>>& dvt_dqt = v_partials.dvt_dqt;
  const std::vector<ocs2::matrix_s_t<T>>& dvt_dqm = v_partials.dvt_dqm;
  const std::vector<ocs2::matrix_s_t<T>>& dtau_dqp = id_partials.dtau_dqp;
  const std::vector<ocs2::matrix_s_t<T>>& dtau_dqt = id_partials.dtau_dqt;
  const std::vector<ocs2::matrix_s_t<T>>& dtau_dqm = id_partials.dtau_dqm;

  // Get mutable references to the non-zero bands of the Hessian
  std::vector<ocs2::matrix_s_t<T>>& A = H->mutable_A();  // 2 rows below diagonal
  std::vector<ocs2::matrix_s_t<T>>& B = H->mutable_B();  // 1 row below diagonal
  std::vector<ocs2::matrix_s_t<T>>& C = H->mutable_C();  // diagonal

  // Fill in the non-zero blocks
  C[0].setIdentity();  // Initial condition q0 fixed at t=0
  for (int t = 1; t < num_steps(); ++t) {
    // dg_t/dq_t
    ocs2::matrix_s_t<T>& dgt_dqt = C[t];
    dgt_dqt = Qq;
    dgt_dqt += dvt_dqt[t].transpose() * Qv * dvt_dqt[t];
    dgt_dqt += dtau_dqp[t - 1].transpose() * R * dtau_dqp[t - 1];
    dgt_dqt += dtau_dqt[t].transpose() * R * dtau_dqt[t];
    if (t < num_steps() - 1) {
      dgt_dqt += dtau_dqm[t + 1].transpose() * R * dtau_dqm[t + 1];
      dgt_dqt += dvt_dqm[t + 1].transpose() * Qv * dvt_dqm[t + 1];
    } else {
      dgt_dqt += dvt_dqm[t + 1].transpose() * Qf_v * dvt_dqm[t + 1];
    }

    // dg_t/dq_{t+1}
    ocs2::matrix_s_t<T>& dgt_dqp = B[t + 1];
    dgt_dqp = dtau_dqp[t].transpose() * R * dtau_dqt[t];
    if (t < num_steps() - 1) {
      dgt_dqp += dtau_dqt[t + 1].transpose() * R * dtau_dqm[t + 1];
      dgt_dqp += dvt_dqt[t + 1].transpose() * Qv * dvt_dqm[t + 1];
    } else {
      dgt_dqp += dvt_dqt[t + 1].transpose() * Qf_v * dvt_dqm[t + 1];
    }

    // dg_t/dq_{t+2}
    if (t < num_steps() - 1) {
      ocs2::matrix_s_t<T>& dgt_dqpp = A[t + 2];
      dgt_dqpp = dtau_dqp[t + 1].transpose() * R * dtau_dqm[t + 1];
    }
  }

  // dg_t/dq_t for the final timestep
  ocs2::matrix_s_t<T>& dgT_dqT = C[num_steps()];
  dgT_dqT = Qf_q;
  dgT_dqT += dvt_dqt[num_steps()].transpose() * Qf_v * dvt_dqt[num_steps()];
  dgT_dqT += dtau_dqp[num_steps() - 1].transpose() * R * dtau_dqp[num_steps() - 1];

  // Copy lower triangular part to upper triangular part
  H->MakeSymmetric();
}

template <typename T>
const PentaDiagonalMatrix<T>& TrajectoryOptimizer<T>::EvalHessian(const TrajectoryOptimizerState<T>& state) const {
  if (!state.cache().hessian_up_to_date) {
    CalcHessian(state, &state.mutable_cache().hessian);
    state.mutable_cache().hessian_up_to_date = true;
  }
  return state.cache().hessian;
}

template <typename T>
void TrajectoryOptimizer<T>::CalcScaledHessian(const TrajectoryOptimizerState<T>& state, PentaDiagonalMatrix<T>* Htilde) const {
  const PentaDiagonalMatrix<T>& H = EvalHessian(state);
  const ocs2::vector_s_t<T>& D = EvalScaleFactors(state);
  *Htilde = PentaDiagonalMatrix<T>(H);
  Htilde->ScaleByDiagonal(D);
}

template <typename T>
const PentaDiagonalMatrix<T>& TrajectoryOptimizer<T>::EvalScaledHessian(const TrajectoryOptimizerState<T>& state) const {
  // Early exit if we're not using scaling
  if (!params_.scaling)
    return EvalHessian(state);

  if (!state.cache().scaled_hessian_up_to_date) {
    CalcScaledHessian(state, &state.mutable_cache().scaled_hessian);
    state.mutable_cache().scaled_hessian_up_to_date = true;
  }
  return state.cache().scaled_hessian;
}

template <typename T>
void TrajectoryOptimizer<T>::CalcScaledGradient(const TrajectoryOptimizerState<T>& state, ocs2::vector_s_t<T>* gtilde) const {
  const ocs2::vector_s_t<T>& g = EvalGradient(state);
  const ocs2::vector_s_t<T>& D = EvalScaleFactors(state);
  *gtilde = D.asDiagonal() * g;
}

template <typename T>
const ocs2::vector_s_t<T>& TrajectoryOptimizer<T>::EvalScaledGradient(const TrajectoryOptimizerState<T>& state) const {
  // Early exit if we're not using scaling
  if (!params_.scaling)
    return EvalGradient(state);

  if (!state.cache().scaled_gradient_up_to_date) {
    CalcScaledGradient(state, &state.mutable_cache().scaled_gradient);
    state.mutable_cache().scaled_gradient_up_to_date = true;
  }
  return state.cache().scaled_gradient;
}

template <typename T>
void TrajectoryOptimizer<T>::CalcScaleFactors(const TrajectoryOptimizerState<T>& state, ocs2::vector_s_t<T>* D) const {
  using std::min;
  using std::sqrt;

  const PentaDiagonalMatrix<T>& H = EvalHessian(state);
  ocs2::vector_s_t<T>& hessian_diag = state.workspace.num_vars_size_tmp1;
  H.ExtractDiagonal(&hessian_diag);

  for (int i = 0; i < D->size(); ++i) {
    switch (params_.scaling_method) {
      case ScalingMethod::kSqrt: {
        (*D)[i] = min(1.0, 1 / sqrt(hessian_diag[i]));
        break;
      }
      case ScalingMethod::kAdaptiveSqrt: {
        (*D)[i] = min((*D)[i], 1 / sqrt(hessian_diag[i]));
        break;
      }
      case ScalingMethod::kDoubleSqrt: {
        (*D)[i] = min(1.0, 1 / sqrt(sqrt(hessian_diag[i])));
        break;
      }
      case ScalingMethod::kAdaptiveDoubleSqrt: {
        (*D)[i] = min((*D)[i], 1 / sqrt(sqrt(hessian_diag[i])));
        break;
      }
    }
  }
}

template <typename T>
const ocs2::vector_s_t<T>& TrajectoryOptimizer<T>::EvalScaleFactors(const TrajectoryOptimizerState<T>& state) const {
  if (!state.cache().scale_factors_up_to_date) {
    CalcScaleFactors(state, &state.mutable_cache().scale_factors);
    state.mutable_cache().scale_factors_up_to_date = true;
  }
  return state.cache().scale_factors;
}

template <typename T>
void TrajectoryOptimizer<T>::CalcEqualityConstraintViolations(const TrajectoryOptimizerState<T>& state, ocs2::vector_s_t<T>* violations) const {
  //   INSTRUMENT_FUNCTION("Assemble torques on unactuated dofs.");
  const std::vector<ocs2::vector_s_t<T>>& tau = EvalTau(state);
  const int num_unactuated_dofs = unactuated_dofs().size();

  for (int t = 0; t < num_steps(); ++t) {
    for (int j = 0; j < num_unactuated_dofs; ++j) {
      (*violations)(t * num_unactuated_dofs + j) = tau[t][unactuated_dofs()[j]];
    }
  }
}

template <typename T>
const ocs2::vector_s_t<T>& TrajectoryOptimizer<T>::EvalEqualityConstraintViolations(const TrajectoryOptimizerState<T>& state) const {
  if (!state.cache().constraint_violation_up_to_date) {
    CalcEqualityConstraintViolations(state, &state.mutable_cache().constraint_violation);
    state.mutable_cache().constraint_violation_up_to_date = true;
  }
  return state.cache().constraint_violation;
}

template <typename T>
void TrajectoryOptimizer<T>::CalcEqualityConstraintJacobian(const TrajectoryOptimizerState<T>& state, ocs2::matrix_s_t<T>* J) const {
  //   INSTRUMENT_FUNCTION("Assemble equality constraint Jacobian.");
  assert(J->cols() == (num_steps() + 1) * model_.nq);
  assert(J->rows() == num_equality_constraints());

  const InverseDynamicsPartials<T>& id_partials = EvalInverseDynamicsPartials(state);

  const int nq = model_.nq;
  const int n_steps = num_steps();
  const int n_unactuated = unactuated_dofs().size();

  for (int t = 0; t < n_steps; ++t) {
    for (int i = 0; i < n_unactuated; ++i) {
      // ∂hₜⁱ/∂qₜ₊₁
      J->block(t * n_unactuated + i, (t + 1) * nq, 1, nq) = id_partials.dtau_dqp[t].row(unactuated_dofs()[i]);

      // ∂hₜⁱ/∂qₜ
      if (t > 0) {
        J->block(t * n_unactuated + i, t * nq, 1, nq) = id_partials.dtau_dqt[t].row(unactuated_dofs()[i]);
      }

      // ∂hₜⁱ/∂qₜ₋₁
      if (t > 1) {
        J->block(t * n_unactuated + i, (t - 1) * nq, 1, nq) = id_partials.dtau_dqm[t].row(unactuated_dofs()[i]);
      }
    }
  }

  // With scaling enabled, the KKT conditions become
  //   [ DHD  DJ'][Δq] = [-g]
  //   [ JD    0 ][ λ]   [-h]
  // so we'll return the scaled version of the constraint Jacobian J̃ = JD
  if (params_.scaling) {
    const ocs2::vector_s_t<T>& D = EvalScaleFactors(state);
    *J = (*J) * D.asDiagonal();
  }
}

template <typename T>
const ocs2::matrix_s_t<T>& TrajectoryOptimizer<T>::EvalEqualityConstraintJacobian(const TrajectoryOptimizerState<T>& state) const {
  if (!state.cache().constraint_jacobian_up_to_date) {
    CalcEqualityConstraintJacobian(state, &state.mutable_cache().constraint_jacobian);
    state.mutable_cache().constraint_jacobian_up_to_date = true;
  }
  return state.cache().constraint_jacobian;
}

template <typename T>
void TrajectoryOptimizer<T>::CalcLagrangeMultipliers(const TrajectoryOptimizerState<T>&, ocs2::vector_s_t<T>*) const {
  // We need to perform linear system solves to compute the lagrange
  // multipliers, so we don't support autodiff here.
  throw std::runtime_error("CalcLagrangeMultipliers() only supports T=double");
}

template <>
void TrajectoryOptimizer<double>::CalcLagrangeMultipliers(const TrajectoryOptimizerState<double>& state, ocs2::vector_t* lambda) const {
  //   INSTRUMENT_FUNCTION("Compute lagrange multipliers.");
  // λ = (J H⁻¹ Jᵀ)⁻¹ (h − J H⁻¹ g)
  const PentaDiagonalMatrix<double>& H = EvalScaledHessian(state);
  const ocs2::vector_t& g = EvalScaledGradient(state);
  const ocs2::vector_t& h = EvalEqualityConstraintViolations(state);
  const ocs2::matrix_t& J = EvalEqualityConstraintJacobian(state);

  // compute H⁻¹ Jᵀ
  // TODO(vincekurtz): add options for other linear systems solvers
  ocs2::matrix_t& Hinv_JT = state.workspace.num_vars_by_num_eq_cons_tmp;
  Hinv_JT = J.transpose();
  PentaDiagonalFactorization Hlu(H);
  assert(Hlu.status() == PentaDiagonalFactorizationStatus::kSuccess);
  for (int i = 0; i < Hinv_JT.cols(); ++i) {
    // We need this variable to avoid taking the address of a temporary object
    ocs2::vector_t ith_column = Hinv_JT.col(i);
    Hlu.SolveInPlace(&ith_column);
    Hinv_JT.col(i) = ith_column;
  }

  // TODO(vincekurtz): it may be possible to exploit the structure of JH⁻¹Jᵀ to
  // perform this step more efficiently.
  *lambda = (J * Hinv_JT).ldlt().solve(h - Hinv_JT.transpose() * g);
}

template <typename T>
const ocs2::vector_s_t<T>& TrajectoryOptimizer<T>::EvalLagrangeMultipliers(const TrajectoryOptimizerState<T>& state) const {
  // We shouldn't be calling this unless equality constraints are enabled
  assert(params_.equality_constraints);

  if (!state.cache().lagrange_multipliers_up_to_date) {
    CalcLagrangeMultipliers(state, &state.mutable_cache().lagrange_multipliers);
    state.mutable_cache().lagrange_multipliers_up_to_date = true;
  }
  return state.cache().lagrange_multipliers;
}

template <typename T>
void TrajectoryOptimizer<T>::CalcMeritFunction(const TrajectoryOptimizerState<T>& state, T* merit) const {
  const T& L = EvalCost(state);
  const ocs2::vector_s_t<T>& h = EvalEqualityConstraintViolations(state);
  const ocs2::vector_s_t<T>& lambda = EvalLagrangeMultipliers(state);

  *merit = L + h.dot(lambda);
}

template <typename T>
const T TrajectoryOptimizer<T>::EvalMeritFunction(const TrajectoryOptimizerState<T>& state) const {
  // If we're not using equality constraints, the merit function is simply the
  // unconstrained cost.
  if (!params_.equality_constraints)
    return EvalCost(state);

  if (!state.cache().merit_up_to_date) {
    CalcMeritFunction(state, &state.mutable_cache().merit);
    state.mutable_cache().merit_up_to_date = true;
  }
  return state.cache().merit;
}

template <typename T>
void TrajectoryOptimizer<T>::CalcMeritFunctionGradient(const TrajectoryOptimizerState<T>& state, ocs2::vector_s_t<T>* g_tilde) const {
  const ocs2::vector_s_t<T>& g = EvalScaledGradient(state);
  const ocs2::vector_s_t<T>& lambda = EvalLagrangeMultipliers(state);
  const ocs2::matrix_s_t<T>& J = EvalEqualityConstraintJacobian(state);

  *g_tilde = g + J.transpose() * lambda;
}

template <typename T>
const ocs2::vector_s_t<T>& TrajectoryOptimizer<T>::EvalMeritFunctionGradient(const TrajectoryOptimizerState<T>& state) const {
  // If we're not using equality constraints, just return the regular gradient.
  if (!params_.equality_constraints)
    return EvalScaledGradient(state);

  if (!state.cache().merit_gradient_up_to_date) {
    CalcMeritFunctionGradient(state, &state.mutable_cache().merit_gradient);
    state.mutable_cache().merit_gradient_up_to_date = true;
  }
  return state.cache().merit_gradient;
}

template <typename T>
void TrajectoryOptimizer<T>::CalcCacheTrajectoryData(const TrajectoryOptimizerState<T>& state) const {
  TrajectoryOptimizerCache<T>& cache = state.mutable_cache();

  // The generalized positions that everything is computed from
  const std::vector<ocs2::vector_s_t<T>>& q = state.q();

  // Compute corresponding generalized velocities
  std::vector<ocs2::vector_s_t<T>>& v = cache.trajectory_data.v;
  const std::vector<ocs2::matrix_s_t<T>>& Nplus = EvalNplus(state);
  CalcVelocities(q, Nplus, &v);

  // Compute corresponding generalized accelerations
  std::vector<ocs2::vector_s_t<T>>& a = cache.trajectory_data.a;
  CalcAccelerations(v, &a);

  // Set cache invalidation flag
  cache.trajectory_data.up_to_date = true;
}

template <typename T>
void TrajectoryOptimizer<T>::CalcInverseDynamicsCache(const TrajectoryOptimizerState<T>& state,
    typename TrajectoryOptimizerCache<T>::InverseDynamicsCache* cache) const {
  // Compute corresponding generalized torques
  const std::vector<ocs2::vector_s_t<T>>& a = EvalA(state);
  CalcInverseDynamics(state, a, &cache->tau);

  // Set cache invalidation flag
  cache->up_to_date = true;
}

template <typename T>
void TrajectoryOptimizer<T>::CalcContextCache(const TrajectoryOptimizerState<T>& state, typename TrajectoryOptimizerCache<T>::ContextCache* cache) const {
  const std::vector<ocs2::vector_s_t<T>>& q = state.q();
  const std::vector<ocs2::vector_s_t<T>>& v = EvalV(state);
  auto& plant_contexts = cache->plant_contexts;
  for (int t = 0; t <= num_steps(); ++t) {
    plant_contexts[t]->q_ = q[t];
    plant_contexts[t]->v_ = v[t];
  }
  cache->up_to_date = true;
}

template <typename T>
const Context<T>& TrajectoryOptimizer<T>::EvalPlantContext(const TrajectoryOptimizerState<T>& state, int t) const {
  if (!state.cache().context_cache->up_to_date) {
    CalcContextCache(state, state.mutable_cache().context_cache.get());
  }
  return *state.cache().context_cache->plant_contexts[t];
}

template <typename T>
Context<T>& TrajectoryOptimizer<T>::GetMutablePlantContext(const TrajectoryOptimizerState<T>& state, int t) const {
  state.mutable_cache().context_cache->up_to_date = false;
  return *state.mutable_cache().context_cache->plant_contexts[t];
}

template <typename T>
const std::vector<ocs2::vector_s_t<T>>& TrajectoryOptimizer<T>::EvalV(const TrajectoryOptimizerState<T>& state) const {
  if (!state.cache().trajectory_data.up_to_date)
    CalcCacheTrajectoryData(state);
  return state.cache().trajectory_data.v;
}

template <typename T>
const std::vector<ocs2::vector_s_t<T>>& TrajectoryOptimizer<T>::EvalA(const TrajectoryOptimizerState<T>& state) const {
  if (!state.cache().trajectory_data.up_to_date)
    CalcCacheTrajectoryData(state);
  return state.cache().trajectory_data.a;
}

template <typename T>
const std::vector<ocs2::vector_s_t<T>>& TrajectoryOptimizer<T>::EvalTau(const TrajectoryOptimizerState<T>& state) const {
  if (!state.cache().inverse_dynamics_cache.up_to_date)
    CalcInverseDynamicsCache(state, &state.mutable_cache().inverse_dynamics_cache);
  return state.cache().inverse_dynamics_cache.tau;
}

template <typename T>
void TrajectoryOptimizer<T>::CalcCacheDerivativesData(const TrajectoryOptimizerState<T>& state) const {
  TrajectoryOptimizerCache<T>& cache = state.mutable_cache();

  // Some aliases
  InverseDynamicsPartials<T>& id_partials = cache.derivatives_data.id_partials;
  VelocityPartials<T>& v_partials = cache.derivatives_data.v_partials;

  // Compute partial derivatives of inverse dynamics d(tau)/d(q)
  CalcInverseDynamicsPartials(state, &id_partials);

  // Compute partial derivatives of velocities d(v)/d(q)
  CalcVelocityPartials(state, &v_partials);

  // Set cache invalidation flag
  cache.derivatives_data.up_to_date = true;
}

template <typename T>
const VelocityPartials<T>& TrajectoryOptimizer<T>::EvalVelocityPartials(const TrajectoryOptimizerState<T>& state) const {
  if (!state.cache().derivatives_data.up_to_date)
    CalcCacheDerivativesData(state);
  return state.cache().derivatives_data.v_partials;
}

template <typename T>
const InverseDynamicsPartials<T>& TrajectoryOptimizer<T>::EvalInverseDynamicsPartials(const TrajectoryOptimizerState<T>& state) const {
  if (!state.cache().derivatives_data.up_to_date)
    CalcCacheDerivativesData(state);
  return state.cache().derivatives_data.id_partials;
}

template <typename T>
const std::vector<ocs2::matrix_s_t<T>>& TrajectoryOptimizer<T>::EvalNplus(const TrajectoryOptimizerState<T>& state) const {
  if (!state.cache().n_plus_up_to_date) {
    CalcNplus(state, &state.mutable_cache().N_plus);
    state.mutable_cache().n_plus_up_to_date = true;
  }
  return state.cache().N_plus;
}

template <typename T>
void TrajectoryOptimizer<T>::CalcNplus(const TrajectoryOptimizerState<T>& state, std::vector<ocs2::matrix_s_t<T>>* N_plus) const {
  assert(static_cast<int>(N_plus->size()) == (num_steps() + 1));
  for (int t = 0; t <= num_steps(); ++t) {
    // Get a context storing q at time t
    // TODO(vincekurtz): consider using EvalPlantContext instead. In that case
    // we do need to be a bit careful, however, since EvalPlantContext requires
    // EvalV, which in turn requires EvalNplus.

    // Compute N+(q_t)
    N_plus->at(t) = ocs2::matrix_s_t<T>::Identity(model_.nv, model_.nv);
  }
}

template <typename T>
std::tuple<double, int> TrajectoryOptimizer<T>::Linesearch(const TrajectoryOptimizerState<T>& state,
    const ocs2::vector_s_t<T>& dq,
    TrajectoryOptimizerState<T>* scratch_state) const {
  // The state's cache must be up to date, since we'll use the gradient and cost
  // information stored there.
  if (params_.linesearch_method == LinesearchMethod::kArmijo) {
    return ArmijoLinesearch(state, dq, scratch_state);
  } else if (params_.linesearch_method == LinesearchMethod::kBacktracking) {
    return BacktrackingLinesearch(state, dq, scratch_state);
  } else {
    throw std::runtime_error("Unknown linesearch method");
  }
}

template <typename T>
std::tuple<double, int> TrajectoryOptimizer<T>::BacktrackingLinesearch(const TrajectoryOptimizerState<T>& state,
    const ocs2::vector_s_t<T>& dq,
    TrajectoryOptimizerState<T>* scratch_state) const {
  using std::abs;

  // Compute the cost and gradient
  double mu = 0.0;
  if (params_.equality_constraints) {
    // Use an exact l1 penalty function as the merit function if equality
    // constraints are enforced exactly.
    // TODO(vincekurtz): add equality constraints to Armijo linesearch
    mu = 1e3;
  }
  const ocs2::vector_s_t<T>& h = EvalEqualityConstraintViolations(state);
  const T L = EvalCost(state) + mu * h.cwiseAbs().sum();
  const ocs2::vector_s_t<T>& g = EvalGradient(state);

  // Linesearch parameters
  const double c = 1e-4;
  const double rho = 0.8;

  double alpha = 1.0;
  T L_prime = g.transpose() * dq - mu * h.cwiseAbs().sum();  // gradient of L w.r.t. alpha

  // Make sure this is a descent direction
  assert(L_prime <= 0);

  // Exit early with alpha = 1 when we are close to convergence
  const double convergence_threshold = std::sqrt(std::numeric_limits<double>::epsilon());
  if (abs(L_prime) / abs(L) <= convergence_threshold) {
    return {1.0, 0};
  }

  // Try with alpha = 1
  scratch_state->set_q(state.q());
  scratch_state->AddToQ(alpha * dq);
  T L_old = EvalCost(*scratch_state) + mu * EvalEqualityConstraintViolations(*scratch_state).cwiseAbs().sum();

  // L_new stores cost at iteration i:   L(q + alpha_i * dq)
  // L_old stores cost at iteration i-1: L(q + alpha_{i-1} * dq)
  T L_new = L_old;

  // We'll keep reducing alpha until (1) we meet the Armijo convergence
  // criteria and (2) the cost increases, indicating that we're near a local
  // minimum.
  int i = 0;
  bool armijo_met = false;
  while (!(armijo_met && (L_new > L_old))) {
    // Save L_old = L(q + alpha_{i-1} * dq)
    L_old = L_new;

    // Reduce alpha
    alpha *= rho;

    // Compute L_new = L(q + alpha_i * dq)
    scratch_state->set_q(state.q());
    scratch_state->AddToQ(alpha * dq);
    L_new = EvalCost(*scratch_state) + mu * EvalEqualityConstraintViolations(*scratch_state).cwiseAbs().sum();

    // Check the Armijo conditions
    if (L_new <= L + c * alpha * L_prime) {
      armijo_met = true;
    }

    ++i;
  }

  return {alpha / rho, i};
}

template <typename T>
std::tuple<double, int> TrajectoryOptimizer<T>::ArmijoLinesearch(const TrajectoryOptimizerState<T>& state,
    const ocs2::vector_s_t<T>& dq,
    TrajectoryOptimizerState<T>* scratch_state) const {
  using std::abs;

  // Compute the cost and gradient
  const T L = EvalCost(state);
  const ocs2::vector_s_t<T>& g = EvalGradient(state);

  // Linesearch parameters
  const double c = 1e-4;
  const double rho = 0.8;

  double alpha = 1.0 / rho;        // get alpha = 1 on first iteration
  T L_prime = g.transpose() * dq;  // gradient of L w.r.t. alpha
  T L_new;                         // L(q + alpha * dq)

  // Make sure this is a descent direction
  assert(L_prime <= 0);

  // Exit early with alpha = 1 when we are close to convergence
  const double convergence_threshold = 10 * std::numeric_limits<double>::epsilon() / time_step() / time_step();
  if (abs(L_prime) / abs(L) <= convergence_threshold) {
    return {1.0, 0};
  }

  int i = 0;  // Iteration counter
  do {
    // Reduce alpha
    // N.B. we start with alpha = 1/rho, so we get alpha = 1 on the first
    // iteration.
    alpha *= rho;

    // Compute L_ls = L(q + alpha * dq)
    scratch_state->set_q(state.q());
    scratch_state->AddToQ(alpha * dq);
    L_new = EvalCost(*scratch_state);

    ++i;
  } while ((L_new > L + c * alpha * L_prime) && (i < params_.max_linesearch_iterations));

  return {alpha, i};
}

template <typename T>
T TrajectoryOptimizer<T>::CalcTrustRatio(const TrajectoryOptimizerState<T>& state,
    const ocs2::vector_s_t<T>& dq,
    TrajectoryOptimizerState<T>* scratch_state) const {
  // Quantities at the current iteration (k)
  const T merit_k = EvalMeritFunction(state);
  const ocs2::vector_s_t<T>& g_tilde_k = EvalMeritFunctionGradient(state);
  const PentaDiagonalMatrix<T>& H_k = EvalScaledHessian(state);

  // Quantities at the next iteration if we accept the step (kp = k+1)
  // TODO(vincekurtz): if we do end up accepting the step, it would be nice to
  // somehow reuse these cached quantities, which we're currently trashing
  scratch_state->set_q(state.q());
  scratch_state->AddToQ(dq);
  T merit_kp = EvalCost(*scratch_state);
  if (params_.equality_constraints) {
    // N.B. We use λₖ rather than λₖ₊₁ to compute the merit function
    // ϕₖ₊₁ = L(qₖ₊₁) + h(qₖ₊₁)ᵀλₖ here because we are assuming that λ is
    // constant.
    const ocs2::vector_s_t<T>& lambda_k = EvalLagrangeMultipliers(state);
    const ocs2::vector_s_t<T>& h_kp = EvalEqualityConstraintViolations(*scratch_state);
    merit_kp += h_kp.dot(lambda_k);
  }

  // Compute predicted reduction in the merit function, −gᵀΔq − 1/2 ΔqᵀHΔq
  ocs2::vector_s_t<T>& dq_scaled = state.workspace.num_vars_size_tmp1;
  if (params_.scaling) {
    const ocs2::vector_s_t<T>& D = EvalScaleFactors(state);
    // TODO(vincekurtz): consider caching D^{-1}
    dq_scaled = D.cwiseInverse().asDiagonal() * dq;
  } else {
    dq_scaled = dq;
  }
  ocs2::vector_s_t<T>& Hdq = state.workspace.num_vars_size_tmp2;
  H_k.MultiplyBy(dq_scaled, &Hdq);  // Hdq = H_k * dq
  const T hessian_term = 0.5 * dq_scaled.transpose() * Hdq;
  T gradient_term = g_tilde_k.dot(dq_scaled);
  const T predicted_reduction = -gradient_term - hessian_term;

  // Compute actual reduction in the merit function
  const T actual_reduction = merit_k - merit_kp;

  // Threshold for determining when the actual and predicted reduction in cost
  // are essentially zero. This is determined by the approximate level of
  // floating point error in our computation of the cost, L(q).
  const double eps = 10 * std::numeric_limits<T>::epsilon() / time_step() / time_step();
  if ((predicted_reduction < eps) && (actual_reduction < eps)) {
    // Actual and predicted improvements are both essentially zero, so we set
    // the trust ratio to a value such that the step will be accepted, but the
    // size of the trust region will not change.
    return 0.5;
  }

  return actual_reduction / predicted_reduction;
}

template <typename T>
T TrajectoryOptimizer<T>::SolveDoglegQuadratic(const T& a, const T& b, const T& c) const {
  using std::sqrt;
  // Check that a is positive
  assert(a > 0);

  T s;
  if (a < std::numeric_limits<double>::epsilon()) {
    // If a is essentially zero, just solve bx + c = 0
    s = -c / b;
  } else {
    // Normalize everything by a
    const T b_tilde = b / a;
    const T c_tilde = c / a;

    const T determinant = b_tilde * b_tilde - 4 * c_tilde;
    assert(determinant > 0);  // We know a real root exists

    // We know that there is only one positive root, so we just take the big
    // root
    s = (-b_tilde + sqrt(determinant)) / 2;
  }

  // We know the solution is between zero and one
  assert(0 < s);
  assert(s < 1);

  return s;
}

template <typename T>
void TrajectoryOptimizer<T>::SolveLinearSystemInPlace(const PentaDiagonalMatrix<T>&, ocs2::vector_s_t<T>*) const {
  // Only T=double is supported here, since most of our solvers only support
  // double.
  throw std::runtime_error("TrajectoryOptimizer::SolveLinearSystemInPlace() only supports T=double");
}

template <>
void TrajectoryOptimizer<double>::SolveLinearSystemInPlace(const PentaDiagonalMatrix<double>& H, ocs2::vector_t* b) const {
  switch (params_.linear_solver) {
    case SolverParameters::LinearSolverType::kPentaDiagonalLu: {
      PentaDiagonalFactorization Hlu(H);
      assert(Hlu.status() == PentaDiagonalFactorizationStatus::kSuccess);
      Hlu.SolveInPlace(b);
      break;
    }
    case SolverParameters::LinearSolverType::kDenseLdlt: {
      const ocs2::matrix_t Hdense = H.MakeDense();
      const auto& Hldlt = Hdense.ldlt();
      *b = Hldlt.solve(*b);
      assert(Hldlt.info() == Eigen::Success);
      break;
    }
  }
}

template <typename T>
bool TrajectoryOptimizer<T>::CalcDoglegPoint(const TrajectoryOptimizerState<T>&, const double, ocs2::vector_s_t<T>*, ocs2::vector_s_t<T>*) const {
  // Only T=double is supported here, since pentadigonal matrix factorization is
  // (sometimes) required to compute the dogleg point.
  throw std::runtime_error("TrajectoryOptimizer::CalcDoglegPoint only supports T=double");
}

template <>
bool TrajectoryOptimizer<double>::CalcDoglegPoint(const TrajectoryOptimizerState<double>& state,
    const double Delta,
    ocs2::vector_t* dq,
    ocs2::vector_t* dqH) const {
  //   INSTRUMENT_FUNCTION("Find search direction with dogleg method.");

  // If params_.scaling = false, this returns the regular Hessian.
  const PentaDiagonalMatrix<double>& H = EvalScaledHessian(state);

  // If equality constraints are active, we'll use the gradient of the merit
  // function, g̃ = g + J'λ. This means the full step pH satisfies the KKT
  // conditions
  //     [H  J']*[pH] = [-g]
  //     [J  0 ] [ λ]   [-h]
  // while the shortened step pU minimizes the quadratic approximation in the
  // direction of -g - J'λ.
  const ocs2::vector_t& g = EvalMeritFunctionGradient(state);

  ocs2::vector_t& Hg = state.workspace.num_vars_size_tmp1;
  H.MultiplyBy(g, &Hg);
  const double gHg = g.transpose() * Hg;

  // Compute the full Gauss-Newton step
  // N.B. We can avoid computing pH when pU is the dog-leg solution.
  // However, we compute it here for logging stats since thus far the cost of
  // computing pH is negligible compared to other costs (namely the computation
  // of gradients of the inverse dynamics.)
  // TODO(amcastro-tri): move this to after pU whenever we make the cost of
  // gradients computation negligible.
  ocs2::vector_t& pH = state.workspace.num_vars_size_tmp2;

  pH = -g / Delta;  // normalize by Δ
  SolveLinearSystemInPlace(H, &pH);

  if (params_.debug_compare_against_dense) {
    // From experiments in penta_diagonal_solver_test.cc
    // (PentaDiagonalMatrixTest.SolvePentaDiagonal), LDLT is the most stable
    // solver to round-off errors. We therefore use it as a reference solution
    // for debugging.
    const ocs2::vector_t pH_dense = H.MakeDense().ldlt().solve(-g / Delta);
  }

  *dqH = pH * Delta;

  // Compute the unconstrained minimizer of m(δq) = L(q) + g(q)'*δq + 1/2
  // δq'*H(q)*δq along -g
  ocs2::vector_t& pU = state.workspace.num_vars_size_tmp3;
  pU = -(g.dot(g) / gHg) * g / Delta;  // normalize by Δ

  // Check if the trust region is smaller than this unconstrained minimizer
  if (1.0 <= pU.norm()) {
    // If so, δq is where the first leg of the dogleg path intersects the trust
    // region.
    *dq = (Delta / pU.norm()) * pU;
    if (params_.scaling) {
      *dq = EvalScaleFactors(state).asDiagonal() * (*dq);
    }
    return true;  // the trust region constraint is active
  }

  // Check if the trust region is large enough to just take the full Newton step
  if (1.0 >= pH.norm()) {
    *dq = pH * Delta;
    if (params_.scaling) {
      // TODO(vincekurtz): consider adding a MultiplyByScaleFactors method
      *dq = EvalScaleFactors(state).asDiagonal() * (*dq);
    }
    return false;  // the trust region constraint is not active
  }

  // Compute the intersection between the second leg of the dogleg path and the
  // trust region. We'll do this by solving the (scalar) quadratic
  //
  //    ‖ pU + s( pH − pU ) ‖² = y²
  //
  // for s ∈ (0,1),
  //
  // and setting
  //
  //    δq = pU + s( pH − pU ).
  //
  // Note that we normalize by Δ to minimize roundoff error.
  const double a = (pH - pU).dot(pH - pU);
  const double b = 2 * pU.dot(pH - pU);
  const double c = pU.dot(pU) - 1.0;
  const double s = SolveDoglegQuadratic(a, b, c);

  *dq = (pU + s * (pH - pU)) * Delta;
  if (params_.scaling) {
    *dq = EvalScaleFactors(state).asDiagonal() * (*dq);
  }
  return true;  // the trust region constraint is active
}

template <typename T>
SolverFlag TrajectoryOptimizer<T>::Solve(const std::vector<ocs2::vector_s_t<T>>&,
    TrajectoryOptimizerSolution<T>*,
    TrajectoryOptimizerStats<T>*,
    ConvergenceReason*) const {
  throw std::runtime_error("TrajectoryOptimizer::Solve only supports T=double.");
}

template <>
SolverFlag TrajectoryOptimizer<double>::Solve(const std::vector<ocs2::vector_t>& q_guess,
    TrajectoryOptimizerSolution<double>* solution,
    TrajectoryOptimizerStats<double>* stats,
    ConvergenceReason* reason) const {
  //   INSTRUMENT_FUNCTION("Main entry point.");

  // The guess must be consistent with the initial condition
  assert(q_guess[0] == prob_.q_init);
  assert(static_cast<int>(q_guess.size()) == num_steps() + 1);

  // stats must be empty
  assert(stats->is_empty());

  if (params_.method == SolverMethod::kLinesearch) {
    return SolveWithLinesearch(q_guess, solution, stats);
  } else if (params_.method == SolverMethod::kTrustRegion) {
    return SolveWithTrustRegion(q_guess, solution, stats, reason);
  } else {
    throw std::runtime_error("Unsupported solver strategy!");
  }
}

template <typename T>
SolverFlag TrajectoryOptimizer<T>::SolveWithLinesearch(const std::vector<ocs2::vector_s_t<T>>&,
    TrajectoryOptimizerSolution<T>*,
    TrajectoryOptimizerStats<T>*) const {
  throw std::runtime_error("TrajectoryOptimizer::SolveWithLinesearch only supports T=double.");
}

template <>
SolverFlag TrajectoryOptimizer<double>::SolveWithLinesearch(const std::vector<ocs2::vector_t>& q_guess,
    TrajectoryOptimizerSolution<double>* solution,
    TrajectoryOptimizerStats<double>* stats) const {
  // Allocate a state variable
  TrajectoryOptimizerState<double> state = CreateState();
  state.set_q(q_guess);

  // Allocate a separate state variable for linesearch
  TrajectoryOptimizerState<double> scratch_state = CreateState();

  // Allocate cost and search direction
  double cost;
  ocs2::vector_t dq((num_steps() + 1) * model_.nq);

  if (params_.verbose) {
    // Define printout data
    std::cout << "-------------------------------------------------------------"
                 "----------------------"
              << std::endl;
    std::cout << "|  iter  |   cost   |  alpha  |  LS_iters  |  time (s)  |  "
                 "|g|/cost  |    |h|     |"
              << std::endl;
    std::cout << "-------------------------------------------------------------"
                 "----------------------"
              << std::endl;
  }

  // Allocate timing variables
  auto start_time = std::chrono::high_resolution_clock::now();
  auto iter_start_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> iter_time;
  std::chrono::duration<double> solve_time;

  // Gauss-Newton iterations
  int k = 0;                       // iteration counter
  bool linesearch_failed = false;  // linesearch success flag
  do {
    iter_start_time = std::chrono::high_resolution_clock::now();

    // Compute the total cost
    cost = EvalCost(state);

    // Evaluate constraint violations (for logging)
    const ocs2::vector_t& h = EvalEqualityConstraintViolations(state);

    // Compute gradient and Hessian
    const ocs2::vector_t& g = EvalMeritFunctionGradient(state);
    const PentaDiagonalMatrix<double>& H = EvalHessian(state);

    // Compute the search direction. If equality constraints are active, this
    // solves the KKT conditions
    //    [H  J']*[dq] = [-g]
    //    [J  0 ] [ λ]   [-h]
    // for the search direction (since we've defined g as g + J'λ). Otherwise,
    // we solve H*dq = -g.
    dq = -g;
    SolveLinearSystemInPlace(H, &dq);

    // Solve the linsearch
    // N.B. we use a separate state variable since we will need to compute
    // L(q+alpha*dq) (at the very least), and we don't want to change state.q
    auto [alpha, ls_iters] = Linesearch(state, dq, &scratch_state);

    if (ls_iters >= params_.max_linesearch_iterations) {
      linesearch_failed = true;

      if (params_.verbose) {
        std::cout << "LINESEARCH FAILED" << std::endl;
        std::cout << "Reached maximum linesearch iterations (" << params_.max_linesearch_iterations << ")." << std::endl;
      }
    }

    // Compute the trust ratio (actual cost reduction / model cost reduction)
    double trust_ratio = CalcTrustRatio(state, alpha * dq, &scratch_state);

    // Update the decision variables
    state.AddToQ(alpha * dq);

    iter_time = std::chrono::high_resolution_clock::now() - iter_start_time;

    // Nice little printout of our problem data
    if (params_.verbose) {
      printf("| %6d ", k);
      printf("| %8.3f ", cost);
      printf("| %7.4f ", alpha);
      printf("| %6d     ", ls_iters);
      printf("| %8.8f ", iter_time.count());
      printf("| %10.3e ", g.norm() / cost);
      printf("| %10.3e |\n", h.norm());
    }

    // Print additional debuging information
    if (params_.print_debug_data) {
      double condition_number = 1 / H.MakeDense().ldlt().rcond();
      double L_prime = g.transpose() * dq;
      std::cout << "Condition #: " << condition_number << std::endl;
      std::cout << "|| dq ||   : " << dq.norm() << std::endl;
      std::cout << "||  g ||   : " << g.norm() << std::endl;
      std::cout << "L'         : " << L_prime << std::endl;
      std::cout << "L          : " << cost << std::endl;
      std::cout << "L' / L     : " << L_prime / cost << std::endl;
      std::cout << "||diag(H)||: " << H.MakeDense().diagonal().norm() << std::endl;
      if (k > 0) {
        std::cout << "L[k] - L[k-1]: " << cost - stats->iteration_costs[k - 1] << std::endl;
      }
    }

    const double dL_dq = g.dot(dq) / cost;

    // Record iteration data
    stats->push_data(iter_time.count(),  // iteration time
        cost,                            // cost
        ls_iters,                        // sub-problem iterations
        alpha,                           // linesearch parameter
        NAN,                             // trust region size
        state.norm(),                    // q norm
        dq.norm(),                       // step size
        dq.norm(),                       // step size
        trust_ratio,                     // trust ratio
        g.norm(),                        // gradient size
        dL_dq,                           // gradient along dq
        h.norm(),                        // equality constraint violation
        cost);                           // merit function

    ++k;
  } while (k < params_.max_iterations && !linesearch_failed);

  // End the problem data printout
  if (params_.verbose) {
    std::cout << "-------------------------------------------------------------"
                 "----------------------"
              << std::endl;
  }

  // Record the total solve time
  solve_time = std::chrono::high_resolution_clock::now() - start_time;
  stats->solve_time = solve_time.count();

  // Record the solution
  solution->q = state.q();
  solution->v = EvalV(state);
  solution->tau = EvalTau(state);

  if (linesearch_failed) {
    return SolverFlag::kLinesearchMaxIters;
  } else {
    return SolverFlag::kSuccess;
  }
}

template <typename T>
SolverFlag TrajectoryOptimizer<T>::SolveWithTrustRegion(const std::vector<ocs2::vector_s_t<T>>&,
    TrajectoryOptimizerSolution<T>*,
    TrajectoryOptimizerStats<T>*,
    ConvergenceReason*) const {
  throw std::runtime_error("TrajectoryOptimizer::SolveWithTrustRegion only supports T=double.");
}

template <>
SolverFlag TrajectoryOptimizer<double>::SolveWithTrustRegion(const std::vector<ocs2::vector_t>& q_guess,
    TrajectoryOptimizerSolution<double>* solution,
    TrajectoryOptimizerStats<double>* stats,
    ConvergenceReason* reason_out) const {
  //   INSTRUMENT_FUNCTION("Trust region solver.");

  // Allocate a warm start, which includes the initial guess along with state
  // variables and the trust region radius.
  WarmStart warm_start(num_steps(), model_, data_, num_equality_constraints(), q_guess, params_.Delta0);

  return SolveFromWarmStart(&warm_start, solution, stats, reason_out);
  return SolverFlag::kSuccess;
}

template <typename T>
SolverFlag TrajectoryOptimizer<T>::SolveFromWarmStart(WarmStart*, TrajectoryOptimizerSolution<T>*, TrajectoryOptimizerStats<T>*, ConvergenceReason*) const {
  throw std::runtime_error("TrajectoryOptimizer::SolveFromWarmStart only supports T=double.");
}

template <>
SolverFlag TrajectoryOptimizer<double>::SolveFromWarmStart(WarmStart* warm_start,
    TrajectoryOptimizerSolution<double>* solution,
    TrajectoryOptimizerStats<double>* stats,
    ConvergenceReason* reason_out) const {
  using std::min;
  //   INSTRUMENT_FUNCTION("Solve with warm start.");

  // Allocate timing variables
  auto start_time = std::chrono::high_resolution_clock::now();
  auto iter_start_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> iter_time;
  std::chrono::duration<double> solve_time;

  // Warm-starting doesn't support the linesearch method
  assert(params_.method == SolverMethod::kTrustRegion);
  // State variable stores q and everything that is computed from q
  TrajectoryOptimizerState<double>& state = warm_start->state;
  TrajectoryOptimizerState<double>& scratch_state = warm_start->scratch_state;

  // The update vector q_{k+1} = q_k + dq and full Newton step (for logging)
  ocs2::vector_t& dq = warm_start->dq;
  ocs2::vector_t& dqH = warm_start->dqH;

  // Trust region parameters
  const double Delta_max = params_.Delta_max;  // Maximum trust region size
  const double eta = 0.0;                      // Trust ratio threshold - we accept steps if
                                               // the trust ratio is above this threshold

  // Variables that we'll update throughout the main loop
  int k = 0;                          // iteration counter
  double& Delta = warm_start->Delta;  // trust region size
  double rho;                         // trust region ratio
  bool tr_constraint_active;          // flag for whether the trust region
                                      // constraint is active

  // Define printout strings
  const std::string separator_bar =
      "------------------------------------------------------------------------"
      "---------------------";
  const std::string printout_labels =
      "|  iter  |   cost   |    Δ    |    ρ    |  time (s)  |  |g|/cost  | "
      "dL_dq/cost |    |h|     |";

  double previous_cost = EvalCost(state);
  while (k < params_.max_iterations) {
    // Obtain the candiate update dq
    tr_constraint_active = CalcDoglegPoint(state, Delta, &dq, &dqH);

    if (params_.print_debug_data) {
      // Print some info about the Hessian
      const ocs2::matrix_t H = EvalHessian(state).MakeDense();
      const ocs2::matrix_t H_scaled = EvalScaledHessian(state).MakeDense();
      const double condition_number = 1 / H.ldlt().rcond();
      const double condition_number_scaled = 1 / H_scaled.ldlt().rcond();
      PRINT_VAR(condition_number);
      PRINT_VAR(condition_number_scaled);
    }

    // Compute some quantities for logging.
    // N.B. These should be computed before q is updated.
    const ocs2::vector_t& g = EvalMeritFunctionGradient(state);
    const ocs2::vector_t& h = EvalEqualityConstraintViolations(state);

    const double cost = EvalCost(state);
    const double merit = EvalMeritFunction(state);
    const double q_norm = state.norm();
    double dL_dq;
    if (params_.scaling) {
      const ocs2::vector_t& D = EvalScaleFactors(state);
      dL_dq = g.dot(D.cwiseInverse().asDiagonal() * dq) / cost;
    } else {
      dL_dq = g.dot(dq) / cost;
    }

    // Compute the trust region ratio
    rho = CalcTrustRatio(state, dq, &scratch_state);

    // With a positive definite Hessian, steps should not oppose the descent
    // direction
    assert(dL_dq < std::numeric_limits<double>::epsilon());

    // If the ratio is large enough, accept the change
    if (rho > eta) {
      state.AddToQ(dq);  // q += dq
    }
    // Else (rho <= eta), the trust region ratio is too small to accept dq, so
    // we'll need to so keep reducing the trust region. Note that the trust
    // region will be reduced in this case, since eta < 0.25.

    // N.B. if this is the case (q_{k+1} = q_k), we haven't touched state, so we
    // should be reusing the cached gradient and Hessian in the next iteration.
    // TODO(vincekurtz): should we be caching the factorization of the Hessian,
    // as well as the Hessian itself?

    // Compute iteration timing
    // N.B. this is in kind of a weird place because we want to record
    // statistics before updating the trust-region size. That ensures that
    // ‖ δq ‖ ≤ Δ in our logs.
    iter_time = std::chrono::high_resolution_clock::now() - iter_start_time;
    iter_start_time = std::chrono::high_resolution_clock::now();

    // Printout statistics from this iteration
    if (params_.verbose) {
      if ((k % 50) == 0) {
        // Refresh the labels for easy reading
        std::cout << separator_bar << std::endl;
        std::cout << printout_labels << std::endl;
        std::cout << separator_bar << std::endl;
      }
    }

    // Record statistics from this iteration
    stats->push_data(iter_time.count(),  // iteration time
        cost,                            // cost
        0,                               // linesearch iterations
        NAN,                             // linesearch parameter
        Delta,                           // trust region size
        q_norm,                          // q norm
        dq.norm(),                       // step size
        dqH.norm(),                      // Unconstrained step size
        rho,                             // trust region ratio
        g.norm(),                        // gradient size
        dL_dq,                           // gradient along dq
        h.norm(),                        // equality constraint violation
        merit);                          // merit function

    // Only check convergence criteria for valid steps.
    ConvergenceReason reason{ConvergenceReason::kNoConvergenceCriteriaSatisfied};
    if (params_.check_convergence && (rho > eta)) {
      reason = VerifyConvergenceCriteria(state, previous_cost, dq);
      previous_cost = EvalCost(state);
      if (reason_out)
        *reason_out = reason;
    }

    if (reason != ConvergenceReason::kNoConvergenceCriteriaSatisfied) {
      break;
    }

    // Update the size of the trust-region, if necessary
    if (rho < 0.25) {
      // If the ratio is small, our quadratic approximation is bad, so reduce
      // the trust region
      Delta *= 0.25;
    } else if ((rho > 0.75) && tr_constraint_active) {
      // If the ratio is large and we're at the boundary of the trust
      // region, increase the size of the trust region.
      Delta = min(2 * Delta, Delta_max);
    }

    ++k;
  }

  // Finish our printout
  if (params_.verbose) {
    std::cout << separator_bar << std::endl;
  }

  solve_time = std::chrono::high_resolution_clock::now() - start_time;
  stats->solve_time = solve_time.count();

  // Record the solution
  solution->q = state.q();
  solution->v = EvalV(state);
  solution->tau = EvalTau(state);

  if (k == params_.max_iterations) {
    return SolverFlag::kMaxIterationsReached;
  }

  return SolverFlag::kSuccess;
}

template <typename T>
ConvergenceReason TrajectoryOptimizer<T>::VerifyConvergenceCriteria(const TrajectoryOptimizerState<T>& state,
    const T& previous_cost,
    const ocs2::vector_s_t<T>& dq) const {
  using std::abs;

  const auto& tolerances = params_.convergence_tolerances;

  int reason(ConvergenceReason::kNoConvergenceCriteriaSatisfied);

  // Cost reduction criterion:
  //   |Lᵏ−Lᵏ⁺¹| < εₐ + εᵣ Lᵏ⁺¹
  const T cost = EvalCost(state);
  if (abs(previous_cost - cost) < tolerances.abs_cost_reduction + tolerances.rel_cost_reduction * cost) {
    reason |= ConvergenceReason::kCostReductionCriterionSatisfied;
  }

  // Gradient criterion:
  //   g⋅Δq < εₐ + εᵣ Lᵏ
  const ocs2::vector_s_t<T>& g = EvalMeritFunctionGradient(state);
  if (abs(g.dot(dq)) < tolerances.abs_gradient_along_dq + tolerances.rel_gradient_along_dq * cost) {
    reason |= ConvergenceReason::kGradientCriterionSatisfied;
  }

  // Relative state (q) change:
  //   ‖Δq‖ < εₐ + εᵣ‖qᵏ‖
  const T q_norm = state.norm();
  const T dq_norm = dq.norm();
  if (dq_norm < tolerances.abs_state_change + tolerances.rel_state_change * q_norm) {
    reason |= ConvergenceReason::kSateCriterionSatisfied;
  }

  return ConvergenceReason(reason);
}

template class TrajectoryOptimizer<double>;

}  // namespace optimizer
}  // namespace idto
