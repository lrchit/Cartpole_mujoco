
#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include "trajectory_optimizer.h"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/container/aligned-vector.hpp>
#include <pinocchio/spatial/force.hpp>
#include "pinocchio/algorithm/crba.hpp"
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>

namespace idto {
namespace optimizer {

template <typename T>
const T TrajectoryOptimizer<T>::EvalCost(const TrajectoryOptimizerState<T>& state) const {
  if (!state.cache().cost_up_to_date) {
    state.mutable_cache().cost = CalcCost(state);
    state.mutable_cache().cost_up_to_date = true;
  }
  return state.cache().cost;
}

template <typename T>
T TrajectoryOptimizer<T>::CalcCost(const TrajectoryOptimizerState<T>& state) const {
  const std::vector<ocs2::vector_s_t<T>>& v = EvalV(state);
  const std::vector<ocs2::vector_s_t<T>>& tau = EvalTau(state);
  T cost = CalcCost(state.q(), v, tau, &state.workspace);

  return cost;
}

template <typename T>
T TrajectoryOptimizer<T>::CalcCost(const std::vector<ocs2::vector_s_t<T>>& q,
    const std::vector<ocs2::vector_s_t<T>>& v,
    const std::vector<ocs2::vector_s_t<T>>& tau,
    TrajectoryOptimizerWorkspace<T>* workspace) const {
  T cost = 0;
  ocs2::vector_s_t<T>& q_err = workspace->q_size_tmp1;
  ocs2::vector_s_t<T>& v_err = workspace->v_size_tmp1;

  // Running cost
  for (int t = 0; t < num_steps(); ++t) {
    q_err = q[t] - prob_.q_nom[t];
    v_err = v[t] - prob_.v_nom[t];
    cost += T(q_err.transpose() * prob_.Qq * q_err);
    cost += T(v_err.transpose() * prob_.Qv * v_err);
    cost += T(tau[t].transpose() * (prob_.R + prob_.dSymmetricControlCost_dtaudtau) * tau[t]);

    // std::cerr << "q_err: " << (q_err.transpose() * prob_.Qq * q_err).transpose() << std::endl;
    // std::cerr << "v_err: " << (v_err.transpose() * prob_.Qv * v_err).transpose() << std::endl;
    // std::cerr << "tau[t]: " << tau[t].transpose() << std::endl;
    // std::cerr << "tau cost: " << (tau[t].transpose() * prob_.R * tau[t]).transpose() << std::endl;
  }

  // Scale running cost by dt (so the optimization problem we're solving doesn't
  // change so dramatically when we change the time step).
  cost *= time_step();

  // Terminal cost
  q_err = q[num_steps()] - prob_.q_nom[num_steps()];
  v_err = v[num_steps()] - prob_.v_nom[num_steps()];
  cost += T(q_err.transpose() * prob_.Qf_q * q_err);
  cost += T(v_err.transpose() * prob_.Qf_v * v_err);

  //   std::cerr << "total cost: " << cost << std::endl;

  return cost;
}

template <typename T>
void TrajectoryOptimizer<T>::CalcVelocities(const std::vector<ocs2::vector_s_t<T>>& q,
    const std::vector<ocs2::matrix_s_t<T>>& Nplus,
    std::vector<ocs2::vector_s_t<T>>* v) const {
  // x = [x0, x1, ..., xT]
  assert(static_cast<int>(q.size()) == num_steps() + 1);
  assert(static_cast<int>(Nplus.size()) == num_steps() + 1);
  assert(static_cast<int>(v->size()) == num_steps() + 1);

  v->at(0) = prob_.v_init;
  for (int t = 1; t <= num_steps(); ++t) {
    v->at(t) = Nplus[t] * (q[t] - q[t - 1]) / time_step();
  }
}

template <typename T>
void TrajectoryOptimizer<T>::CalcAccelerations(const std::vector<ocs2::vector_s_t<T>>& v, std::vector<ocs2::vector_s_t<T>>* a) const {
  assert(static_cast<int>(v.size()) == num_steps() + 1);
  assert(static_cast<int>(a->size()) == num_steps());

  for (int t = 0; t < num_steps(); ++t) {
    a->at(t) = (v[t + 1] - v[t]) / time_step();
  }
}

template <typename T>
void TrajectoryOptimizer<T>::CalcInverseDynamics(const TrajectoryOptimizerState<T>& state,
    const std::vector<ocs2::vector_s_t<T>>& a,
    std::vector<ocs2::vector_s_t<T>>* tau) const {
  // Generalized forces aren't defined for the last timestep
  // TODO(vincekurtz): additional checks that q_t, v_t, tau_t are the right size
  // for the plant?
  assert(static_cast<int>(a.size()) == num_steps());
  assert(static_cast<int>(tau->size()) == num_steps());

#if defined(_OPENMP)
#pragma omp parallel for num_threads(params_.num_threads)
#endif
  for (int t = 0; t < num_steps(); ++t) {
    TrajectoryOptimizerWorkspace<T>& workspace = state.per_timestep_workspace[t];
    const Context<T>& context_tp = EvalPlantContext(state, t + 1);
    // All dynamics terms are treated implicitly, i.e.,
    // tau[t] = M(q[t+1]) * a[t] - k(q[t+1],v[t+1]) - fext[t+1]
    CalcInverseDynamicsSingleTimeStep(context_tp, a[t], &workspace, &tau->at(t));
  }
  //   std::cerr << "tau at 0: " << tau->at(0).transpose() << std::endl;
  //   std::cerr << "tau at num_steps(): " << tau->at(num_steps() - 1).transpose() << std::endl;
}

template <typename T>
void TrajectoryOptimizer<T>::CalcInverseDynamicsSingleTimeStep(const Context<T>& context,
    const ocs2::vector_s_t<T>& a,
    TrajectoryOptimizerWorkspace<T>* workspace,
    ocs2::vector_s_t<T>* tau) const {
  CalcContactForceContribution(context, &workspace->fext);

  // Inverse dynamics computes tau = M*a - k(q,v) - fext
  auto data = data_;
  *tau = pinocchio::rnea(model_, data, context.q_, context.v_, a, workspace->fext);
}

template <typename T>
void TrajectoryOptimizer<T>::CalcContactForceContribution(const Context<T>& context, pinocchio::container::aligned_vector<pinocchio::ForceTpl<T>>* fext) const {
  using std::abs;
  using std::exp;
  using std::log;
  using std::max;
  using std::pow;
  using std::sqrt;

  auto data = data_;

  // Compliant contact parameters
  const double k = params_.contact_stiffness;
  const double sigma = params_.smoothing_factor;

  // Compute the distance at which contact forces are zero: we don't need to do
  // any geometry queries beyond this distance
  const double eps = sqrt(std::numeric_limits<double>::epsilon());
  double threshold = -sigma * log(exp(eps / (sigma * k)) - 1.0);  // 0.210176

  pinocchio::forwardKinematics(model_, data, context.q_, context.v_);
  pinocchio::updateFramePlacements(model_, data);

  *fext = pinocchio::container::aligned_vector<pinocchio::ForceTpl<T>>(model_.njoints, pinocchio::ForceTpl<T>::Zero());
  for (const int frameIndex : footId_) {
    // Normal outwards from A.
    const ocs2::vector3_s_t<T> nhat = ocs2::vector3_s_t<T>(T(0), T(0), T(1.0));

    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const ocs2::vector_s_t<T> q = context.q_;
    const ocs2::vector_s_t<T> v = context.v_;

    const ocs2::vector3_s_t<T> pos = data.oMf[frameIndex].translation();
    const ocs2::vector3_s_t<T> vel = pinocchio::getFrameVelocity(model_, data, frameIndex, rf).linear();
    const T distance = nhat.dot(pos);

    // compute contact force through soft contact model
    if (distance < threshold) {
      ocs2::vector3_s_t<T> lambda_in_world;
      if (params_.which_contact_model == 0) {
        lambda_in_world = computeContactForceThroughSoftModel(distance, vel, nhat);
      } else if (params_.which_contact_model == 1) {
        lambda_in_world = computeContactForceThroughSpringDamperModel(distance, vel, nhat);
      }

      const auto jointIndex = model_.frames[frameIndex].parent;
      const ocs2::vector3_s_t<T> translationJointFrameToContactFrame = model_.frames[frameIndex].placement.translation();
      const ocs2::matrix3_s_t<T> rotationWorldFrameToJointFrame = data.oMi[jointIndex].rotation().transpose();
      const ocs2::vector3_s_t<T> contactForce = rotationWorldFrameToJointFrame * lambda_in_world;
      fext->at(jointIndex).linear() = contactForce;
      fext->at(jointIndex).angular() = translationJointFrameToContactFrame.cross(contactForce);
    }
  }
}

template <typename T>
const ocs2::vector3_s_t<T> TrajectoryOptimizer<T>::computeContactForceThroughSoftModel(const T distance,
    const ocs2::vector3_s_t<T> v,
    const ocs2::vector3_s_t<T> nhat) const {
  using std::abs;
  using std::exp;
  using std::log;
  using std::max;
  using std::pow;
  using std::sqrt;

  // Compliant contact parameters
  const double k = params_.contact_stiffness;
  const double sigma = params_.smoothing_factor;
  const double dissipation_velocity = params_.dissipation_velocity;

  // Friction parameters.
  const double vs = params_.stiction_velocity;     // Regularization.
  const double mu = params_.friction_coefficient;  // Coefficient of friction.

  // Split into normal and tangential components.
  const T vn = nhat.dot(v);
  const ocs2::vector3_s_t<T> vt = v - vn * nhat;

  // Normal dissipation follows a smoothed Hunt and Crossley model
  T dissipation_factor = 0.0;
  const T s = vn / dissipation_velocity;
  if (s < 0) {
    dissipation_factor = 1 - s;
  } else if (s < 2) {
    dissipation_factor = (s - 2) * (s - 2) / 4;
  }

  // (Compliant) force in the normal direction increases linearly at a rate
  // of k Newtons per meter, with some smoothing defined by sigma.
  T compliant_fn;
  const T exponent = -distance / sigma;
  if (exponent >= 37) {
    // If the exponent is going to be very large, replace with the
    // functional limit.
    // N.B. x = 37 is the first integer such that exp(x)+1 = exp(x) in
    // double precision.
    compliant_fn = -k * distance;
  } else {
    compliant_fn = sigma * k * log(1 + exp(exponent));
  }
  const T fn = compliant_fn * dissipation_factor;

  // Tangential frictional component.
  // N.B. This model is algebraically equivalent to:
  //  ft = -mu*fn*sigmoid(||vt||/vs)*vt/||vt||.
  // with the algebraic sigmoid function defined as sigmoid(x) =
  // x/sqrt(1+x^2). The algebraic simplification is performed to avoid
  // division by zero when vt = 0 (or loss of precision when close to zero).
  const ocs2::vector3_s_t<T> that_regularized = -vt / sqrt(vs * vs + vt.squaredNorm());
  const ocs2::vector3_s_t<T> ft_BC_W = that_regularized * mu * fn;

  // Total contact force on B at C, expressed in W.
  return nhat * fn + ft_BC_W;
}

template <typename T>
const ocs2::vector3_s_t<T> TrajectoryOptimizer<T>::computeContactForceThroughSpringDamperModel(const T distance,
    const ocs2::vector3_s_t<T> v,
    const ocs2::vector3_s_t<T> nhat) const {
  using std::abs;
  using std::exp;
  using std::log;
  using std::max;
  using std::pow;
  using std::sqrt;

  // Compliant contact parameters
  const double k = params_.k_spring;
  const double d = params_.d_damper;
  const double alpha = params_.damper_smooth;
  const double alpha_n = params_.spring_smooth;
  const double zOffset = params_.zOffset;

  // damper force
  ocs2::vector3_s_t<T> f_damper;
  f_damper = -d * v;                            // damper
  f_damper *= 1 / (1 + exp(distance * alpha));  //smoothing

  // spring force
  const T f_springer = k * exp(-alpha_n * (distance - zOffset));  //spring

  // Total contact force on B at C, expressed in W.
  return nhat * f_springer + f_damper;
}

template <typename T>
void TrajectoryOptimizer<T>::CalcInverseDynamicsPartials(const TrajectoryOptimizerState<T>& state, InverseDynamicsPartials<T>* id_partials) const {
  //   INSTRUMENT_FUNCTION("Computes dtau/dq.");
  using std::abs;
  using std::max;
  // Check that id_partials has been allocated correctly.
  assert(id_partials->size() == num_steps());

  // Get the trajectory data
  const std::vector<ocs2::vector_s_t<T>>& q = state.q();
  const std::vector<ocs2::vector_s_t<T>>& v = EvalV(state);
  const std::vector<ocs2::vector_s_t<T>>& a = EvalA(state);
  const std::vector<ocs2::vector_s_t<T>>& tau = EvalTau(state);

  // Get references to the partials that we'll be setting
  std::vector<ocs2::matrix_s_t<T>>& dtau_dqm = id_partials->dtau_dqm;
  std::vector<ocs2::matrix_s_t<T>>& dtau_dqt = id_partials->dtau_dqt;
  std::vector<ocs2::matrix_s_t<T>>& dtau_dqp = id_partials->dtau_dqp;

  // Get kinematic mapping matrices for each time step
  const std::vector<ocs2::matrix_s_t<T>>& Nplus = EvalNplus(state);

  // Allocate small perturbations to q, v, and a at each time step
  const double eps = sqrt(std::numeric_limits<double>::epsilon());
  std::vector<T> dq_is(num_steps() + 1);
  std::vector<T> dv_is(num_steps() + 1);
  std::vector<T> da_is(num_steps() + 1);

#if defined(_OPENMP)
#pragma omp parallel for num_threads(params_.num_threads)
#endif
  for (int t = 1; t <= num_steps(); ++t) {
    // N.B. A perturbation of qt propagates to tau[t-1], tau[t] and tau[t+1].
    // Therefore we compute one column of grad_tau at a time. That is, once the
    // loop on position indices i is over, we effectively computed the t-th
    // column of grad_tau.

    // N.B. we need a separate workspace for each timestep, otherwise threads
    // will fight over the same workspace
    TrajectoryOptimizerWorkspace<T>& workspace = state.per_timestep_workspace[t];

    // Get references to perturbed versions of q, v, tau, and a, at (t-1, t, t).
    // These are all of the quantities that change when we perturb q_t.
    ocs2::vector_s_t<T>& q_eps_t = workspace.q_size_tmp1;
    ocs2::vector_s_t<T>& v_eps_t = workspace.v_size_tmp1;
    ocs2::vector_s_t<T>& v_eps_tp = workspace.v_size_tmp2;
    ocs2::vector_s_t<T>& a_eps_tm = workspace.a_size_tmp1;
    ocs2::vector_s_t<T>& a_eps_t = workspace.a_size_tmp2;
    ocs2::vector_s_t<T>& tau_eps_tm = workspace.tau_size_tmp1;
    ocs2::vector_s_t<T>& tau_eps_t = workspace.tau_size_tmp2;

    // Mass matrix, for analytical computation of dtau[t+1]/dq[t]
    ocs2::matrix_s_t<T>& M = workspace.mass_matrix_size_tmp;

    // Small perturbations
    T& dq_i = dq_is[t];
    T& dv_i = dv_is[t];
    T& da_i = da_is[t];

    // Set perturbed versions of variables
    q_eps_t = q[t];
    v_eps_t = v[t];
    if (t < num_steps()) {
      // v[num_steps + 1] is not defined
      v_eps_tp = v[t + 1];
      // a[num_steps] is not defined
      a_eps_t = a[t];
    }
    a_eps_tm = a[t - 1];

    // Get a context for this time step
    Context<T>& context_t = GetMutablePlantContext(state, t);

    for (int i = 0; i < model_.nq; ++i) {
      // Determine perturbation sizes to avoid losing precision to floating
      // point error
      dq_i = eps * max(1.0, abs(q_eps_t(i)));

      // Make dqt_i exactly representable to minimize floating point error
      const T temp = q_eps_t(i) + dq_i;
      dq_i = temp - q_eps_t(i);

      dv_i = dq_i / time_step();
      da_i = dv_i / time_step();

      // Perturb q_t[i], v_t[i], and a_t[i]
      q_eps_t(i) += dq_i;

      v_eps_t += dv_i * Nplus[t].col(i);
      a_eps_tm += da_i * Nplus[t].col(i);
      if (t < num_steps()) {
        v_eps_tp -= dv_i * Nplus[t + 1].col(i);
        a_eps_t -= da_i * (Nplus[t + 1].col(i) + Nplus[t].col(i));
      }

      // Compute perturbed tau(q) and calculate the nonzero entries of dtau/dq
      // via finite differencing

      // tau[t-1] = ID(q[t], v[t], a[t-1])
      context_t.q_ = q_eps_t;
      context_t.v_ = v_eps_t;
      CalcInverseDynamicsSingleTimeStep(context_t, a_eps_tm, &workspace, &tau_eps_tm);
      dtau_dqp[t - 1].col(i) = (tau_eps_tm - tau[t - 1]) / dq_i;

      // tau[t] = ID(q[t+1], v[t+1], a[t])
      if (t < num_steps()) {
        context_t.q_ = q[t + 1];
        context_t.v_ = v_eps_tp;
        CalcInverseDynamicsSingleTimeStep(context_t, a_eps_t, &workspace, &tau_eps_t);
        dtau_dqt[t].col(i) = (tau_eps_t - tau[t]) / dq_i;
      }

      // Unperturb q_t[i], v_t[i], and a_t[i]
      q_eps_t = q[t];
      v_eps_t = v[t];
      a_eps_tm = a[t - 1];
      if (t < num_steps()) {
        v_eps_tp = v[t + 1];
        a_eps_t = a[t];
      }
    }

    // tau[t+1] = ID(q[t+2], v[t+2], a[t+1])
    // N.B. Since tau[t+1] depends on q only through a, we can compute
    // dtau_dqm[t+1] analytically rather than relying on column-wise forward
    // differences
    if (t < num_steps() - 1) {
      //   context_t.q_ = q[t + 2];
      //   context_t.v_ = v[t + 2];
      auto data = data_;
      M = pinocchio::crba(model_, data, q[t + 2]);
      M.template triangularView<Eigen::StrictlyLower>() = M.transpose().template triangularView<Eigen::StrictlyLower>();
      dtau_dqm[t + 1] = 1 / time_step() / time_step() * M * Nplus[t + 1];
    }
  }
}

template <typename T>
void TrajectoryOptimizer<T>::CalcVelocityPartials(const TrajectoryOptimizerState<T>& state, VelocityPartials<T>* v_partials) const {
  const std::vector<ocs2::matrix_s_t<T>>& Nplus = EvalNplus(state);
  for (int t = 0; t <= num_steps(); ++t) {
    v_partials->dvt_dqt[t] = 1 / time_step() * Nplus[t];
    if (t > 0) {
      v_partials->dvt_dqm[t] = -1 / time_step() * Nplus[t];
    }
  }
}

}  // namespace optimizer
}  // namespace idto