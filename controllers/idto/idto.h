#pragma once

#include <controller.h>

#include <Types.h>

#include <trajectory_optimizer.h>
#include <problem_definition.h>
#include <solver_parameters.h>

#include <yaml-cpp/yaml.h>

namespace idto {
namespace optimizer {

class idto : public ControllerBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  public:
  idto(YAML::Node config,
      ocs2::matrix_t K,
      const pinocchio::ModelTpl<double>& model,
      const pinocchio::DataTpl<double>& data,
      const std::vector<pinocchio::FrameIndex> footId)
      : probParam_(ProblemDefinition(config)), solverParam_(SolverParameters(config)), footId_(footId), model_(model), data_(data) {
    nq_ = model.nq;
    nv_ = model.nv;
    nu_ = nv_ - 6;

    K_ = K;

    const double time_step = config["mpc"]["dt"].as<double>();
    solver_max_iter_ = config["idto"]["mpc_iters"].as<int>();

    // initialize
    q_guess_ = std::vector<ocs2::vector_t>(probParam_.num_steps + 1, ocs2::vector_t::Zero(nq_));
    xtraj = std::vector<ocs2::vector_t>(probParam_.num_steps + 1, ocs2::vector_t::Zero(nq_ + nv_));
    utraj = std::vector<ocs2::vector_t>(probParam_.num_steps, ocs2::vector_t::Zero(nu_));
    xref = std::vector<ocs2::vector_t>(probParam_.num_steps + 1, ocs2::vector_t::Zero(nq_ + nv_));
    solution_.q = std::vector<ocs2::vector_t>(probParam_.num_steps + 1, ocs2::vector_t::Zero(nq_));
    solution_.v = std::vector<ocs2::vector_t>(probParam_.num_steps + 1, ocs2::vector_t::Zero(nv_));
    solution_.tau = std::vector<ocs2::vector_t>(probParam_.num_steps, ocs2::vector_t::Zero(nu_));

    // setup solver
    optimizer_.reset(new TrajectoryOptimizer(model, data, probParam_, time_step, footId, solverParam_));
    warm_start_.reset(new WarmStart(optimizer_->num_steps(), model, data, optimizer_->num_equality_constraints(), solution_.q, solverParam_.Delta0));
  }

  ~idto() = default;

  virtual void launch_controller(const ocs2::vector_t& xcur, const std::vector<ocs2::vector_t>& x_ref) override {
    if (solverFirstRun_) {
      optimizer_->params().max_iterations = 100;  // calc until converge for initial guess
      for (int k = 0; k < probParam_.num_steps + 1; ++k) {
        solution_.q[k] = x_ref[k].head(nq_);
      }
      solverFirstRun_ = false;
    } else {
      optimizer_->params().max_iterations = solver_max_iter_;  // set to max iter for mpc
    }
    // update current state
    const ocs2::vector_t q0 = xcur.head(nq_);
    const ocs2::vector_t v0 = xcur.tail(nv_);

    q_guess_[0] = q0;  // guess must be consistent with the initial condition
    for (int k = 1; k < probParam_.num_steps; ++k) {
      q_guess_[k] = solution_.q[1 + k];
    }
    q_guess_[probParam_.num_steps] = solution_.q[probParam_.num_steps];
    warm_start_->set_q(q_guess_);

    // simulate an action standing without moving, so q_norm and v_norm are constant, but should change in practice.
    for (int k = 0; k < probParam_.num_steps + 1; ++k) {
      probParam_.q_nom[k] = x_ref[k].head(nq_);
      probParam_.v_nom[k] = x_ref[k].tail(nv_);
    }
    optimizer_->UpdateNominalTrajectory(probParam_.q_nom, probParam_.v_nom);

    // Solve the trajectory optimization problem from the new initial condition
    optimizer_->ResetInitialConditions(q0, v0);

    // solve
    optimizer_->SolveFromWarmStart(warm_start_.get(), &solution_, &stats_);

    // update solution to xtraj and utraj
    updateSolution();
  }

  virtual ocs2::matrix_t getFeedBackMatrix() override { return K_; }

  private:
  void updateSolution() {
    for (int k = 0; k < probParam_.num_steps; ++k) {
      xtraj[k].head(nq_) = solution_.q[k];
      xtraj[k].tail(nv_) = solution_.v[k];
      utraj[k] = solution_.tau[k].tail(nu_);
    }
    xtraj[probParam_.num_steps].head(nq_) = solution_.q[probParam_.num_steps];
    xtraj[probParam_.num_steps].tail(nv_) = solution_.v[probParam_.num_steps];

    // for (int k = 0; k < probParam_.num_steps; ++k) {
    //   std::cerr << "############ " << k << " ############" << std::endl;
    //   pinocchio::forwardKinematics(model_, data_, solution_.q[k]);
    //   pinocchio::updateFramePlacements(model_, data_);
    //   std::cerr << "q = " << solution_.q[k].head(6).transpose() << std::endl;
    //   std::cerr << "foot = " << data_.oMf[footId_[0]].translation().transpose() << std::endl;
    //   std::cerr << "foot = " << data_.oMf[footId_[1]].translation().transpose() << std::endl;
    //   std::cerr << "foot = " << data_.oMf[footId_[2]].translation().transpose() << std::endl;
    //   std::cerr << "foot = " << data_.oMf[footId_[3]].translation().transpose() << std::endl;
    //   std::cerr << "v = " << solution_.v[k].transpose() << std::endl;
    //   std::cerr << "tau = " << solution_.tau[k].transpose() << std::endl;
    // }
  }

  int nq_;
  int nv_;
  int nu_;
  bool solverFirstRun_ = true;
  int solver_max_iter_;

  std::vector<pinocchio::FrameIndex> footId_;
  pinocchio::ModelTpl<ocs2::scalar_t> model_;
  pinocchio::DataTpl<ocs2::scalar_t> data_;

  ocs2::matrix_t K_;
  std::vector<ocs2::vector_t> q_guess_;

  ProblemDefinition probParam_;
  SolverParameters solverParam_;
  TrajectoryOptimizerStats<double> stats_;
  TrajectoryOptimizerSolution<double> solution_;

  std::unique_ptr<TrajectoryOptimizer<double>> optimizer_;
  std::unique_ptr<WarmStart> warm_start_;
};

}  // namespace optimizer
}  // namespace idto