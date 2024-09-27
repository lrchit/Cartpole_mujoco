

#include <pinocchio/fwd.hpp>  // always include it before any other header

#include <pinocchio/parsers/urdf.hpp>
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/parsers/sample-models.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/spatial/explog.hpp"
#include <pinocchio/algorithm/center-of-mass.hpp>

#include <utils.h>

#include <trajectory_optimizer_workspace.h>
#include <trajectory_optimizer_state.h>
#include <trajectory_optimizer.h>
#include <velocity_partials.h>
#include <inverse_dynamics_partials.h>
#include <problem_definition.h>
#include <penta_diagonal_matrix.h>
#include <penta_diagonal_solver.h>
#include <solver_parameters.h>

#include <yaml-cpp/yaml.h>

std::vector<ocs2::vector_t> MakeLinearInterpolation(const ocs2::vector_t& start, const ocs2::vector_t& end, int N) {
  std::vector<ocs2::vector_t> result;
  double lambda = 0;
  for (int i = 0; i < N; ++i) {
    lambda = i / (N - 1.0);
    result.push_back((1 - lambda) * start + lambda * end);
  }
  return result;
}

int main() {
  const std::string urdfFile = "../models/quadruped/urdf/a1.urdf";
  const std::vector<std::string> jointNames{"FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
      "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"};
  const pinocchio::ModelTpl<ocs2::scalar_t> model = createPinocchioModel(urdfFile, jointNames);
  pinocchio::DataTpl<ocs2::scalar_t> data(model);
  const std::vector<pinocchio::FrameIndex> footId{
      model.getFrameId("FL_foot"), model.getFrameId("FR_foot"), model.getFrameId("RL_foot"), model.getFrameId("RR_foot")};

  YAML::Node config = YAML::LoadFile("../test/config/optimizer.yaml");
  const int horizon = config["horizon"].as<int>();
  const double time_step = config["dt"].as<double>();

  idto::optimizer::TrajectoryOptimizerWorkspace work_space = idto::optimizer::TrajectoryOptimizerWorkspace(horizon, model, data);
  idto::optimizer::VelocityPartials vel_partials = idto::optimizer::VelocityPartials<double>(horizon, model.nv, model.nq);
  idto::optimizer::InverseDynamicsPartials invdyn_partials = idto::optimizer::InverseDynamicsPartials<double>(horizon, model.nv, model.nq);
  idto::optimizer::ProblemDefinition probParam = idto::optimizer::ProblemDefinition(config);
  idto::optimizer::SolverParameters solverParam = idto::optimizer::SolverParameters(config);
  idto::optimizer::TrajectoryOptimizerState trajOptState =
      idto::optimizer::TrajectoryOptimizerState(horizon, model, data, work_space.num_vars_by_num_eq_cons_tmp.cols());

  idto::optimizer::internal::PentaDiagonalMatrix pentaDiagMatrix = idto::optimizer::internal::PentaDiagonalMatrix<double>(horizon, model.nv, true);
  idto::optimizer::internal::PentaDiagonalFactorization pentaDiagFactor = idto::optimizer::internal::PentaDiagonalFactorization(pentaDiagMatrix);

  std::vector<ocs2::vector_t> q_guess = MakeLinearInterpolation(probParam.q_init, probParam.q_init, probParam.num_steps + 1);

  // initial solution
  idto::optimizer::TrajectoryOptimizer initialOptimizer = idto::optimizer::TrajectoryOptimizer(model, data, probParam, time_step, footId, solverParam);
  idto::optimizer::TrajectoryOptimizerSolution<double> initialSolution;
  idto::optimizer::TrajectoryOptimizerStats<double> initialStats;
  idto::optimizer::ConvergenceReason reason;

  idto::optimizer::SolverFlag status = initialOptimizer.Solve(q_guess, &initialSolution, &initialStats, &reason);

  std::cerr << "############ initial solution ############" << std::endl;
  pinocchio::forwardKinematics(model, data, initialSolution.q[1]);
  pinocchio::updateFramePlacements(model, data);
  std::cerr << "q = " << initialSolution.q[1].head(6).transpose() << std::endl;
  std::cerr << "foot = " << data.oMf[footId[0]].translation().transpose() << std::endl;
  std::cerr << "foot = " << data.oMf[footId[1]].translation().transpose() << std::endl;
  std::cerr << "foot = " << data.oMf[footId[2]].translation().transpose() << std::endl;
  std::cerr << "foot = " << data.oMf[footId[3]].translation().transpose() << std::endl;
  std::cerr << "v = " << initialSolution.v[1].transpose() << std::endl;
  std::cerr << "tau = " << initialSolution.tau[1].transpose() << std::endl;

  // mpc test
  solverParam.max_iterations = config["mpc_iters"].as<int>();
  idto::optimizer::TrajectoryOptimizer mpcOptimizer = idto::optimizer::TrajectoryOptimizer(model, data, probParam, time_step, footId, solverParam);
  idto::optimizer::WarmStart warm_start =
      idto::optimizer::WarmStart(mpcOptimizer.num_steps(), model, data, mpcOptimizer.num_equality_constraints(), initialSolution.q, solverParam.Delta0);

  idto::optimizer::TrajectoryOptimizerStats<double> stats;
  idto::optimizer::TrajectoryOptimizerSolution<double> solution = initialSolution;

  const int mpc_n_times = 100;
  for (int iter = 0; iter < mpc_n_times; ++iter) {
    std::cerr << "############ iter " << iter << " ############" << std::endl;
    std::chrono::time_point<std::chrono::system_clock> t_start = std::chrono::system_clock::now();

    // simulate a time step right equals to mpc replan time step
    const ocs2::vector_t q0 = solution.q[1];
    const ocs2::vector_t v0 = solution.v[1];

    // Set the initial guess from the stored solution
    std::vector<ocs2::vector_t> q_guess(horizon + 1, ocs2::vector_t(model.nq));
    q_guess[0] = q0;  // guess must be consistent with the initial condition
    for (int k = 1; k < horizon; ++k) {
      q_guess[k] = solution.q[1 + k];
    }
    q_guess[horizon] = solution.q[horizon];
    warm_start.set_q(q_guess);

    // simulate an action standing without moving, so q_norm and v_norm are constant, but should change in practice.
    mpcOptimizer.UpdateNominalTrajectory(probParam.q_nom, probParam.v_nom);

    // Solve the trajectory optimization problem from the new initial condition
    mpcOptimizer.ResetInitialConditions(q0, v0);

    mpcOptimizer.SolveFromWarmStart(&warm_start, &solution, &stats);
    std::chrono::time_point<std::chrono::system_clock> t_end = std::chrono::system_clock::now();
    double time_record = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    std::cout << "mpc_solve_time: " << time_record / 1000 << "\n";

    pinocchio::forwardKinematics(model, data, solution.q[0]);
    pinocchio::updateFramePlacements(model, data);
    std::cerr << "q = " << solution.q[0].head(6).transpose() << std::endl;
    std::cerr << "foot = " << data.oMf[footId[0]].translation().transpose() << std::endl;
    std::cerr << "foot = " << data.oMf[footId[1]].translation().transpose() << std::endl;
    std::cerr << "foot = " << data.oMf[footId[2]].translation().transpose() << std::endl;
    std::cerr << "foot = " << data.oMf[footId[3]].translation().transpose() << std::endl;
    // std::cerr << "v = " << solution.v[0].transpose() << std::endl;
    // std::cerr << "tau = " << solution.tau[0].transpose() << std::endl;
  }

  return 0;
}