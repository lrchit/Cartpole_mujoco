
#pragma once

#include "iLQR.h"

#include <cartpole_cost.h>
#include <cartpole_dynamics.h>
#include <cartpole_constraint.h>
#include <mpc.h>
#include <iLQR.h>
#include <direct_multiple_shooting.h>

class Cartpole_Example : public Example {
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Cartpole_Example(std::string yaml_name) : Example(yaml_name) {
    YAML::Node config = YAML::LoadFile(yaml_name);

    std::shared_ptr<Cartpole_Dynamics> cartpole_dynamics = std::make_shared<Cartpole_Dynamics>(config);
    std::shared_ptr<Cartpole_Cost> cartpole_cost = std::make_shared<Cartpole_Cost>(config);
    std::shared_ptr<Cartpole_Constraint> cartpole_constraint = std::make_shared<Cartpole_Constraint>(config);

    // mpc setting
    std::vector<double> K = config["dms"]["K"].as<std::vector<double>>();
    ocs2::matrix_t K_(nu, nx);
    for (int i = 0; i < nx; ++i) {
      K_(0, i) = K[i];
    }
    const int use_which_solver = config["use_which_solver"].as<int>();
    switch (use_which_solver) {
      case 1:  // iLQR
        solver.reset(new iLQR_Solver(config, cartpole_dynamics, cartpole_cost, ocs2::matrix_t::Zero(1, 4)));
        mpc.reset(new MpcController(config, std::move(solver)));
        break;
      case 2:  // direct multiple shooting
        solver.reset(new DirectMultipleShooting(config, K_, cartpole_dynamics, cartpole_cost, cartpole_constraint));
        mpc.reset(new MpcController(config, std::move(solver)));
        break;
      default:
        throw std::runtime_error("Choose from 1 for iLQR, 2 for dms");
    }
  }

  ~Cartpole_Example() {}

  virtual void load_initial_state(mjData* d) override {
    YAML::Node config = YAML::LoadFile(yaml_name_);
    // Initial state
    d->qpos[0] = config["initial_state"]["pos"].as<double>();
    d->qpos[1] = config["initial_state"]["theta"].as<double>();
    d->qvel[0] = config["initial_state"]["vel"].as<double>();
    d->qvel[1] = config["initial_state"]["omega"].as<double>();
  }

  virtual void computeInput(mjData* d) override {
    xcur << d->sensordata[0], d->sensordata[1], d->sensordata[2], d->sensordata[3];
    angleNormalize(xcur[1]);
    std::vector<ocs2::vector_t> x_ref(Nt, xtarget);
    x_ref[0] = xcur;
    mpc->resetProblem(xcur, x_ref);

    const ocs2::vector_t command = mpc->getCommand();
    for (int i = 0; i < nu; ++i) {
      d->ctrl[i] = fmin(fmax(command[i], -100), 100);
    }
  }

  private:
  // normalize angle
  void angleNormalize(double& angle) {
    while (angle > std::numbers::pi)
      angle -= 2 * std::numbers::pi;
    while (angle < -std::numbers::pi)
      angle += 2 * std::numbers::pi;
  }

  std::unique_ptr<ControllerBase> solver;
};