
#pragma once

#include "iLQR.h"

#include <cartpole_cost.h>
#include <cartpole_dynamics.h>
#include <mpc.h>

class Cartpole_Example : public Example {
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Cartpole_Example(std::string yaml_name) : Example(yaml_name) {
    YAML::Node config = YAML::LoadFile(yaml_name);
    const ocs2::matrix_t Kguess = ocs2::matrix_t::Zero(1, 4);

    std::shared_ptr<Cartpole_Dynamics> cartpole_dynamics = std::make_shared<Cartpole_Dynamics>(config);
    std::shared_ptr<Cartpole_Cost> cartpole_cost = std::make_shared<Cartpole_Cost>(config);
    std::unique_ptr<iLQR_Solver> iLQR = std::make_unique<iLQR_Solver>(config, cartpole_dynamics, cartpole_cost, Kguess);
    mpc.reset(new MpcController(config, std::move(iLQR)));
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

    mpc->resetProblem(xcur, xref);

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
};