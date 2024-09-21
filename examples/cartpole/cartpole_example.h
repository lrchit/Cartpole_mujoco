
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
    std::shared_ptr<Cartpole_Dynamics> cartpole_dynamics = std::make_shared<Cartpole_Dynamics>(config);
    std::shared_ptr<Cartpole_Cost> cartpole_cost = std::make_shared<Cartpole_Cost>(config);
    std::unique_ptr<iLQR_Solver> iLQR = std::make_unique<iLQR_Solver>(config, cartpole_dynamics, cartpole_cost);
    mpc.reset(new mpc_controller(config, std::move(iLQR)));
  }

  ~Cartpole_Example() {}

  private:
  virtual void computeInput(mjData* d) override {
    xcur << d->sensordata[0], d->sensordata[1], d->sensordata[2], d->sensordata[3];
    for (int k = 0; k < Nt; ++k) {
      x_goal[k].setZero(nx);
    }
    mpc->resetProblem(xcur, x_goal);

    d->ctrl[0] = fmin(fmax(mpc->getCommand().value(), -100), 100);
    d->ctrl[1] = 0;
  }
};