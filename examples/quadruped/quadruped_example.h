
#pragma once

// #include <quadruped_cost.h>
// #include <quadruped_dynamics.h>
#include <state_estimator.h>

#include <mpc.h>

class Quadruped_Example : public Example {
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Quadruped_Example(std::string yaml_name) : Example(yaml_name) {
    YAML::Node config = YAML::LoadFile(yaml_name);

    nq = config["nq"].as<int>();
    nv = config["nv"].as<int>();
    if (nx != nq + nv) {
      throw std::invalid_argument("nx != nq + nv");
    }

    // std::shared_ptr<Quadruped_Dynamics> quadruped_dynamics = std::make_shared<Quadruped_Dynamics>(config);
    // std::shared_ptr<Quadruped_Cost> quadruped_cost = std::make_shared<Quadruped_Cost>(config);
    // std::unique_ptr<iLQR_Solver> iLQR = std::make_unique<iLQR_Solver>(config, quadruped_dynamics, quadruped_cost);
    // mpc.reset(new MpcController(config, std::move(iLQR)));
    // stateEstimator.reset(new StateEstimator(config));
  }

  ~Quadruped_Example() {}

  virtual void load_initial_state(mjData* d) override {
    YAML::Node config = YAML::LoadFile(yaml_name_);

    // Initial state
    d->qpos[7] = config["initial_state"]["p1"].as<double>();
    d->qpos[8] = config["initial_state"]["p2"].as<double>();
    d->qpos[9] = config["initial_state"]["p3"].as<double>();
    d->qpos[10] = config["initial_state"]["p4"].as<double>();
    d->qpos[11] = config["initial_state"]["p5"].as<double>();
    d->qpos[12] = config["initial_state"]["p6"].as<double>();
    d->qpos[13] = config["initial_state"]["p7"].as<double>();
    d->qpos[14] = config["initial_state"]["p8"].as<double>();
    d->qpos[15] = config["initial_state"]["p9"].as<double>();
    d->qpos[16] = config["initial_state"]["p10"].as<double>();
    d->qpos[17] = config["initial_state"]["p11"].as<double>();
    d->qpos[18] = config["initial_state"]["p12"].as<double>();
  }

  virtual void computeInput(mjData* d) override {
    stateEstimator->callStateEstimator(d->sensordata);
    xcur.head(nq) = stateEstimator->getGeneralizedCoordinates();
    xcur.tail(nv) = stateEstimator->getGeneralizedVelocities();
    for (int k = 0; k < Nt; ++k) {
      x_goal[k].setZero(nx);
    }
    mpc->resetProblem(xcur, x_goal);

    // const ocs2::vector_t input = mpc->getCommand();
    const ocs2::vector_t input = ocs2::vector_t::Zero(nu);
    for (int i = 0; i < nu; ++i) {
      d->ctrl[0] = fmin(fmax(input[i], -30), 30);
    }
  }

  private:
  int nq;
  int nv;

  std::unique_ptr<StateEstimator> stateEstimator;
};