
#pragma once

#include <utils.h>
#include <quadruped_cost.h>
#include <quadruped_dynamics.h>
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

    xref[0][0] = config["norminal_state"]["p1"].as<double>();
    xref[0][1] = config["norminal_state"]["p2"].as<double>();
    xref[0][2] = config["norminal_state"]["p3"].as<double>();
    xref[0][3] = config["norminal_state"]["p4"].as<double>();
    xref[0][4] = config["norminal_state"]["p5"].as<double>();
    xref[0][5] = config["norminal_state"]["p6"].as<double>();
    xref[0][6] = config["norminal_state"]["p7"].as<double>();
    xref[0][7] = config["norminal_state"]["p8"].as<double>();
    xref[0][8] = config["norminal_state"]["p9"].as<double>();
    xref[0][9] = config["norminal_state"]["p10"].as<double>();
    xref[0][10] = config["norminal_state"]["p11"].as<double>();
    xref[0][11] = config["norminal_state"]["p12"].as<double>();
    xref[0][12] = config["norminal_state"]["p13"].as<double>();
    xref[0][13] = config["norminal_state"]["p14"].as<double>();
    xref[0][14] = config["norminal_state"]["p15"].as<double>();
    xref[0][15] = config["norminal_state"]["p16"].as<double>();
    xref[0][16] = config["norminal_state"]["p17"].as<double>();
    xref[0][17] = config["norminal_state"]["p18"].as<double>();
    for (int k = 1; k < Nt; ++k) {
      xref[k] = xref[0];
    }

    ocs2::matrix_t Kguess = ocs2::matrix_t::Zero(12, 36);
    for (int leg = 0; leg < 4; ++leg) {
      Kguess(3 * leg + 0, 6 + 3 * leg + 0) = config["Kguess"]["kp"]["abad"].as<double>();
      Kguess(3 * leg + 1, 6 + 3 * leg + 1) = config["Kguess"]["kp"]["hip"].as<double>();
      Kguess(3 * leg + 2, 6 + 3 * leg + 2) = config["Kguess"]["kp"]["knee"].as<double>();
      Kguess(3 * leg + 0, 24 + 3 * leg + 0) = config["Kguess"]["kd"]["abad"].as<double>();
      Kguess(3 * leg + 1, 24 + 3 * leg + 1) = config["Kguess"]["kd"]["hip"].as<double>();
      Kguess(3 * leg + 2, 24 + 3 * leg + 2) = config["Kguess"]["kd"]["knee"].as<double>();
    }
    std::cerr << Kguess << std::endl;

    // pinocchio model
    const std::string urdfFile = "../models/quadruped/urdf/a1.urdf";
    const std::vector<std::string> footName{"FL_foot", "FR_foot", "RL_foot", "RR_foot"};
    const std::vector<std::string> jointNames{"FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"};
    const pinocchio::ModelTpl<ocs2::scalar_t> model = createPinocchioModel(urdfFile, jointNames);
    pinocchio::DataTpl<ocs2::scalar_t> data = pinocchio::DataTpl<ocs2::scalar_t>(model);
    std::vector<pinocchio::FrameIndex> footId;
    for (int leg = 0; leg < 4; ++leg) {
      footId.push_back(model.getFrameId(footName[leg]));
    }

    stateEstimator.reset(new StateEstimator(config));
    std::shared_ptr<Quadruped_Dynamics> quadruped_dynamics = std::make_shared<Quadruped_Dynamics>(config, model, footId);
    std::shared_ptr<Quadruped_Cost> quadruped_cost = std::make_shared<Quadruped_Cost>(config, model, footId);
    std::unique_ptr<iLQR_Solver> iLQR = std::make_unique<iLQR_Solver>(config, quadruped_dynamics, quadruped_cost, Kguess);
    mpc.reset(new MpcController(config, std::move(iLQR)));
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
    xcur.head(nq) << 0, 0, 0.3, 0, 0, 0, -0., 0.72, -1.44, 0., 0.72, -1.44, -0., 0.72, -1.44, 0., 0.72, -1.44;
    xcur.tail(nv) = stateEstimator->getGeneralizedVelocities();

    mpc->resetProblem(xcur, xref);
    // std::cerr << "input = " << mpc->getCommand().transpose() << std::endl;

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