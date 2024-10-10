
#pragma once

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
#include <state_estimator.h>

#include <mpc.h>
#include <idto.h>
#include <quadruped_cost.h>
#include <quadruped_dynamics.h>
#include <direct_multiple_shooting.h>

class Quadruped_Example : public Example {
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Quadruped_Example(std::string yaml_name) : Example(yaml_name) {
    YAML::Node config = YAML::LoadFile(yaml_name);
    nq = config["nq"].as<int>();
    nv = config["nv"].as<int>();
    nu = config["nu"].as<int>();
    horizontal_time = (Nt - 1) * config["mpc"]["dt"].as<double>();
    demo_vel_x = config["demo_vel_x"].as<double>();

    std::vector<double> q_init = config["q_init"].as<std::vector<double>>();
    std::vector<double> v_init = config["v_init"].as<std::vector<double>>();
    for (int i = 0; i < nq; ++i) {
      xtarget[i] = q_init[i];
      xtarget[nq + i] = v_init[i];
    }

    timer.reset();

    const std::string urdfFile = "../models/quadruped/urdf/a1.urdf";
    const std::vector<std::string> jointNames{"FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"};
    const pinocchio::ModelTpl<ocs2::scalar_t> model = createPinocchioModel(urdfFile, jointNames);
    pinocchio::DataTpl<ocs2::scalar_t> data(model);
    // const std::vector<pinocchio::FrameIndex> footId{model.getFrameId("FL_foot"), model.getFrameId("FR_foot"), model.getFrameId("RL_foot"),
    //     model.getFrameId("RR_foot"), model.getFrameId("FL_hip"), model.getFrameId("FR_hip"), model.getFrameId("RL_hip"), model.getFrameId("RR_hip"),
    //     model.getFrameId("FL_thigh"), model.getFrameId("FR_thigh"), model.getFrameId("RL_thigh"), model.getFrameId("RR_thigh")};
    const std::vector<pinocchio::FrameIndex> footId{
        model.getFrameId("FL_foot"), model.getFrameId("FR_foot"), model.getFrameId("RL_foot"), model.getFrameId("RR_foot")};

    stateEstimator.reset(new StateEstimator(config, model, footId));

    // setting mpc
    ocs2::matrix_t K(nu, nx);
    const int use_which_solver = config["use_which_solver"].as<int>();
    if (use_which_solver == 1) {
      std::vector<double> Kp = config["idto"]["Kp"].as<std::vector<double>>();
      std::vector<double> Kd = config["idto"]["Kd"].as<std::vector<double>>();
      for (int i = 0; i < nu; ++i) {
        K.middleCols(6, nu).diagonal()[i] = Kp[i];
        K.middleCols(6 + nq, nu).diagonal()[i] = Kd[i];
      }
      std::unique_ptr<idto::optimizer::idto> inverseDynamicsController = std::make_unique<idto::optimizer::idto>(config, K, model, data, footId);
      mpc.reset(new MpcController(config, std::move(inverseDynamicsController)));
    } else if (use_which_solver == 2) {
      std::vector<double> Kp = config["dms"]["Kp"].as<std::vector<double>>();
      std::vector<double> Kd = config["dms"]["Kd"].as<std::vector<double>>();
      for (int i = 0; i < nu; ++i) {
        K.middleCols(6, nu).diagonal()[i] = Kp[i];
        K.middleCols(6 + nq, nu).diagonal()[i] = Kd[i];
      }
      std::shared_ptr<Quadruped_Dynamics> quadruped_dynamics = std::make_shared<Quadruped_Dynamics>(config, model, footId);
      std::shared_ptr<Quadruped_Cost> quadruped_cost = std::make_shared<Quadruped_Cost>(config, model, footId);
      std::unique_ptr<DirectMultipleShooting> dmsController = std::make_unique<DirectMultipleShooting>(config, K, quadruped_dynamics, quadruped_cost);
      mpc.reset(new MpcController(config, std::move(dmsController)));
    } else {
      throw std::runtime_error("Choose from 1 for idto, 2 for dms");
    }
  }

  ~Quadruped_Example() {}

  virtual void load_initial_state(mjData* d) override {
    YAML::Node config = YAML::LoadFile(yaml_name_);

    // Initial state
    std::vector<double> q_init = config["initial_state"].as<std::vector<double>>();
    // d->qpos[3] = 0;
    // d->qpos[4] = 1;
    // d->qpos[5] = 0;
    // d->qpos[6] = 0;
    for (int i = 0; i < nq - 7; ++i) {
      d->qpos[i + 7] = q_init[i];
    }

    // initialize sensor data for estimator
    stateEstimator->setSensorData(d->sensordata);
  }

  virtual void computeInput(mjData* d) override {
    xcur.head(nq) = stateEstimator->getGeneralizedCoordinates();
    xcur.tail(nv) = stateEstimator->getGeneralizedVelocities();

    // if (timer.getCurrentTime() > 4) {
    //   xtarget[4] = 0.8;
    // }
    if (timer.getCurrentTime() > 4) {
      xtarget[0] = xcur[0] + demo_vel_x * horizontal_time;
      // xtarget[1] = xcur[1];
      xtarget[nq + 0] = demo_vel_x;
      xtarget[4] = 0.0;
    }
    std::vector<ocs2::vector_t> x_ref = makeLinearInterpolation(xcur, xtarget);
    // std::vector<ocs2::vector_t> x_ref(Nt, xtarget);
    mpc->resetProblem(xcur, x_ref);

    const ocs2::vector_t input = mpc->getCommand();
    for (int i = 0; i < nu; ++i) {
      d->ctrl[i] = fmin(fmax(input[i], -30), 30);
    }
  }

  private:
  std::vector<ocs2::vector_t> makeLinearInterpolation(const ocs2::vector_t& start, const ocs2::vector_t& end) {
    std::vector<ocs2::vector_t> result;
    double lambda = 0;
    // for (int i = 0; i < Nt; ++i) {
    //   lambda = i / (Nt - 1.0);
    //   result.push_back((1 - lambda) * start + lambda * end);
    // }
    for (int i = 0; i < Nt; ++i) {
      lambda = i / (Nt - 1.0);
      result.push_back((ocs2::vector_t(nq + nv) << (1 - lambda) * start.head(nq) + lambda * end.head(nq), end.tail(nv)).finished());
    }
    result[0].tail(nv) = start.tail(nv);
    return result;
  }

  int nq;
  int nv;
  int nu;
  double horizontal_time;
  double demo_vel_x;

  Timer timer;
  std::unique_ptr<StateEstimator> stateEstimator;
};