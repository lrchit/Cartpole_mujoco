
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

#include <mutex>
#include <thread>

#include <Types.h>

#include <RotationTransforms.h>

#include <interpolation.h>

#include <yaml-cpp/yaml.h>

class StateEstimator {
  public:
  StateEstimator(YAML::Node config, const pinocchio::ModelTpl<ocs2::scalar_t> model, std::vector<pinocchio::FrameIndex> footId);
  ~StateEstimator() {
    stop_thread_ = true;  // signal the thread to stop
    if (estimator_thread_.joinable()) {
      estimator_thread_.join();
    }
  }

  void callStateEstimator();

  void setSensorData(double* sensordata) { sensorData_ = sensordata; }

  ocs2::vector_t getGeneralizedCoordinates() { return q_; }
  ocs2::vector_t getGeneralizedVelocities() { return qd_; }

  private:
  void cheaterComputeState();
  void linearKalmanFilterComputeState();

  private:
  int nq_;
  int nv_;
  bool use_cheater_mode_;
  ocs2::vector_t q_;
  ocs2::vector_t qd_;

  Eigen::Matrix<double, 18, 1> _xhat;
  Eigen::Matrix<double, 12, 1> _ps;
  Eigen::Matrix<double, 12, 1> _vs;
  Eigen::Matrix<double, 18, 18> _A;
  Eigen::Matrix<double, 18, 18> _Q0;
  Eigen::Matrix<double, 18, 18> _P;
  Eigen::Matrix<double, 28, 28> _R0;
  Eigen::Matrix<double, 18, 3> _B;
  Eigen::Matrix<double, 28, 18> _C;

  double process_noise_pimu_;
  double process_noise_vimu_;
  double process_noise_pfoot_;
  double sensor_noise_pimu_rel_foot_;
  double sensor_noise_vimu_rel_foot_;
  double sensor_noise_zfoot_;
  double footRadius_;

  Timer timer_;
  double last_estimate_time_;

  double* sensorData_;
  double estimatorWaitingTime = 2;
  double estimatorFrequency_;
  std::thread estimator_thread_;
  std::atomic<bool> stop_thread_;  // Atomic flag to control thread exit

  std::vector<pinocchio::FrameIndex> footId_;
  pinocchio::ModelTpl<ocs2::scalar_t> model_;
  pinocchio::DataTpl<ocs2::scalar_t> data_;
};