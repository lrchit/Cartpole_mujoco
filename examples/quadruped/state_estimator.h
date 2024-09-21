
#pragma once

#include <Types.h>

class StateEstimator {
  public:
  StateEstimator(YAML::Node config) {
    nq_ = config["nq"].as<int>();
    nv_ = config["nv"].as<int>();
  }

  void callStateEstimator(double* sensordata) { cheaterComputeState(sensordata); }

  ocs2::vector_t getGeneralizedCoordinates() { return q_; }
  ocs2::vector_t getGeneralizedVelocities() { return qd_; }

  private:
  void cheaterComputeState(double* sensordata) {
    // --- get pos ---
    int pos_sensor_adr = 0;
    q_.head(3) = Eigen::Vector3d({*sensordata, *(sensordata + 1), *(sensordata + 2)});

    // --- get rpy ---
    int quat_sensor_adr = 3;
    q_.segment(3, 4) = Eigen::Vector4d(
        *(sensordata + quat_sensor_adr + 0), *(sensordata + quat_sensor_adr + 1), *(sensordata + quat_sensor_adr + 2), *(sensordata + quat_sensor_adr + 3));

    // --- get leg_qpos ---
    int qpos_sensor_adr = 16;
    for (int i = 0; i < nq_ - 7; ++i) {
      q_[7 + i] = *(sensordata + qpos_sensor_adr + i);
    }

    // --- get linvel ---
    int linvel_sensor_adr = 7;
    qd_.head(3) = Eigen::Vector3d({*(sensordata + linvel_sensor_adr), *(sensordata + linvel_sensor_adr + 1), *(sensordata + linvel_sensor_adr + 2)});

    // --- get angvel ---
    int angvel_sensor_adr = 10;
    qd_.segment(3, 3) = Eigen::Vector3d({*(sensordata + angvel_sensor_adr), *(sensordata + angvel_sensor_adr + 1), *(sensordata + angvel_sensor_adr + 2)});

    // --- get leg_qvel ---
    int qvel_sensor_adr = 28;
    for (int i = 0; i < nv_ - 6; ++i) {
      qd_[6 + i] = *(sensordata + qvel_sensor_adr + i);
    }
  }

  private:
  int nq_;
  int nv_;
  ocs2::vector_t q_;
  ocs2::vector_t qd_;
};