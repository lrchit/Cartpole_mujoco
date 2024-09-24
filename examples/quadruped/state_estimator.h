
#pragma once

#include <Types.h>

#include <RotationTransforms.h>

class StateEstimator {
  public:
  StateEstimator(YAML::Node config) {
    nq_ = config["nq"].as<int>();
    nv_ = config["nv"].as<int>();

    q_.resize(nq_);
    qd_.resize(nv_);
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
    Eigen::Quaternion<double> quaternion;
    quaternion.w() = *(sensordata + quat_sensor_adr + 0);
    quaternion.x() = *(sensordata + quat_sensor_adr + 1);
    quaternion.y() = *(sensordata + quat_sensor_adr + 2);
    quaternion.z() = *(sensordata + quat_sensor_adr + 3);
    const Eigen::Vector3d zyxEulerAngle = ocs2::quatToZyx(quaternion);
    q_.segment(3, 3) = zyxEulerAngle;

    // --- get leg_qpos ---
    int qpos_sensor_adr = 16;
    for (int i = 0; i < nq_ - 6; ++i) {
      q_[6 + i] = *(sensordata + qpos_sensor_adr + i);
    }

    // --- get linvel ---
    int linvel_sensor_adr = 7;
    qd_.head(3) = Eigen::Vector3d({*(sensordata + linvel_sensor_adr), *(sensordata + linvel_sensor_adr + 1), *(sensordata + linvel_sensor_adr + 2)});

    // --- get angvel ---
    int angvel_sensor_adr = 10;
    const Eigen::Vector3d localAngularVel =
        Eigen::Vector3d({*(sensordata + angvel_sensor_adr), *(sensordata + angvel_sensor_adr + 1), *(sensordata + angvel_sensor_adr + 2)});
    qd_.segment(3, 3) = ocs2::getEulerAnglesZyxDerivativesFromLocalAngularVelocity(zyxEulerAngle, localAngularVel);

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