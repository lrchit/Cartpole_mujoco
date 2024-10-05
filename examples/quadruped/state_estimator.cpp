
#include <state_estimator.h>

StateEstimator::StateEstimator(YAML::Node config, const pinocchio::ModelTpl<ocs2::scalar_t> model, std::vector<pinocchio::FrameIndex> footId)
    : model_(model), data_(model), footId_(footId) {
  nq_ = config["nq"].as<int>();
  nv_ = config["nv"].as<int>();
  use_cheater_mode_ = config["use_cheater_mode"].as<bool>();
  estimatorFrequency_ = config["estimatorFrequency"].as<double>();

  q_.setZero(nq_);
  qd_.setZero(nv_);

  _xhat.setZero();
  _ps.setZero();
  _vs.setZero();
  _A.setZero();
  _A.block(0, 0, 3, 3) = ocs2::matrix3_t::Identity();
  _A.block(3, 3, 3, 3) = ocs2::matrix3_t::Identity();
  _A.block(6, 6, 12, 12) = Eigen::Matrix<double, 12, 12>::Identity();
  _B.setZero();
  Eigen::Matrix<double, 3, 6> C1;
  C1 << ocs2::matrix3_t::Identity(), ocs2::matrix3_t::Zero();
  Eigen::Matrix<double, 3, 6> C2;
  C2 << ocs2::matrix3_t::Zero(), ocs2::matrix3_t::Identity();
  _C.setZero();
  _C.block(0, 0, 3, 6) = C1;
  _C.block(3, 0, 3, 6) = C1;
  _C.block(6, 0, 3, 6) = C1;
  _C.block(9, 0, 3, 6) = C1;
  _C.block(0, 6, 12, 12) = -1 * Eigen::Matrix<double, 12, 12>::Identity();
  _C.block(12, 0, 3, 6) = C2;
  _C.block(15, 0, 3, 6) = C2;
  _C.block(18, 0, 3, 6) = C2;
  _C.block(21, 0, 3, 6) = C2;
  _C(27, 17) = 1;
  _C(26, 14) = 1;
  _C(25, 11) = 1;
  _C(24, 8) = 1;
  _P.setIdentity();
  _P = 100 * _P;
  _Q0.setIdentity();
  _R0.setIdentity();

  process_noise_pimu_ = config["imu_process_noise_position"].as<double>();
  process_noise_vimu_ = config["imu_process_noise_velocity"].as<double>();
  process_noise_pfoot_ = config["foot_process_noise_position"].as<double>();
  sensor_noise_pimu_rel_foot_ = config["foot_sensor_noise_position"].as<double>();
  sensor_noise_vimu_rel_foot_ = config["foot_sensor_noise_velocity"].as<double>();
  sensor_noise_zfoot_ = config["foot_height_sensor_noise"].as<double>();
  footRadius_ = config["footRadius"].as<double>();

  timer_.reset();
  last_estimate_time_ = 0;

  estimator_thread_ = std::thread(&StateEstimator::callStateEstimator, this);
}

void StateEstimator::callStateEstimator() {
  while (sensorData_ == nullptr) {  // wait for data
    std::this_thread::sleep_for(std::chrono::duration<double>(1 / estimatorFrequency_));
  }

  // start estimate loop
  while (!stop_thread_) {
    const double start_time = timer_.getCurrentTime();
    if (use_cheater_mode_) {
      cheaterComputeState();
    } else {
      linearKalmanFilterComputeState();
    }

    // sleep if mpc is too fast
    const double duration_time = timer_.getCurrentTime() - start_time;
    if (duration_time < 1 / estimatorFrequency_) {
      const std::chrono::duration<double> interval(1.0 / estimatorFrequency_ - duration_time);
      std::this_thread::sleep_for(interval);
    }  // compute for next solution immediately if it's too slow
  }
}

void StateEstimator::cheaterComputeState() {
  // --- get pos ---
  int pos_sensor_adr = 0;
  q_.head(3) = ocs2::vector3_t({*sensorData_, *(sensorData_ + 1), *(sensorData_ + 2)});

  // --- get rpy ---
  int quat_sensor_adr = 3;
  Eigen::Quaternion<double> quaternion;
  quaternion.w() = *(sensorData_ + quat_sensor_adr + 0);
  quaternion.x() = *(sensorData_ + quat_sensor_adr + 1);
  quaternion.y() = *(sensorData_ + quat_sensor_adr + 2);
  quaternion.z() = *(sensorData_ + quat_sensor_adr + 3);
  const ocs2::vector3_t zyxEulerAngle = ocs2::quatToZyx(quaternion);
  q_.segment(3, 3) = zyxEulerAngle;

  // --- get leg_qpos ---
  int qpos_sensor_adr = 16;
  for (int i = 0; i < nq_ - 6; ++i) {
    q_[6 + i] = *(sensorData_ + qpos_sensor_adr + i);
  }

  // --- get linvel ---
  int linvel_sensor_adr = 7;
  qd_.head(3) = ocs2::vector3_t({*(sensorData_ + linvel_sensor_adr), *(sensorData_ + linvel_sensor_adr + 1), *(sensorData_ + linvel_sensor_adr + 2)});

  // --- get angvel ---
  int angvel_sensor_adr = 10;
  const ocs2::vector3_t localAngularVel =
      ocs2::vector3_t({*(sensorData_ + angvel_sensor_adr), *(sensorData_ + angvel_sensor_adr + 1), *(sensorData_ + angvel_sensor_adr + 2)});
  qd_.segment(3, 3) = ocs2::getEulerAnglesZyxDerivativesFromLocalAngularVelocity(zyxEulerAngle, localAngularVel);

  // --- get leg_qvel ---
  int qvel_sensor_adr = 28;
  for (int i = 0; i < nv_ - 6; ++i) {
    qd_[6 + i] = *(sensorData_ + qvel_sensor_adr + i);
  }
}

void StateEstimator::linearKalmanFilterComputeState() {
  // reset A, B, Q0
  double dt = timer_.getCurrentTime() - last_estimate_time_;
  last_estimate_time_ = timer_.getCurrentTime();
  std::cerr << "dt = " << dt << std::endl;
  _A.block(0, 3, 3, 3) = dt * ocs2::matrix3_t::Identity();
  _B.block(3, 0, 3, 3) = dt * ocs2::matrix3_t::Identity();
  _Q0.block(0, 0, 3, 3) = (dt / 20) * ocs2::matrix3_t::Identity();
  _Q0.block(3, 3, 3, 3) = (dt * 9.8 / 20) * ocs2::matrix3_t::Identity();
  _Q0.block(6, 6, 12, 12) = dt * Eigen::Matrix<double, 12, 12>::Identity();

  // --- get rpy ---
  int quat_sensor_adr = 3;
  Eigen::Quaternion<double> quaternion;
  quaternion.w() = *(sensorData_ + quat_sensor_adr + 0);
  quaternion.x() = *(sensorData_ + quat_sensor_adr + 1);
  quaternion.y() = *(sensorData_ + quat_sensor_adr + 2);
  quaternion.z() = *(sensorData_ + quat_sensor_adr + 3);
  const ocs2::vector3_t zyxEulerAngle = ocs2::quatToZyx(quaternion);
  q_.segment(3, 3) = zyxEulerAngle;

  // --- get angvel ---
  int angvel_sensor_adr = 10;
  const ocs2::vector3_t localAngularVel =
      ocs2::vector3_t({*(sensorData_ + angvel_sensor_adr), *(sensorData_ + angvel_sensor_adr + 1), *(sensorData_ + angvel_sensor_adr + 2)});
  qd_.segment(3, 3) = ocs2::getEulerAnglesZyxDerivativesFromLocalAngularVelocity(zyxEulerAngle, localAngularVel);

  // --- get linear acceleration ---
  int linacc_sensor_adr = 13;
  const ocs2::vector3_t a = ocs2::getRotationMatrixFromZyxEulerAngles(zyxEulerAngle) *
                                ocs2::vector3_t(sensorData_[linacc_sensor_adr + 0], sensorData_[linacc_sensor_adr + 1], sensorData_[linacc_sensor_adr + 2]) +
                            ocs2::vector3_t(0, 0, -9.81);

  // --- get leg_qpos ---
  int qpos_sensor_adr = 16;
  for (int i = 0; i < nq_ - 6; ++i) {
    q_[6 + i] = *(sensorData_ + qpos_sensor_adr + i);
  }

  // --- get leg_qvel ---
  int qvel_sensor_adr = 28;
  for (int i = 0; i < nv_ - 6; ++i) {
    qd_[6 + i] = *(sensorData_ + qvel_sensor_adr + i);
  }

  // --- get contact force ---
  int contact_sensor_adr = 40;
  std::vector<bool> contact_flags;
  for (int i = 0; i < 4; ++i) {
    if (*(sensorData_ + contact_sensor_adr + i) > 0.001) {
      contact_flags.push_back(true);
    } else {
      contact_flags.push_back(false);
    }
    std::cerr << "contact_flags = " << contact_flags[i] << std::endl;
  }

  // Kalman filtering

  Eigen::Matrix<double, 18, 18> Q = Eigen::Matrix<double, 18, 18>::Identity();
  Q.block(0, 0, 3, 3) = _Q0.block(0, 0, 3, 3) * process_noise_pimu_;
  Q.block(3, 3, 3, 3) = _Q0.block(3, 3, 3, 3) * process_noise_vimu_;
  Q.block(6, 6, 12, 12) = _Q0.block(6, 6, 12, 12) * process_noise_pfoot_;

  Eigen::Matrix<double, 28, 28> R = Eigen::Matrix<double, 28, 28>::Identity();
  R.block(0, 0, 12, 12) = _R0.block(0, 0, 12, 12) * sensor_noise_pimu_rel_foot_;
  R.block(12, 12, 12, 12) = _R0.block(12, 12, 12, 12) * sensor_noise_vimu_rel_foot_;
  R.block(24, 24, 4, 4) = _R0.block(24, 24, 4, 4) * sensor_noise_zfoot_;

  int qindex = 0;
  int rindex1 = 0;
  int rindex2 = 0;
  int rindex3 = 0;

  Eigen::Matrix<double, 4, 1> pzs = Eigen::Matrix<double, 4, 1>::Zero();
  for (int i = 0; i < 4; ++i) {
    int i1 = 3 * i;

    const int frameIndex = footId_[i];
    pinocchio::forwardKinematics(model_, data_, q_, qd_);
    pinocchio::updateFramePlacements(model_, data_);
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const ocs2::vector3_t p_f = data_.oMf[frameIndex].translation();
    const ocs2::vector3_t dp_f = pinocchio::getFrameVelocity(model_, data_, frameIndex, rf).linear();
    std::cerr << "p_f" << p_f.transpose() << std::endl;

    qindex = 6 + i1;
    rindex1 = i1;
    rindex2 = 12 + i1;
    rindex3 = 24 + i;

    double high_suspect_number(100);
    Q.block(qindex, qindex, 3, 3) = (contact_flags[i] ? 1 : high_suspect_number) * Q.block(qindex, qindex, 3, 3);
    // R.block(rindex1, rindex1, 3, 3) = 1 * R.block(rindex1, rindex1, 3, 3);
    R.block(rindex1, rindex1, 3, 3) = (contact_flags[i] ? 1 : high_suspect_number) * R.block(rindex1, rindex1, 3, 3);
    R.block(rindex2, rindex2, 3, 3) = (contact_flags[i] ? 1 : high_suspect_number) * R.block(rindex2, rindex2, 3, 3);
    R(rindex3, rindex3) = (contact_flags[i] ? 1 : high_suspect_number) * R(rindex3, rindex3);

    _ps.segment(i1, 3) = -p_f;
    _ps.segment(i1, 3)[2] += footRadius_;
    _vs.segment(i1, 3) = -dp_f;
    pzs(i) = contact_flags[i] ? footRadius_ : (_xhat[2] + p_f[2]);
  }

  Eigen::Matrix<double, 28, 1> y;
  y << _ps, _vs, pzs;
  _xhat = _A * _xhat + _B * a;
  Eigen::Matrix<double, 18, 18> At = _A.transpose();
  Eigen::Matrix<double, 18, 18> Pm = _A * _P * At + Q;
  Eigen::Matrix<double, 18, 28> Ct = _C.transpose();
  Eigen::Matrix<double, 28, 1> yModel = _C * _xhat;
  Eigen::Matrix<double, 28, 1> ey = y - yModel;
  Eigen::Matrix<double, 28, 28> S = _C * Pm * Ct + R;

  // todo compute LU only once
  Eigen::Matrix<double, 28, 1> S_ey = S.lu().solve(ey);
  _xhat += Pm * Ct * S_ey;

  Eigen::Matrix<double, 28, 18> S_C = S.lu().solve(_C);
  _P = (Eigen::Matrix<double, 18, 18>::Identity() - Pm * Ct * S_C) * Pm;

  Eigen::Matrix<double, 18, 18> Pt = _P.transpose();
  _P = (_P + Pt) / 2;

  if (_P.block(0, 0, 2, 2).determinant() > 0.000001) {
    _P.block(0, 2, 2, 16).setZero();
    _P.block(2, 0, 16, 2).setZero();
    _P.block(0, 0, 2, 2) /= 10;
  }

  // update pos, lin_vel, foot_pos
  q_.head(3) = _xhat.block(0, 0, 3, 1);
  qd_.head(3) = _xhat.block(3, 0, 3, 1);

  std::cerr << "q_" << q_.head(3).transpose() << std::endl;
  ocs2::vector3_t pos_real = ocs2::vector3_t({*sensorData_, *(sensorData_ + 1), *(sensorData_ + 2)});
  std::cout << "pos_err = \n" << (q_.head(3) - pos_real).transpose() << std::endl;
}