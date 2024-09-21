
#pragma once

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <mujoco/mujoco.h>

inline void pd_controller(mjData *d) {
  // PD控制参数
  double kp_1 = 1000.0; // 位置增益
  double kd_1 = 100.0;  // 速度增益
  double kp_2 = 200.0;  // 位置增益
  double kd_2 = 100.0;  // 速度增益

  // cart's pos and vel
  double cartPos = d->sensordata[0];
  double cartVel = d->sensordata[2];
  // pole的角度和角速度
  double poleAngle = d->sensordata[1];
  double poleVelocity = d->sensordata[3];

  // 计算PD控制力
  double controlForce = kp_1 * (poleAngle) + kd_1 * (poleVelocity) +
                        kp_2 * (cartPos) + kd_2 * (cartVel);
  controlForce = fmin(fmax(controlForce, -100), 100);

  // std::cout << "controlForce = " << controlForce << std::endl;

  // 设置控制力
  d->ctrl[0] = controlForce;
  d->ctrl[1] = 0; // pole没有直接控制
}