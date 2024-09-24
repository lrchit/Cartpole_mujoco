
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

#include <ModelHelperFunctions.h>
#include <utils.h>

int main() {
  Eigen::Vector<double, 36> state = Eigen::Vector<double, 36>::Zero();
  Eigen::Vector<double, 12> lambda = Eigen::Vector<double, 12>::Zero();
  Eigen::Vector<double, 18> vdot = Eigen::Vector<double, 18>::Zero();
  Eigen::Vector<double, 12> tau = Eigen::Vector<double, 12>::Zero();

  state.head(18) << 0, 0, 0.3, 0, 0, 0, -0., 0.72, -1.44, 0., 0.72, -1.44, -0., 0.72, -1.44, 0., 0.72, -1.44;
  tau << -1.78151, 0.294809, 3.82716, 1.78151, 0.294809, 3.82716, -1.78151, 0.294809, 3.82716, 1.78151, 0.294809, 3.82716;

  const std::string urdfFile = "../models/quadruped/urdf/a1.urdf";
  const std::vector<std::string> jointNames{"FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
      "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"};
  const pinocchio::ModelTpl<ocs2::scalar_t> model = createPinocchioModel(urdfFile, jointNames);
  pinocchio::Data data(model);

  lambda << 0, 0, pinocchio::computeTotalMass(model) * 9.81 / 4, 0, 0, pinocchio::computeTotalMass(model) * 9.81 / 4, 0, 0,
      pinocchio::computeTotalMass(model) * 9.81 / 4, 0, 0, pinocchio::computeTotalMass(model) * 9.81 / 4;
  std::cerr << "lambda = " << lambda.transpose() << std::endl;

  std::vector<pinocchio::FrameIndex> footId;
  footId.push_back(model.getFrameId("FL_foot"));
  footId.push_back(model.getFrameId("FR_foot"));
  footId.push_back(model.getFrameId("RL_foot"));
  footId.push_back(model.getFrameId("RR_foot"));

  pinocchio::forwardKinematics(model, data, state.head(18));
  pinocchio::updateFramePlacements(model, data);
  auto x = data.oMf[footId[0]].translation();
  std::cerr << "x = " << x.transpose() << std::endl;
  x = data.oMf[footId[1]].translation();
  std::cerr << "x = " << x.transpose() << std::endl;
  x = data.oMf[footId[2]].translation();
  std::cerr << "x = " << x.transpose() << std::endl;
  x = data.oMf[footId[3]].translation();
  std::cerr << "x = " << x.transpose() << std::endl;

  Eigen::VectorXd torque = ocs2::legged_robot::inverseDynamics<double>(model, data, state, lambda, vdot, footId);
  std::cerr << "torque = " << torque.transpose() << std::endl;

  Eigen::Vector<double, 24> input = (Eigen::VectorXd(tau.rows() + lambda.rows()) << tau, lambda).finished();
  Eigen::Vector<double, 18> a = ocs2::legged_robot::originForwardDynamics<double>(model, data, state, input, footId);
  std::cerr << "a = " << a.transpose() << std::endl;

  ocs2::legged_robot::ContactModelParam param;
  param.spring_k = 100;   // stiffness of vertical spring
  param.damper_d = 100;   // damper coefficient
  param.alpha = 200;      // velocity smoothing coefficient
  param.alpha_n = 100;    // normal force smoothing coefficient
  param.zOffset = -0.02;  // z offset of the plane with respect to (0, 0, 0), we currently assume flat ground at height zero penetration is only z height
  param.smoothing = 1;    // the type of velocity smoothing, NONE = 0, SIGMOID = 1, TANH = 2, ABS = 3
  Eigen::Vector<double, 18> a_contact = ocs2::legged_robot::forwardDynamics<double>(model, data, state, tau, param, footId);
  std::cerr << "a_contact = " << a_contact.transpose() << std::endl;
}