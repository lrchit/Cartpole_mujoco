
#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include "ModelHelperFunctions.h"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/container/aligned-vector.hpp>
#include <pinocchio/spatial/force.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>

#include <iostream>
#include <string>

namespace ocs2 {
namespace legged_robot {

template <typename SCALAR_T>
vector_s_t<SCALAR_T> originForwardDynamics(const pinocchio::ModelTpl<SCALAR_T>& model,
    const pinocchio::DataTpl<SCALAR_T>& data,
    const vector_s_t<SCALAR_T>& state,
    const vector_s_t<SCALAR_T>& input,
    const std::vector<size_t>& endEffectorFrameIndices) {
  auto data_ = data;

  const vector_s_t<SCALAR_T> q = state.head(18);
  const vector_s_t<SCALAR_T> v = state.tail(18);
  const vector_s_t<SCALAR_T> torque = input.head(12);
  const vector_s_t<SCALAR_T> tau = (vector_s_t<SCALAR_T>(18) << vector_s_t<SCALAR_T>::Zero(6), torque).finished();
  const vector_s_t<SCALAR_T> lambda = input.tail(12);

  pinocchio::forwardKinematics(model, data_, q, v);
  pinocchio::updateFramePlacements(model, data_);

  pinocchio::container::aligned_vector<pinocchio::ForceTpl<SCALAR_T>> fext(model.njoints, pinocchio::ForceTpl<SCALAR_T>::Zero());
  for (size_t i = 0; i < 4; i++) {
    const auto frameIndex = endEffectorFrameIndices[i];
    const auto jointIndex = model.frames[frameIndex].parent;
    const vector3_s_t<SCALAR_T> translationJointFrameToContactFrame = model.frames[frameIndex].placement.translation();
    const matrix3_s_t<SCALAR_T> rotationWorldFrameToJointFrame = data_.oMi[jointIndex].rotation().transpose();
    const vector3_s_t<SCALAR_T> contactForce = rotationWorldFrameToJointFrame * lambda.segment(3 * i, 3);
    fext[jointIndex].linear() = contactForce;
    fext[jointIndex].angular() = translationJointFrameToContactFrame.cross(contactForce);
  }

  return pinocchio::aba(model, data_, q, v, tau, fext);
}
template vector_t originForwardDynamics(const pinocchio::ModelTpl<scalar_t>&,
    const pinocchio::DataTpl<scalar_t>&,
    const vector_t&,
    const vector_t&,
    const std::vector<size_t>&);
template ad_vector_t originForwardDynamics(const pinocchio::ModelTpl<ad_scalar_t>&,
    const pinocchio::DataTpl<ad_scalar_t>&,
    const ad_vector_t&,
    const ad_vector_t&,
    const std::vector<size_t>&);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename SCALAR_T>
vector_s_t<SCALAR_T> forwardDynamics(const pinocchio::ModelTpl<SCALAR_T>& model,
    const pinocchio::DataTpl<SCALAR_T>& data,
    const vector_s_t<SCALAR_T>& state,
    const vector_s_t<SCALAR_T>& input,
    const ContactModelParam& param,
    const std::vector<size_t>& endEffectorFrameIndices) {
  auto data_temp = data;

  const vector_s_t<SCALAR_T> q = state.head(18);
  const vector_s_t<SCALAR_T> v = state.tail(18);
  const vector_s_t<SCALAR_T> torque = input;
  const vector_s_t<SCALAR_T> tau = (vector_s_t<SCALAR_T>(18) << vector_s_t<SCALAR_T>::Zero(6), torque).finished();

  pinocchio::forwardKinematics(model, data_temp, q);
  pinocchio::updateFramePlacements(model, data_temp);

  pinocchio::container::aligned_vector<pinocchio::ForceTpl<SCALAR_T>> fext(model.njoints, pinocchio::ForceTpl<SCALAR_T>::Zero());
  for (size_t i = 0; i < 4; i++) {
    const auto frameIndex = endEffectorFrameIndices[i];
    const vector3_s_t<SCALAR_T> contactReactionForce = computeEEForce(model, data_temp, param, frameIndex, state);
    const auto jointIndex = model.frames[frameIndex].parent;
    const vector3_s_t<SCALAR_T> translationJointFrameToContactFrame = model.frames[frameIndex].placement.translation();
    const matrix3_s_t<SCALAR_T> rotationWorldFrameToJointFrame = data_temp.oMi[jointIndex].rotation().transpose();
    const vector3_s_t<SCALAR_T> contactForce = rotationWorldFrameToJointFrame * contactReactionForce;
    fext[jointIndex].linear() = contactForce;
    fext[jointIndex].angular() = translationJointFrameToContactFrame.cross(contactForce);
  }

  return pinocchio::aba(model, data_temp, q, v, tau, fext);
}
template vector_t forwardDynamics(const pinocchio::ModelTpl<scalar_t>&,
    const pinocchio::DataTpl<scalar_t>&,
    const vector_t&,
    const vector_t&,
    const ContactModelParam&,
    const std::vector<size_t>&);
template ad_vector_t forwardDynamics(const pinocchio::ModelTpl<ad_scalar_t>&,
    const pinocchio::DataTpl<ad_scalar_t>&,
    const ad_vector_t&,
    const ad_vector_t&,
    const ContactModelParam&,
    const std::vector<size_t>&);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename SCALAR_T>
vector3_s_t<SCALAR_T> getEEPosition(const pinocchio::ModelTpl<SCALAR_T>& model,
    const pinocchio::DataTpl<SCALAR_T>& data,
    const size_t endEffectorFrameIndex,
    const vector_s_t<SCALAR_T>& state) {
  auto data_temp = data;

  const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
  const vector_s_t<SCALAR_T> q = state.head(18);

  const vector3_s_t<SCALAR_T> pos = data_temp.oMf[endEffectorFrameIndex].translation();

  return pos;
}
template vector3_t getEEPosition(const pinocchio::ModelTpl<scalar_t>&, const pinocchio::DataTpl<scalar_t>&, const size_t, const vector_t&);
template ad_vector3_t getEEPosition(const pinocchio::ModelTpl<ad_scalar_t>&, const pinocchio::DataTpl<ad_scalar_t>&, const size_t, const ad_vector_t&);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename SCALAR_T>
vector6_s_t<SCALAR_T> getEEPosVel(const pinocchio::ModelTpl<SCALAR_T>& model,
    const pinocchio::DataTpl<SCALAR_T>& data,
    const size_t endEffectorFrameIndex,
    const vector_s_t<SCALAR_T>& state) {
  auto data_temp = data;

  const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
  const vector_s_t<SCALAR_T> q = state.head(18);
  const vector_s_t<SCALAR_T> v = state.tail(18);

  const vector3_s_t<SCALAR_T> pos = data_temp.oMf[endEffectorFrameIndex].translation();
  const vector3_s_t<SCALAR_T> vel = pinocchio::getFrameVelocity(model, data_temp, endEffectorFrameIndex, rf).linear();
  vector6_s_t<SCALAR_T> posVel;
  posVel.head(3) = pos;
  posVel.tail(3) = vel;
  return posVel;
}
template vector6_t getEEPosVel(const pinocchio::ModelTpl<scalar_t>&, const pinocchio::DataTpl<scalar_t>&, const size_t, const vector_t&);
template ad_vector6_t getEEPosVel(const pinocchio::ModelTpl<ad_scalar_t>&, const pinocchio::DataTpl<ad_scalar_t>&, const size_t, const ad_vector_t&);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename SCALAR_T>
vector3_s_t<SCALAR_T> computeEEForce(const pinocchio::ModelTpl<SCALAR_T>& model,
    const pinocchio::DataTpl<SCALAR_T>& data,
    const ContactModelParam& param,
    const size_t endEffectorFrameIndex,
    const vector_s_t<SCALAR_T>& state) {
  auto data_temp = data;

  const vector6_s_t<SCALAR_T> eePosVel = getEEPosVel(model, data_temp, endEffectorFrameIndex, state);
  vector3_s_t<SCALAR_T> eePenetration;
  eePenetration << SCALAR_T(0.0), SCALAR_T(0.0), eePosVel[2];

  vector3_s_t<SCALAR_T> eeForce;
  const bool isContact = param.smoothing == ContactModelParam::smoothingType::NONE && getContactFlagFromPenetration(eePenetration) ? false : true;
  if (isContact) {
    const vector3_s_t<SCALAR_T> eeVelocity = eePosVel.tail(3);
    computeDamperForce(eeForce, param, eePenetration, eeVelocity);
    smoothEEForce(eeForce, param, eePenetration);
    computeNormalSpring(eeForce, param, eePenetration(2) - SCALAR_T(param.zOffset), eeVelocity(2));
  } else {
    eeForce.setZero();
  }
  return eeForce;
}
template vector3_t computeEEForce(const pinocchio::ModelTpl<scalar_t>&,
    const pinocchio::DataTpl<scalar_t>&,
    const ContactModelParam&,
    const size_t,
    const vector_t&);
template ad_vector3_t computeEEForce(const pinocchio::ModelTpl<ad_scalar_t>&,
    const pinocchio::DataTpl<ad_scalar_t>&,
    const ContactModelParam&,
    const size_t,
    const ad_vector_t&);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename SCALAR_T>
void computeDamperForce(vector3_s_t<SCALAR_T>& eeForce,
    const ContactModelParam& param,
    const vector3_s_t<SCALAR_T>& eePenetration,
    const vector3_s_t<SCALAR_T>& eeVelocity) {
  eeForce = -SCALAR_T(param.damper_d) * eeVelocity;
}
template void computeDamperForce(vector3_t&, const ContactModelParam&, const vector3_t&, const vector3_t&);
template void computeDamperForce(ad_vector3_t&, const ContactModelParam&, const ad_vector3_t&, const ad_vector3_t&);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename SCALAR_T>
void smoothEEForce(vector3_s_t<SCALAR_T>& eeForce, const ContactModelParam& param, const vector3_s_t<SCALAR_T>& eePenetration) {
  switch (param.smoothing) {
    case ContactModelParam::smoothingType::NONE:
      return;
    case ContactModelParam::smoothingType::SIGMOID:
      eeForce *= SCALAR_T(1.0) / (SCALAR_T(1.0) + exp(eePenetration[2] * SCALAR_T(param.alpha)));
      return;
    case ContactModelParam::smoothingType::TANH:
      // same as sigmoid, maybe cheaper / more expensive to compute?
      eeForce *= SCALAR_T(0.5) * tanh(-SCALAR_T(0.5) * eePenetration[2] * SCALAR_T(param.alpha)) + SCALAR_T(0.5);
      return;
    case ContactModelParam::smoothingType::ABS:
      eeForce *= SCALAR_T(0.5) * -eePenetration[2] * SCALAR_T(param.alpha) / (SCALAR_T(1.0) + fabs(-eePenetration[2] * SCALAR_T(param.alpha))) + SCALAR_T(0.5);
      return;
    default:
      throw std::runtime_error("undefined smoothing function");
  }
}
template void smoothEEForce(vector3_t&, const ContactModelParam&, const vector3_t&);
template void smoothEEForce(ad_vector3_t&, const ContactModelParam&, const ad_vector3_t&);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename SCALAR_T>
void computeNormalSpring(vector3_s_t<SCALAR_T>& eeForce, const ContactModelParam& param, const SCALAR_T& p_N, const SCALAR_T& p_dot_N) {
  if (param.alpha_n > 0.0) {
    eeForce[2] += SCALAR_T(param.spring_k) * exp(-SCALAR_T(param.alpha_n) * p_N);
  } else if (p_N <= SCALAR_T(0.0)) {
    eeForce[2] -= SCALAR_T(param.spring_k) * p_N;
  }
}
template void computeNormalSpring(vector3_t&, const ContactModelParam&, const scalar_t&, const scalar_t&);
template void computeNormalSpring(ad_vector3_t&, const ContactModelParam&, const ad_scalar_t&, const ad_scalar_t&);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename SCALAR_T>
vector_s_t<SCALAR_T> inverseDynamics(const pinocchio::ModelTpl<SCALAR_T>& model,
    const pinocchio::DataTpl<SCALAR_T>& data,
    const vector_s_t<SCALAR_T>& state,
    const vector_s_t<SCALAR_T>& lambda,
    const vector_s_t<SCALAR_T>& vdot,
    const std::vector<size_t>& endEffectorFrameIndices) {
  auto data_temp = data;

  const vector_s_t<SCALAR_T> q = state.head(18);
  const vector_s_t<SCALAR_T> v = state.tail(18);

  pinocchio::forwardKinematics(model, data_temp, q);
  pinocchio::updateFramePlacements(model, data_temp);

  pinocchio::container::aligned_vector<pinocchio::ForceTpl<SCALAR_T, 0>> fext(model.njoints, pinocchio::ForceTpl<SCALAR_T, 0>::Zero());
  for (size_t i = 0; i < 4; i++) {
    const auto frameIndex = endEffectorFrameIndices[i];
    const auto jointIndex = model.frames[frameIndex].parent;
    const vector3_s_t<SCALAR_T> translationJointFrameToContactFrame = model.frames[frameIndex].placement.translation();
    const matrix3_s_t<SCALAR_T> rotationWorldFrameToJointFrame = data_temp.oMi[jointIndex].rotation().transpose();
    const vector3_s_t<SCALAR_T> contactForce = rotationWorldFrameToJointFrame * lambda.segment(3 * i, 3);
    fext[jointIndex].linear() = contactForce;
    fext[jointIndex].angular() = translationJointFrameToContactFrame.cross(contactForce);
  }

  vector_s_t<SCALAR_T> tau = pinocchio::rnea(model, data_temp, q, v, vdot, fext);
  return tau.tail(12);
}
template vector_t inverseDynamics(const pinocchio::ModelTpl<scalar_t>&,
    const pinocchio::DataTpl<scalar_t>&,
    const vector_t&,
    const vector_t&,
    const vector_t&,
    const std::vector<size_t>&);
template ad_vector_t inverseDynamics(const pinocchio::ModelTpl<ad_scalar_t>&,
    const pinocchio::DataTpl<ad_scalar_t>&,
    const ad_vector_t&,
    const ad_vector_t&,
    const ad_vector_t&,
    const std::vector<size_t>&);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename SCALAR_T>
vector_s_t<SCALAR_T> weightCompensatingInput(const pinocchio::ModelTpl<SCALAR_T>& model,
    const pinocchio::DataTpl<SCALAR_T>& data,
    const vector_s_t<SCALAR_T>& state,
    const std::vector<size_t>& endEffectorFrameIndices) {
  auto data_temp = data;

  const vector_s_t<SCALAR_T> q = state.head(18);
  const vector_s_t<SCALAR_T> v = state.tail(18);

  pinocchio::forwardKinematics(model, data_temp, q, v);
  pinocchio::updateFramePlacements(model, data_temp);

  // lambda part:
  vector_s_t<SCALAR_T> lambda = vector_s_t<SCALAR_T>::Zero(12);
  size_t numStanceLegs = 0;
  contact_flag_t contactFlags;
  for (size_t i = 0; i < 4; i++) {
    const vector3_s_t<SCALAR_T> eePenetration = getEEPosition(model, data, endEffectorFrameIndices[i], state);
    contactFlags[i] = getContactFlagFromPenetration(eePenetration);
    if (contactFlags[i]) {
      numStanceLegs++;
    }
  }
  if (numStanceLegs > 0) {
    const SCALAR_T totalWeight = pinocchio::computeTotalMass(model) * 9.81;
    const vector3_s_t<SCALAR_T> forceInInertialFrame(SCALAR_T(0.0), SCALAR_T(0.0), totalWeight / SCALAR_T(numStanceLegs));
    for (size_t i = 0; i < contactFlags.size(); i++) {
      if (contactFlags[i]) {
        lambda.segment(3 * i, 3).noalias() = forceInInertialFrame;
      }
    }  // end of i loop
  }

  // tau part:
  const vector_s_t<SCALAR_T> vdot = vector_s_t<SCALAR_T>::Zero(18);
  return inverseDynamics(model, data_temp, state, lambda, vdot, endEffectorFrameIndices);
}
template vector_t weightCompensatingInput(const pinocchio::ModelTpl<scalar_t>&,
    const pinocchio::DataTpl<scalar_t>&,
    const vector_t&,
    const std::vector<size_t>&);
template ad_vector_t weightCompensatingInput(const pinocchio::ModelTpl<ad_scalar_t>&,
    const pinocchio::DataTpl<ad_scalar_t>&,
    const ad_vector_t&,
    const std::vector<size_t>&);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename SCALAR_T>
bool getContactFlagFromPenetration(const vector3_s_t<SCALAR_T>& eePenetration) {
  return eePenetration[2] > SCALAR_T(0.0) ? false : true;
}
template bool getContactFlagFromPenetration(const vector3_t&);
template bool getContactFlagFromPenetration(const ad_vector3_t&);

}  // namespace legged_robot
}  // namespace ocs2