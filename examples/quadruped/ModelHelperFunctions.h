
#pragma once

#include <pinocchio/fwd.hpp>  // always include it before any other header

#include "auto_diff/Types.h"

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

// namespace Eigen {
// namespace numext {
// template <>
// EIGEN_STRONG_INLINE bool isfinite(const ocs2::ad_scalar_t& x) {
//   return true;
// }
// }  // namespace numext
// }  // namespace Eigen

namespace ocs2 {
namespace legged_robot {

template <typename SCALAR_T>
vector_s_t<SCALAR_T> originForwardDynamics(const pinocchio::ModelTpl<SCALAR_T>& model,
    const pinocchio::DataTpl<SCALAR_T>& data,
    const vector_s_t<SCALAR_T>& state,
    const vector_s_t<SCALAR_T>& input,
    const std::vector<size_t>& endEffectorFrameIndices);

/**
 * Compute acceleration given generalized coordinates, generalized velocities
 * torque and contact force
 *
 * @param [in] q: generalized coordinates with ZYX euler angles in global frames
 * 
 * @param [in] v: generalized velocity with euler angles zyx derivatives and linear velocity in world frames
 * 
 * @param [in] torque: joint torques
 * 
 * @param [in] lambda: the contact forces at foot tips with order [LF, LH, RF, RH]
 * 
 * @return state derivative xdot = [qdot, vdot]
 */

struct ContactModelParam {
  double k_spring;       // stiffness of vertical spring
  double d_damper;       // damper coefficient
  double damper_smooth;  // velocity smoothing coefficient
  double spring_smooth;  // normal force smoothing coefficient
  double zOffset;        // z offset of the plane with respect to (0, 0, 0), we currently assume flat ground at height zero penetration is only z height

  double contact_stiffness;
  double dissipation_velocity;
  double smoothing_factor;
  double friction_coefficient;
  double stiction_velocity;

  int which_contact_model;
};

template <typename SCALAR_T>
vector_s_t<SCALAR_T> forwardDynamics(const pinocchio::ModelTpl<SCALAR_T>& model,
    const pinocchio::DataTpl<SCALAR_T>& data,
    const vector_s_t<SCALAR_T>& state,
    const vector_s_t<SCALAR_T>& input,
    const ContactModelParam& param,
    const std::vector<size_t>& endEffectorFrameIndices);

template <typename SCALAR_T>
vector_s_t<SCALAR_T> inverseDynamics(const pinocchio::ModelTpl<SCALAR_T>& model,
    const pinocchio::DataTpl<SCALAR_T>& data,
    const vector_s_t<SCALAR_T>& state,
    const vector_s_t<SCALAR_T>& lambda,
    const vector_s_t<SCALAR_T>& vdot,
    const std::vector<size_t>& endEffectorFrameIndices);

template <typename SCALAR_T>
vector_s_t<SCALAR_T> weightCompensatingInput(const pinocchio::ModelTpl<SCALAR_T>& model,
    const pinocchio::DataTpl<SCALAR_T>& data,
    const vector_s_t<SCALAR_T>& state,
    const std::vector<size_t>& endEffectorFrameIndices);

template <typename SCALAR_T>
vector3_s_t<SCALAR_T> getEEPosition(const pinocchio::ModelTpl<SCALAR_T>& model,
    const pinocchio::DataTpl<SCALAR_T>& data,
    const size_t endEffectorFrameIndex,
    const vector_s_t<SCALAR_T>& state);  // call "pinocchio::forwardKinematics(model, data, q);pinocchio::updateFramePlacements(model, data);" first

template <typename SCALAR_T>
vector6_s_t<SCALAR_T> getEEPosVel(const pinocchio::ModelTpl<SCALAR_T>& model,
    const pinocchio::DataTpl<SCALAR_T>& data,
    const size_t endEffectorFrameIndex,
    const vector_s_t<SCALAR_T>& state);  // call "pinocchio::forwardKinematics(model, data, q, v);pinocchio::updateFramePlacements(model, data);" first

template <typename SCALAR_T>
vector3_s_t<SCALAR_T> computeEEForce(const pinocchio::ModelTpl<SCALAR_T>& model,
    const pinocchio::DataTpl<SCALAR_T>& data,
    const ContactModelParam& param,
    const size_t endEffectorFrameIndex,
    const vector_s_t<SCALAR_T>& state);

template <typename SCALAR_T>
bool getContactFlagFromPenetration(const vector3_s_t<SCALAR_T>& eePenetration);

}  // namespace legged_robot
}  // namespace ocs2