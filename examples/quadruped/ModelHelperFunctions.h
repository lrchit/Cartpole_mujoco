
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
  double spring_k;  // stiffness of vertical spring
  double damper_d;  // damper coefficient
  double alpha;     // velocity smoothing coefficient
  double alpha_n;   // normal force smoothing coefficient
  double zOffset;   // z offset of the plane with respect to (0, 0, 0), we currently assume flat ground at height zero penetration is only z height
  int smoothing;    // the type of velocity smoothing, NONE = 0, SIGMOID = 1, TANH = 2, ABS = 3

  enum smoothingType { NONE = 0, SIGMOID, TANH, ABS };
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
void computeDamperForce(vector3_s_t<SCALAR_T>& eeForce,
    const ContactModelParam& param,
    const vector3_s_t<SCALAR_T>& eePenetration,
    const vector3_s_t<SCALAR_T>& eeVelocity);

template <typename SCALAR_T>
void smoothEEForce(vector3_s_t<SCALAR_T>& eeForce, const ContactModelParam& param, const vector3_s_t<SCALAR_T>& eePenetration);

template <typename SCALAR_T>
void computeNormalSpring(vector3_s_t<SCALAR_T>& eeForce, const ContactModelParam& param, const SCALAR_T& p_N, const SCALAR_T& p_dot_N);

template <typename SCALAR_T>
bool getContactFlagFromPenetration(const vector3_s_t<SCALAR_T>& eePenetration);

}  // namespace legged_robot
}  // namespace ocs2