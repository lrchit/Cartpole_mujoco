
#pragma once

#include <vector>
#include <yaml-cpp/yaml.h>

#include <Types.h>

class ControllerBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  public:
  ControllerBase() = default;
  virtual ~ControllerBase() = default;

  virtual void launch_controller(const ocs2::vector_t& xcur, const std::vector<ocs2::vector_t>& x_ref) = 0;

  virtual ocs2::matrix_t getFeedBackMatrix() = 0;
  std::vector<ocs2::vector_t> getStateTrajectory() { return xtraj; }
  std::vector<ocs2::vector_t> getInputTrajectory() { return utraj; }

  protected:
  std::vector<ocs2::vector_t> xtraj;
  std::vector<ocs2::vector_t> utraj;
  std::vector<ocs2::vector_t> xref;
};