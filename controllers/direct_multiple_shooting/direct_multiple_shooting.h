
#pragma once

#include <hpipm_interface.h>

#include <dynamics.h>
#include <cost.h>
#include <controller.h>

#include <yaml-cpp/yaml.h>

class DirectMultipleShooting : public ControllerBase {
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  DirectMultipleShooting(YAML::Node config, std::shared_ptr<Dynamics> dynamics_model, std::shared_ptr<Cost> cost);
  ~DirectMultipleShooting() {}

  virtual ocs2::matrix_t getFeedBackMatrix() override { return K_; };
  virtual void launch_controller(const ocs2::vector_t& xcur, const std::vector<ocs2::vector_t>& x_ref) override;

  void setupProblem();
  ocs2::scalar_t calcCost();

  private:
  int nx_;
  int nu_;
  int horizon_;
  int max_inter_ = 100;
  ocs2::matrix_t K_;
  Derivatives derivatives;

  std::shared_ptr<HpipmInterface> hpipmInterface_;
  std::vector<std::shared_ptr<Cost>> cost_;
  std::vector<std::shared_ptr<Dynamics>> dynamics_model_;
};