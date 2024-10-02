
#include <direct_multiple_shooting.h>

DirectMultipleShooting::DirectMultipleShooting(YAML::Node config, std::shared_ptr<Dynamics> dynamics_model, std::shared_ptr<Cost> cost) {
  nx_ = config["nx"].as<int>();
  nu_ = config["nu"].as<int>();
  horizon_ = config["horizon"].as<int>() + 1;
  hpipmInterface_.reset(new HpipmInterface(config));
  // dynamics and cost
  for (int i = 0; i < horizon_; ++i) {
    dynamics_model_.push_back(dynamics_model);
    cost_.push_back(cost);
  }

  K_.setZero(nu_, nx_);
  std::vector<double> K = config["K"].as<std::vector<double>>();
  for (int i = 0; i < nu_; ++i) {
    for (int j = 0; j < nx_; ++j) {
      K_(i, j) = K[i * nx_ + nx_];
    }
  }

  derivatives.lx.resize(horizon_);
  derivatives.lu.resize(horizon_);
  derivatives.lxx.resize(horizon_);
  derivatives.lux.resize(horizon_);
  derivatives.luu.resize(horizon_);
  derivatives.fx.resize(horizon_ - 1);
  derivatives.fu.resize(horizon_ - 1);
  xtraj.resize(horizon_);
  utraj.resize(horizon_ - 1);
  xref.resize(horizon_);
  for (int k = 0; k < horizon_ - 1; ++k) {
    xtraj[k].setZero(nx_);
    utraj[k].setZero(nu_);
    xref[k].setZero(nx_);
  }
  xtraj[horizon_ - 1].setZero(nx_);
  xref[horizon_ - 1].setZero(nx_);
}

void DirectMultipleShooting::launch_controller(const ocs2::vector_t& xcur, const std::vector<ocs2::vector_t>& x_ref) {
  // set initial state
  hpipmInterface_->setInitialState(xcur);

  for (int k = 0; k < horizon_ - 1; ++k) {
    // Calculate derivatives
    std::tie(derivatives.lx[k], derivatives.lu[k]) = cost_[k]->getFirstDerivatives(xtraj[k], utraj[k], xref[k]);
    std::tie(derivatives.lxx[k], derivatives.lux[k], derivatives.luu[k]) = cost_[k]->getSecondDerivatives(xtraj[k], utraj[k], xref[k]);

    // df/dx for A and df/du for B
    std::tie(derivatives.fx[k], derivatives.fu[k]) = dynamics_model_[k]->getFirstDerivatives(xtraj[k], utraj[k]);

    // set to solver
    hpipmInterface_->setDynamics(derivatives.fx[k], derivatives.fu[k], k);
    hpipmInterface_->setCosts(derivatives.lx[k], derivatives.lu[k], derivatives.lxx[k], derivatives.lux[k], derivatives.luu[k], k);
  }
  std::tie(derivatives.lx[horizon_ - 1], derivatives.lu[horizon_ - 1]) = cost_[horizon_ - 1]->getFirstDerivatives(xtraj[horizon_ - 1], xref[horizon_ - 1]);
  std::tie(derivatives.lxx[horizon_ - 1], derivatives.lux[horizon_ - 1], derivatives.luu[horizon_ - 1]) =
      cost_[horizon_ - 1]->getSecondDerivatives(xtraj[horizon_ - 1], xref[horizon_ - 1]);
  // set to solver
  hpipmInterface_->setCosts(derivatives.lx[horizon_], derivatives.lu[horizon_], derivatives.lxx[horizon_ - 1], derivatives.lux[horizon_ - 1],
      derivatives.luu[horizon_ - 1], horizon_);

  // these are not support currently
  hpipmInterface_->setBounds();
  hpipmInterface_->setPolytopicConstraints();
  hpipmInterface_->setSoftConstraints();

  hpipmInterface_->solve(xtraj, utraj);
}