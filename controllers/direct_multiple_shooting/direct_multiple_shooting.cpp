
#include <direct_multiple_shooting.h>

DirectMultipleShooting::DirectMultipleShooting(YAML::Node config, std::shared_ptr<Dynamics> dynamics_model, std::shared_ptr<Cost> cost) {
  nx_ = config["nx"].as<int>();
  nu_ = config["nu"].as<int>();
  horizon_ = config["horizon"].as<int>() + 1;
  max_iter_ = config["max_iteration"].as<int>();
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
  derivatives.b.resize(horizon_ - 1);
  xtraj.resize(horizon_);
  utraj.resize(horizon_ - 1);
  xref.resize(horizon_);
  for (int k = 0; k < horizon_ - 1; ++k) {
    xtraj[k].setZero(nx_);
    utraj[k].setZero(nu_);
    xref[k].setZero(nx_);
    derivatives.b[k].setZero(nx_);
  }
  xtraj[horizon_ - 1].setZero(nx_);
  xref[horizon_ - 1].setZero(nx_);
}

ocs2::scalar_t DirectMultipleShooting::calcCost() {
  ocs2::scalar_t cost = 0;
  for (int k = 0; k < horizon_ - 1; ++k) {
    cost += cost_[k]->getValue(xtraj[k], utraj[k], xref[k]);
  }
  cost += cost_[horizon_ - 1]->getValue(xtraj[horizon_ - 1], xref[horizon_ - 1]);

  return cost;
}

void DirectMultipleShooting::setupProblem() {
  for (int k = 0; k < horizon_ - 1; ++k) {
    // Calculate derivatives
    std::tie(derivatives.lx[k], derivatives.lu[k]) = cost_[k]->getFirstDerivatives(xtraj[k], utraj[k], xref[k]);
    std::tie(derivatives.lxx[k], derivatives.lux[k], derivatives.luu[k]) = cost_[k]->getSecondDerivatives(xtraj[k], utraj[k], xref[k]);

    // df/dx for A and df/du for B
    std::tie(derivatives.fx[k], derivatives.fu[k]) = dynamics_model_[k]->getFirstDerivatives(xtraj[k], utraj[k]);
  }
  std::tie(derivatives.lx[horizon_ - 1], derivatives.lu[horizon_ - 1]) = cost_[horizon_ - 1]->getFirstDerivatives(xtraj[horizon_ - 1], xref[horizon_ - 1]);
  std::tie(derivatives.lxx[horizon_ - 1], derivatives.lux[horizon_ - 1], derivatives.luu[horizon_ - 1]) =
      cost_[horizon_ - 1]->getSecondDerivatives(xtraj[horizon_ - 1], xref[horizon_ - 1]);
}

void DirectMultipleShooting::launch_controller(const ocs2::vector_t& xcur, const std::vector<ocs2::vector_t>& x_ref) {
  // set initial state
  xref = x_ref;
  xtraj[0] = xcur;
  for (int k = 0; k < (horizon_ - 1); ++k) {
    utraj[k] = dynamics_model_[k]->getQuasiStaticInput(xtraj[k]);
    // utraj[k] += K_guess * (x_ref[k + 1] - xtraj[k]);
    xtraj[k + 1] = dynamics_model_[k]->getValue(xtraj[k], utraj[k]);
  }

  // start nlp loop
  ocs2::scalar_t cost = 1.0e+9;
  ocs2::scalar_t new_cost = calcCost();
  int counter = 0;
  while ((abs(cost - new_cost) > 1.0e-2) && (counter < max_iter_)) {
    std::cerr << "mpc iter = " << counter + 1 << std::endl;
    cost = new_cost;

    // calc derivatives
    setupProblem();

    // solve nonlinear program
    hpipmInterface_->solve(derivatives.fx, derivatives.fu, derivatives.b, derivatives.lxx, derivatives.lux, derivatives.luu, derivatives.lx, derivatives.lu);
    auto delta_xtraj = hpipmInterface_->get_delta_xtraj();
    auto delta_utraj = hpipmInterface_->get_delta_utraj();

    for (size_t k = 0; k < horizon_ - 1; ++k) {
      xtraj[k + 1] = xtraj[k + 1] + delta_xtraj[k + 1];
      utraj[k] = utraj[k] + delta_utraj[k];
    }

    // update cost
    new_cost = calcCost();
    std::cerr << "new_cost = " << new_cost << std::endl;
    if (new_cost > cost) {
      break;
    }

    counter++;
  }
  for (size_t i = 0; i < utraj.size(); ++i) {
    std::cerr << "utraj " << i << " = " << utraj[i].transpose() << std::endl;
  }
  std::cerr << "mpc iter = " << counter << std::endl;
  // exit(0);
}