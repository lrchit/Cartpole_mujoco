
#include <direct_multiple_shooting.h>

DirectMultipleShooting::DirectMultipleShooting(YAML::Node config,
    ocs2::matrix_t K,
    std::shared_ptr<Dynamics> dynamics_model,
    std::shared_ptr<Cost> cost,
    std::shared_ptr<Constraint> constraint) {
  nx_ = config["nx"].as<int>();
  nu_ = config["nu"].as<int>();
  horizon_ = config["mpc"]["horizon"].as<int>() + 1;
  max_iter_ = config["dms"]["max_iteration"].as<int>();
  tolerance_ = config["dms"]["tolerance"].as<double>();
  hpipmInterface_.reset(new HpipmInterface(config));
  first_run_ = true;
  // dynamics and cost
  for (int i = 0; i < horizon_; ++i) {
    dynamics_model_.push_back(dynamics_model);
    cost_.push_back(cost);
    constraint_.push_back(constraint);
  }

  K_ = K;

  costDerivatives_.reset(new CostDerivatives(horizon_));
  dynamicsDerivatives_.reset(new DynamicsDerivatives(horizon_));
  boxConstraintsStruct_.reset(new BoxConstraintsStruct(horizon_));

  xtraj.resize(horizon_);
  utraj.resize(horizon_ - 1);
  xref.resize(horizon_);
  for (int k = 0; k < horizon_ - 1; ++k) {
    xtraj[k].setZero(nx_);
    utraj[k].setZero(nu_);
    xref[k].setZero(nx_);
    dynamicsDerivatives_->b[k].setZero(nx_);
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
  // #pragma omp parallel for num_threads(4)
  std::pair<ocs2::vector_t, ocs2::vector_t> bound;
  for (int k = 0; k < horizon_ - 1; ++k) {
    // calc cost derivatives
    std::tie(costDerivatives_->lx[k], costDerivatives_->lu[k]) = cost_[k]->getFirstDerivatives(xtraj[k], utraj[k], xref[k]);
    std::tie(costDerivatives_->lxx[k], costDerivatives_->lux[k], costDerivatives_->luu[k]) = cost_[k]->getSecondDerivatives(xtraj[k], utraj[k], xref[k]);
    hpipmInterface_->setCosts(
        k, costDerivatives_->lx[k], costDerivatives_->lu[k], costDerivatives_->lxx[k], costDerivatives_->lux[k], costDerivatives_->luu[k]);

    // calc dynamics derivatives
    std::tie(dynamicsDerivatives_->fx[k], dynamicsDerivatives_->fu[k]) = dynamics_model_[k]->getFirstDerivatives(xtraj[k], utraj[k]);
    hpipmInterface_->setDynamics(k, dynamicsDerivatives_->fx[k], dynamicsDerivatives_->fu[k]);

    // // calc constraint settings
    // Eigen::Matrix<int, Eigen::Dynamic, 1> idx;
    // idx = constraint_[k]->getIndex();
    // boxConstraintsStruct_->idxbx[k] = idx.head(nx_);
    // boxConstraintsStruct_->idxbu[k] = idx.tail(nu_);
    // bound = constraint_[k]->getBounds(xtraj[k], utraj[k]);
    // std::cerr << bound.first.transpose() << std::endl;
    // std::cerr << bound.second.transpose() << std::endl;
    // boxConstraintsStruct_->lbx[k] = bound.first.head(nx_);
    // boxConstraintsStruct_->ubx[k] = bound.first.tail(nu_);
    // boxConstraintsStruct_->lbu[k] = bound.second.head(nx_);
    // boxConstraintsStruct_->ubu[k] = bound.second.tail(nu_);
    // hpipmInterface_->setBounds(k, boxConstraintsStruct_->lbx[k], boxConstraintsStruct_->ubx[k], boxConstraintsStruct_->idxbx[k], boxConstraintsStruct_->lbu[k],
    //     boxConstraintsStruct_->ubu[k], boxConstraintsStruct_->idxbu[k]);
  }
  // calc terminal cost derivatives
  std::tie(costDerivatives_->lx[horizon_ - 1], costDerivatives_->lu[horizon_ - 1]) =
      cost_[horizon_ - 1]->getFirstDerivatives(xtraj[horizon_ - 1], xref[horizon_ - 1]);
  std::tie(costDerivatives_->lxx[horizon_ - 1], costDerivatives_->lux[horizon_ - 1], costDerivatives_->luu[horizon_ - 1]) =
      cost_[horizon_ - 1]->getSecondDerivatives(xtraj[horizon_ - 1], xref[horizon_ - 1]);
  hpipmInterface_->setCosts(horizon_ - 1, costDerivatives_->lx[horizon_ - 1], costDerivatives_->lu[horizon_ - 1], costDerivatives_->lxx[horizon_ - 1],
      costDerivatives_->lux[horizon_ - 1], costDerivatives_->luu[horizon_ - 1]);

  // // calc terminal constraint settings
  // Eigen::Matrix<int, Eigen::Dynamic, 1> idx;
  // idx = constraint_[horizon_ - 1]->getIndex();
  // boxConstraintsStruct_->idxbx[horizon_ - 1] = idx.head(nx_);
  // boxConstraintsStruct_->idxbu[horizon_ - 1] = idx.tail(nu_);
  // bound = constraint_[horizon_ - 1]->getBounds(xtraj[horizon_ - 1]);
  // boxConstraintsStruct_->lbx[horizon_ - 1] = bound.first.head(nx_);
  // boxConstraintsStruct_->ubx[horizon_ - 1] = bound.first.tail(nu_);
  // boxConstraintsStruct_->lbu[horizon_ - 1] = bound.second.head(nx_);
  // boxConstraintsStruct_->ubu[horizon_ - 1] = bound.second.tail(nu_);
  // hpipmInterface_->setBounds(horizon_ - 1, boxConstraintsStruct_->lbx[horizon_ - 1], boxConstraintsStruct_->ubx[horizon_ - 1],
  //     boxConstraintsStruct_->idxbx[horizon_ - 1], boxConstraintsStruct_->lbu[horizon_ - 1], boxConstraintsStruct_->ubu[horizon_ - 1],
  //     boxConstraintsStruct_->idxbu[horizon_ - 1]);

  hpipmInterface_->setBounds();
  hpipmInterface_->setPolytopicConstraints();
  hpipmInterface_->setSoftConstraints();
}

void DirectMultipleShooting::launch_controller(const ocs2::vector_t& xcur, const std::vector<ocs2::vector_t>& x_ref) {
  // set initial state
  xref = x_ref;
  xtraj[0] = xcur;
  for (int k = 0; k < (horizon_ - 1); ++k) {
    std::tie(xtraj[k + 1], utraj[k]) = dynamics_model_[k]->solveQuasiStaticProblem(xtraj[k]);
  }

  // start nlp loop
  ocs2::scalar_t cost = 1.0e+9;
  ocs2::scalar_t new_cost = calcCost();
  int counter = 0;
  while ((abs(cost - new_cost) > tolerance_) && ((counter < max_iter_) || first_run_)) {
    cost = new_cost;

    // calc derivatives
    setupProblem();

    // solve nonlinear program
    hpipmInterface_->solve();
    auto delta_xtraj = hpipmInterface_->get_delta_xtraj();
    auto delta_utraj = hpipmInterface_->get_delta_utraj();

    for (size_t k = 0; k < horizon_ - 1; ++k) {
      xtraj[k + 1] = xtraj[k + 1] + delta_xtraj[k + 1];
      utraj[k] = utraj[k] + delta_utraj[k];
    }

    // update cost
    new_cost = calcCost();
    // std::cerr << "cost  = " << cost << std::endl;
    // std::cerr << "new_cost  = " << new_cost << std::endl;
    if (new_cost > cost) {
      std::cerr << "dms solver failed!" << std::endl;
      exit(0);
    }

    counter++;
  }
  first_run_ = false;
  // // std::cerr << "counter = " << counter << std::endl;
  // for (int k = 0; k < horizon_ - 1; ++k) {
  //   auto delta_xtraj = hpipmInterface_->get_delta_xtraj();
  //   std::cerr << "delta_xtraj = " << delta_xtraj[k].head(18).transpose() << std::endl;
  //   std::cerr << "xtraj = " << xtraj[k].head(6).transpose() << std::endl;
  //   std::cerr << "utraj = " << utraj[k].transpose() << std::endl;
  //   dynamics_model_[k]->getValue(xtraj[k], utraj[k]);
  // }
  // // exit(0);
}