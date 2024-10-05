#pragma once

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/container/aligned-vector.hpp>
#include <pinocchio/spatial/force.hpp>
#include "pinocchio/algorithm/crba.hpp"
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>

#include <vector>

#include <Types.h>
#include "auto_diff/CppAdInterface.h"

namespace idto {
namespace optimizer {

template <typename T>
struct FootSlipAndClearanceCostPartials {
  FootSlipAndClearanceCostPartials(const int num_steps, const int nv, const int nq) {
    dJ_dq.assign(num_steps + 1, ocs2::vector_s_t<T>::Zero(nq));
    dJ_dv.assign(num_steps + 1, ocs2::vector_s_t<T>::Zero(nv));
    dJ_dqdq.assign(num_steps + 1, ocs2::matrix_s_t<T>::Zero(nq, nq));
    dJ_dvdv.assign(num_steps + 1, ocs2::matrix_s_t<T>::Zero(nv, nv));
    dJ_dqdv.assign(num_steps + 1, ocs2::matrix_s_t<T>::Zero(nq, nv));
  }

  // gradient
  std::vector<ocs2::vector_s_t<T>> dJ_dq;
  std::vector<ocs2::vector_s_t<T>> dJ_dv;

  // Hessian
  std::vector<ocs2::matrix_s_t<T>> dJ_dqdq;
  std::vector<ocs2::matrix_s_t<T>> dJ_dqdv;
  std::vector<ocs2::matrix_s_t<T>> dJ_dvdv;
};

template struct FootSlipAndClearanceCostPartials<double>;

template <typename T>
class FootSlipAndClearanceCost {
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  FootSlipAndClearanceCost(const pinocchio::ModelTpl<T>& model, const std::vector<size_t>& footId, const SolverParameters& param)
      : pinocchioModelCppAd_(model.template cast<ocs2::ad_scalar_t>()),
        pinocchioDataCppAd_(pinocchio::DataTpl<ocs2::ad_scalar_t>(pinocchioModelCppAd_)),
        footId_(footId) {
    auto systemFlowMapFunc = [&](const ocs2::ad_vector_t& x, ocs2::ad_vector_t& y) {
      ocs2::ad_vector_t q = x.head(18);
      ocs2::ad_vector_t v = x.tail(18);
      y = calcFootSlipAndClearanceCost(q, v, param.cf, param.c1);
    };
    systemFlowMapCppAdInterfacePtr_.reset(new ocs2::CppAdInterface(systemFlowMapFunc, 36, "foot_slip_and_clearance_systemFlowMap", "../cppad_generated"));
    if (param.recompileFootSlipAndClearanceCost) {
      systemFlowMapCppAdInterfacePtr_->createModels(ocs2::CppAdInterface::ApproximationOrder::Second, true);
    } else {
      systemFlowMapCppAdInterfacePtr_->loadModelsIfAvailable(ocs2::CppAdInterface::ApproximationOrder::Second, true);
    }
  }
  ~FootSlipAndClearanceCost() = default;

  ocs2::scalar_t getValue(const ocs2::vector_t& q, const ocs2::vector_t& v) {
    const ocs2::vector_t x = (ocs2::vector_t(q.rows() + v.rows()) << q, v).finished();
    return systemFlowMapCppAdInterfacePtr_->getFunctionValue(x).value();
  }

  std::pair<ocs2::vector_t, ocs2::vector_t> getFirstDerivatives(const ocs2::vector_t& q, const ocs2::vector_t& v) {
    const ocs2::vector_t x = (ocs2::vector_t(q.rows() + v.rows()) << q, v).finished();
    ocs2::vector_t gradient = systemFlowMapCppAdInterfacePtr_->getJacobian(x).transpose();
    return std::pair(gradient.topRows(18), gradient.bottomRows(18));  // dJ/dq, dJ/dv
  }

  std::tuple<ocs2::matrix_t, ocs2::matrix_t, ocs2::matrix_t> getSecondDerivatives(const ocs2::vector_t& q, const ocs2::vector_t& v) {
    const ocs2::vector_t x = (ocs2::vector_t(q.rows() + v.rows()) << q, v).finished();
    ocs2::matrix_t hessian = systemFlowMapCppAdInterfacePtr_->getHessian(0, x);
    return std::tuple(hessian.block(0, 0, 18, 18), hessian.block(0, 18, 18, 18), hessian.block(18, 18, 18, 18));  // ddJ/ddq, ddJ/dqdv, ddJ/ddv
  }

  private:
  // discrete
  ocs2::ad_vector_t calcFootSlipAndClearanceCost(const ocs2::ad_vector_t& q, const ocs2::ad_vector_t& v, const double cf, const double c1) {
    ocs2::ad_vector_t cost = ocs2::ad_vector_t::Zero(1);
    const ocs2::ad_vector3_t nhat = ocs2::ad_vector3_t(ocs2::ad_scalar_t(0), ocs2::ad_scalar_t(0), ocs2::ad_scalar_t(1.0));

    pinocchio::forwardKinematics(pinocchioModelCppAd_, pinocchioDataCppAd_, q, v);
    pinocchio::updateFramePlacements(pinocchioModelCppAd_, pinocchioDataCppAd_);

    for (const int frameIndex : footId_) {
      const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
      const ocs2::ad_vector3_t pos = pinocchioDataCppAd_.oMf[frameIndex].translation();
      const ocs2::ad_vector3_t vel = pinocchio::getFrameVelocity(pinocchioModelCppAd_, pinocchioDataCppAd_, frameIndex, rf).linear();
      const ocs2::ad_scalar_t distance = nhat.dot(pos);
      const ocs2::ad_vector3_t vt = vel - nhat.dot(vel) * nhat;

      // compute foot slip and clearance cost
      cost += ocs2::ad_scalar_t(cf) / (1 + exp(-ocs2::ad_scalar_t(c1) * distance)) * vel.transpose() * vel;
    }

    return cost;
  }

  std::vector<size_t> footId_;
  pinocchio::ModelTpl<ocs2::ad_scalar_t> pinocchioModelCppAd_;
  pinocchio::DataTpl<ocs2::ad_scalar_t> pinocchioDataCppAd_;

  std::shared_ptr<ocs2::CppAdInterface> systemFlowMapCppAdInterfacePtr_;  //!< CppAd code gen
};

template struct FootSlipAndClearanceCost<double>;

}  // namespace optimizer
}  // namespace idto
