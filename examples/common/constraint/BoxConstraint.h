
#pragma once

#include <constraint.h>

class BoxConstraint : public Constraint {
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BoxConstraint(int nx, int nu) : nx_(nx), nu_(nu) {
    lb_.setZero(nx_ + nu_);
    ub_.setZero(nx_ + nu_);
  }

  ~BoxConstraint() = default;

  void setStateBounds(const ocs2::vector_t& lbx, const ocs2::vector_t& ubx) {
    lb_.head(nx_) = lbx;
    ub_.head(nx_) = ubx;
  }

  void setInputBounds(const ocs2::vector_t& lbu, const ocs2::vector_t& ubu) {
    lb_.tail(nu_) = lbu;
    ub_.tail(nu_) = ubu;
  }

  virtual Eigen::Matrix<int, Eigen::Dynamic, 1> getIndex() override {
    Eigen::Matrix<int, Eigen::Dynamic, 1> idx(nx_ + nu_);
    for (int i = 0; i < nx_; ++i) {
      idx[i] = i;
    }
    for (int i = 0; i < nu_; ++i) {
      idx[i + nx_] = i;
    }
    return idx;
  }

  virtual std::pair<ocs2::vector_t, ocs2::vector_t> getBounds(const ocs2::vector_t& x, const ocs2::vector_t& u) override {
    const ocs2::vector_t value = getValue(x, u);
    return std::pair(lb_ - value, ub_ - value);
  };
  virtual std::pair<ocs2::vector_t, ocs2::vector_t> getBounds(const ocs2::vector_t& x) override {
    const ocs2::vector_t value = getValue(x);
    return std::pair(lb_ - value, ub_ - value);
  };

  virtual ocs2::vector_t getValue(const ocs2::vector_t& x, const ocs2::vector_t& u) override { return (ocs2::vector_t(nx_ + nu_) << x, u).finished(); }
  virtual ocs2::vector_t getValue(const ocs2::vector_t& x) override { return (ocs2::vector_t(nx_ + nu_) << x, ocs2::vector_t::Zero(nu_)).finished(); }

  virtual std::pair<ocs2::matrix_t, ocs2::matrix_t> getFirstDerivatives(const ocs2::vector_t& x, const ocs2::vector_t& u) override {
    return std::pair(ocs2::matrix_t::Identity(nx_, nx_), ocs2::matrix_t::Identity(nu_, nu_));
  }
  virtual std::pair<ocs2::matrix_t, ocs2::matrix_t> getFirstDerivatives(const ocs2::vector_t& x) override {
    return std::pair(ocs2::matrix_t::Identity(nx_, nx_), ocs2::matrix_t{});
  }

  protected:
  int nx_;
  int nu_;
};