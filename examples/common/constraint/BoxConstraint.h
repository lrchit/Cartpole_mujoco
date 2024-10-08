
#pragma once

#include <constraint.h>

class BoxConstraint : public Constraint {
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BoxConstraint(int nx, int nu) : nx_(nx), nu_(nu) {}

  ~BoxConstraint() = default;

  void setBounds(const ocs2::vector_t& lbx, const ocs2::vector_t& ubx, const ocs2::vector_t& lbu, const ocs2::vector_t& ubu) {
    lbx_ = lbx;
    ubx_ = ubx;
    lbu_ = lbu;
    ubu_ = ubu;
  }

  virtual std::vector<ocs2::vector_t> getBounds(const ocs2::vector_t& x, const ocs2::vector_t& u) override {
    ocs2::vector_t lb, ubx, lbu, ubu;
    if (lbx_.cols() != 0) {
      lbx = lbx_ - x;
    }
    if (ubx_.cols() != 0) {
      ubx = ubx_ - x;
    }
    if (lbu_.cols() != 0) {
      lbu = lbu_ - u;
    }
    if (ubu_.cols() != 0) {
      ubu = ubu_ - u;
    }
    return {lbx, ubx, lbu, ubu};
  }

  virtual std::vector<ocs2::vector_t> getValue(const ocs2::vector_t& x, const ocs2::vector_t& u) override { return {x, u}; }

  virtual std::vector<ocs2::matrix_t> getFirstDerivatives(const ocs2::vector_t& x, const ocs2::vector_t& u) override {
    return {ocs2::matrix_t::Identity(nx_, nx_), ocs2::matrix_t::Identity(nu_, nu_)};
  }

  protected:
  int nx_;
  int nu_;

  ocs2::vector_t lbx_;
  ocs2::vector_t ubx_;
  ocs2::vector_t lbu_;
  ocs2::vector_t ubu_;
};