
#pragma once

#include "auto_diff/CppAdInterface.h"

class Constraint {
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Constraint() {}
  ~Constraint() = default;

  void setConstraintNum(int constraint_num) { constraint_num_ = constraint_num; }

  void setBounds(const ocs2::vector_t& lb, const ocs2::vector_t& ub) {
    assert(lb.cols() == constraint_num_);
    assert(ub.cols() == constraint_num_);
    lb_ = lb;
    ub_ = ub;
  }

  virtual Eigen::Matrix<int, Eigen::Dynamic, 1> getIndex() = 0;
  virtual std::pair<ocs2::vector_t, ocs2::vector_t> getBounds(const ocs2::vector_t& x, const ocs2::vector_t& u) { return std::pair(lb_, ub_); };
  virtual std::pair<ocs2::vector_t, ocs2::vector_t> getBounds(const ocs2::vector_t& x) { return std::pair(lb_, ub_); };

  virtual ocs2::vector_t getValue(const ocs2::vector_t& x, const ocs2::vector_t& u) = 0;
  virtual ocs2::vector_t getValue(const ocs2::vector_t& x) = 0;
  virtual std::pair<ocs2::matrix_t, ocs2::matrix_t> getFirstDerivatives(const ocs2::vector_t& x, const ocs2::vector_t& u) = 0;
  virtual std::pair<ocs2::matrix_t, ocs2::matrix_t> getFirstDerivatives(const ocs2::vector_t& x) = 0;

  protected:
  int constraint_num_;

  ocs2::vector_t lb_;
  ocs2::vector_t ub_;
};