
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

  virtual std::vector<ocs2::vector_t> getBounds(const ocs2::vector_t& x, const ocs2::vector_t& u) = 0;

  virtual std::vector<ocs2::vector_t> getValue(const ocs2::vector_t& x, const ocs2::vector_t& u) = 0;
  virtual std::vector<ocs2::matrix_t> getFirstDerivatives(const ocs2::vector_t& x, const ocs2::vector_t& u) = 0;

  protected:
  int constraint_num_;

  ocs2::vector_t lb_;
  ocs2::vector_t ub_;
};