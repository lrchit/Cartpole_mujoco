#pragma once

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

namespace idto {
namespace optimizer {

struct ConvergenceCriteriaTolerances {
  template <typename Archive>
  void Serialize(Archive& ar) {
    ar& rel_cost_reduction;
    ar& abs_cost_reduction;
    ar& rel_gradient_along_dq;
    ar& abs_gradient_along_dq;
    ar& rel_state_change;
    ar& abs_state_change;
  }

  // Cost reduction criterion: Absolute (εₐ) and relative (εᵣ) tolerances for
  // the criterion:
  //   |Lᵏ−Lᵏ⁺¹| < εₐ + εᵣ Lᵏ
  double rel_cost_reduction{0.0};
  double abs_cost_reduction{0.0};

  // Gradient criterion: Absolute (εₐ) and relative (εᵣ) tolerances for
  // criterion on the directional derivative of the gradient along the search
  // direction, normalized by the cost to make it dimensionless. Typically these
  // tolerances are the same as those for the "Cost reduction criterion",
  // leading to similar stopping behaviour. This criterion reads:
  //   g⋅Δq < εₐ + εᵣ Lᵏ
  double rel_gradient_along_dq{0.0};
  double abs_gradient_along_dq{0.0};

  // Relative state (q) change: Absolute εₐ and εᵣ tolerances for a criterion
  // that monitors the evolution of q:
  //   ‖Δq‖ < εₐ + εᵣ‖qᵏ‖
  double rel_state_change{0.0};
  double abs_state_change{0.0};
};

}  // namespace optimizer
}  // namespace idto
