#include "trajectory_optimizer_solution.h"

namespace idto {
namespace optimizer {

std::string DecodeConvergenceReasons(ConvergenceReason reason) {
  if (reason == ConvergenceReason::kNoConvergenceCriteriaSatisfied) {
    return "no convergence criterion satisfied";
  }
  std::string reasons;
  if ((reason & ConvergenceReason::kCostReductionCriterionSatisfied) != 0)
    reasons = "cost reduction";
  if ((reason & ConvergenceReason::kGradientCriterionSatisfied) != 0) {
    if (!reasons.empty())
      reasons += ", ";
    reasons += "gradient";
  }
  if ((reason & ConvergenceReason::kSateCriterionSatisfied) != 0) {
    if (!reasons.empty())
      reasons += ", ";
    reasons += "state change";
  }
  return reasons;
}

template struct TrajectoryOptimizerSolution<double>;
template struct TrajectoryOptimizerStats<double>;

}  // namespace optimizer
}  // namespace idto
