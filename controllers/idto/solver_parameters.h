#pragma once

#include "convergence_criteria_tolerances.h"

#include <yaml-cpp/yaml.h>

namespace idto {
namespace optimizer {

enum LinesearchMethod {
  // Simple backtracking linesearch with Armijo's condition
  kArmijo,

  // Backtracking linesearch that tries to find a local minimum
  kBacktracking
};

enum SolverMethod { kLinesearch, kTrustRegion };

enum GradientsMethod {
  // First order forward differences.
  kForwardDifferences,
  // Second order central differences.
  kCentralDifferences,
  // Fourth order central differences.
  kCentralDifferences4,
  // Automatic differentiation.
  kAutoDiff,
  // The optimizer will not be used for the computation of gradients. If
  // requested, an exception will be thrown.
  kNoGradients
};

enum ScalingMethod {
  // Method for setting the diagonal scaling matrix D at the k^th iteration
  // based on terms in the Hessian

  // The simplest scaling scheme, which attempts to set diagonal terms of the
  // scaled Hessian H̃ = DHD to approximately 1.
  // Dᵢᵢᵏ = min(1, 1/√Hᵏᵢᵢ).
  kSqrt,

  // Adaptive variant of the sqrt scaling recommended by Moré, "Recent
  // developments in algorithms and software for trust region methods."
  // Springer, 1983.
  // Dᵢᵢᵏ = min(Dᵢᵢᵏ⁻¹, 1/√Hᵢᵢᵏ)
  kAdaptiveSqrt,

  // A less extreme version of the sqrt scaling, which produces
  // closer-to-spherical trust regions. We found that this performs better than
  // sqrt scaling on some of our examples.
  // Dᵢᵢᵏ = min(1, 1/√√Hᵏᵢᵢ)
  kDoubleSqrt,

  // Adaptive version of the double sqrt scaling.
  // Dᵢᵢᵏ = min(Dᵢᵢᵏ⁻¹, 1/√√Hᵢᵢᵏ)
  kAdaptiveDoubleSqrt
};

struct SolverParameters {
  enum LinearSolverType {
    // Dense Eigen::LDLT solver.
    kDenseLdlt,
    // Pentadiagonal LU solver.
    kPentaDiagonalLu,
  };

  // initialize
  SolverParameters(YAML::Node config) {
    convergence_tolerances.rel_cost_reduction = config["tolerances"]["rel_cost_reduction"].as<double>();
    convergence_tolerances.abs_cost_reduction = config["tolerances"]["abs_cost_reduction"].as<double>();
    convergence_tolerances.rel_gradient_along_dq = config["tolerances"]["rel_gradient_along_dq"].as<double>();
    convergence_tolerances.abs_gradient_along_dq = config["tolerances"]["abs_gradient_along_dq"].as<double>();
    convergence_tolerances.rel_state_change = config["tolerances"]["rel_state_change"].as<double>();
    convergence_tolerances.abs_state_change = config["tolerances"]["abs_state_change"].as<double>();
    equality_constraints = config["equality_constraints"].as<bool>();
    scaling = config["scaling"].as<bool>();
    method = static_cast<SolverMethod>(config["method"].as<int>());
    linear_solver = static_cast<SolverParameters::LinearSolverType>(config["linear_solver"].as<int>());
    gradients_method = static_cast<GradientsMethod>(config["gradients_method"].as<int>());
    num_threads = config["num_threads"].as<int>();
    contact_stiffness = config["contact_stiffness"].as<double>();
    dissipation_velocity = config["dissipation_velocity"].as<double>();
    smoothing_factor = config["smoothing_factor"].as<double>();
    friction_coefficient = config["friction_coefficient"].as<double>();
    stiction_velocity = config["stiction_velocity"].as<double>();
    which_contact_model = config["which_contact_model"].as<double>();
  }

  // Flag for whether we should check for convergence, along with default
  // tolerances for the convergence check
  bool check_convergence = false;
  ConvergenceCriteriaTolerances convergence_tolerances;

  SolverParameters() = default;

  // Which overall optimization strategy to use - linesearch or trust region
  SolverMethod method{SolverMethod::kTrustRegion};

  // Which linesearch strategy to use
  LinesearchMethod linesearch_method{LinesearchMethod::kArmijo};

  // Maximum number of Gauss-Newton iterations
  int max_iterations{100};

  // Maximum number of linesearch iterations
  int max_linesearch_iterations{50};

  GradientsMethod gradients_method{kForwardDifferences};

  // Select the linear solver to be used in the Gauss-Newton step computation.
  LinearSolverType linear_solver{LinearSolverType::kPentaDiagonalLu};

  // Flag for whether to print out iteration data
  bool verbose{false};

  // Flag for whether to print (and compute) additional slow-to-compute
  // debugging info, like the condition number, at each iteration
  bool print_debug_data{false};

  // Only for debugging. When `true`, the computation with sparse algebra is
  // checked against a dense LDLT computation. This is an expensive check and
  // must be avoided unless we are trying to debug loss of precision due to
  // round-off errors or similar problems.
  bool debug_compare_against_dense{false};

  // Soft contact model parameters
  // TODO(vincekurtz): this is definitely the wrong place to specify the contact
  // model - figure out the right place and put these parameters there
  double contact_stiffness{100};     // normal force stiffness, N/m
  double dissipation_velocity{0.1};  // Hunt-Crossley velocity, in m/s.
  double stiction_velocity{0.05};    // Regularization of stiction, in m/s.
  double friction_coefficient{0.5};  // Coefficient of friction.
  double smoothing_factor{0.1};      // force at a distance smoothing

  // Spring-damper contact model parameters
  double k_spring{100};
  double d_damper{100};
  double damper_smooth{100};
  double spring_smooth{100};
  double zOffset{-0.02};

  // which_contact_model
  int which_contact_model{0};

  // Flag for rescaling the Hessian, for better numerical conditioning
  bool scaling{true};

  // Method to use for rescaling the Hessian (and thus reshaping the Hessian)
  ScalingMethod scaling_method{ScalingMethod::kDoubleSqrt};

  // Parameter for activating hard equality constraints on unactuated DoFs
  bool equality_constraints{true};

  // Initial and maximum trust region radius
  // N.B. these have very different units depending on whether scaling is used.
  // These defaults are reasonable when scaling=true: without scaling a smaller
  // trust region radius is more appropriate.
  double Delta0{1e-1};
  double Delta_max{1e5};

  // Number of cpu threads for parallel computation of derivatives
  int num_threads{1};

  // Indicator for which DoFs the nominal trajectory is defined as relative to
  // the initial condition. Useful for locomotion or continuous rotation tasks.
  Eigen::Vector<bool, Eigen::Dynamic> q_nom_relative_to_q_init;
};

}  // namespace optimizer
}  // namespace idto
