#pragma once

#include <vector>

#include <Types.h>
#include <yaml-cpp/yaml.h>

namespace idto {
namespace optimizer {

/**
 * A struct for specifying the optimization problem
 *
 *    min x_err(T)'*Qf*x_err(T) + dt*sum{ x_err(t)'*Q*x_err(t) + u(t)'*R*u(t) }
 *    s.t. x(0) = x0
 *         multibody dynamics with contact
 *
 *  where x(t) = [q(t); v(t)], x_err(t) = x(t)-x_nom(t), and
 *  Q = [ Qq  0  ]
 *      [ 0   Qv ].
 */
struct ProblemDefinition {
  // initialize
  ProblemDefinition(YAML::Node config) {
    num_steps = config["mpc"]["horizon"].as<int>();
    q_init.setZero(18);
    v_init.setZero(18);
    Qq.setZero(18, 18);
    Qv.setZero(18, 18);
    Qf_q.setZero(18, 18);
    Qf_v.setZero(18, 18);
    R.setZero(18, 18);

    std::vector<double> q_init_vector = config["q_init"].as<std::vector<double>>();
    std::vector<double> v_init_vector = config["v_init"].as<std::vector<double>>();
    std::vector<double> Qq_vector = config["idto"]["Qq"].as<std::vector<double>>();
    std::vector<double> Qv_vector = config["idto"]["Qv"].as<std::vector<double>>();
    std::vector<double> Qf_q_vector = config["idto"]["Qfq"].as<std::vector<double>>();
    std::vector<double> Qf_v_vector = config["idto"]["Qfv"].as<std::vector<double>>();
    std::vector<double> R_vector = config["idto"]["R"].as<std::vector<double>>();
    int gaitHeuristic = config["idto"]["gaitHeuristic"].as<int>();
    double symmetricControlCostCoefficient = config["idto"]["symmetricControlCostCoefficient"].as<double>();
    for (int i = 0; i < 18; ++i) {
      q_init[i] = q_init_vector[i];
      v_init[i] = v_init_vector[i];
      Qq.diagonal()[i] = Qq_vector[i];
      Qv.diagonal()[i] = Qv_vector[i];
      Qf_q.diagonal()[i] = Qf_q_vector[i];
      Qf_v.diagonal()[i] = Qf_v_vector[i];
      R.diagonal()[i] = R_vector[i];
    }

    // symmetric control cost
    const ocs2::matrix_t I3 = ocs2::matrix_t::Identity(3, 3);
    ocs2::matrix_t D;
    ocs2::matrix_t pairedForceSelectMatrix = ocs2::matrix_t::Zero(2 * 3, 18);
    switch (gaitHeuristic) {
      case 0:
        D = ocs2::matrix_t::Identity(3, 3);
        D.block(1, 1, 2, 2) = -D.block(1, 1, 2, 2);
        pairedForceSelectMatrix.block(0, 6 + 0, 3, 3) = I3;
        pairedForceSelectMatrix.block(3, 6 + 3, 3, 3) = I3;
        pairedForceSelectMatrix.block(0, 6 + 9, 3, 3) = D;
        pairedForceSelectMatrix.block(3, 6 + 6, 3, 3) = D;
        dSymmetricControlCost_dtaudtau = symmetricControlCostCoefficient * pairedForceSelectMatrix.transpose() * pairedForceSelectMatrix;
      case 1:
        D = ocs2::matrix_t::Identity(3, 3);
        D.block(1, 1, 2, 2) = -D.block(1, 1, 2, 2);
        pairedForceSelectMatrix.block(0, 6 + 0, 3, 3) = I3;
        pairedForceSelectMatrix.block(0, 6 + 3, 3, 3) = D;
        pairedForceSelectMatrix.block(3, 6 + 9, 3, 3) = I3;
        pairedForceSelectMatrix.block(3, 6 + 6, 3, 3) = D;
        dSymmetricControlCost_dtaudtau = symmetricControlCostCoefficient * pairedForceSelectMatrix.transpose() * pairedForceSelectMatrix;
      case 2:
        D = -ocs2::matrix_t::Identity(3, 3);
        pairedForceSelectMatrix.block(0, 6 + 0, 3, 3) = I3;
        pairedForceSelectMatrix.block(3, 6 + 3, 3, 3) = D;
        pairedForceSelectMatrix.block(3, 6 + 9, 3, 3) = I3;
        pairedForceSelectMatrix.block(0, 6 + 6, 3, 3) = D;
        dSymmetricControlCost_dtaudtau = symmetricControlCostCoefficient * pairedForceSelectMatrix.transpose() * pairedForceSelectMatrix;
      case 3:
        dSymmetricControlCost_dtaudtau = ocs2::matrix_t::Zero(18, 18);
    }

    for (int k = 0; k < num_steps + 1; ++k) {
      q_nom.push_back(q_init);
      v_nom.push_back(v_init);
    }
  }

  // Time horizon (number of steps) for the optimization problem
  int num_steps;

  // Initial generalized positions
  ocs2::vector_t q_init;

  // Initial generalized velocities
  ocs2::vector_t v_init;

  // Running cost coefficients for generalized positions
  // N.B. these weights are per unit of time
  // TODO(vincekurtz): consider storing these as ocs2::vector_t, assuming they are
  // diagonal, and using X.asDiagonal() when multiplying.
  ocs2::matrix_t Qq;

  // Running cost coefficients for generalized velocities
  // N.B. these weights are per unit of time
  ocs2::matrix_t Qv;

  // Terminal cost coefficients for generalized positions
  ocs2::matrix_t Qf_q;

  // Terminal cost coefficients for generalized velocities
  ocs2::matrix_t Qf_v;

  // Control cost coefficients
  // N.B. these weights are per unit of time
  ocs2::matrix_t R;
  ocs2::matrix_t dSymmetricControlCost_dtaudtau;

  // Target generalized positions at each time step
  std::vector<ocs2::vector_t> q_nom;

  // Target generalized velocities at each time step
  std::vector<ocs2::vector_t> v_nom;
};

}  // namespace optimizer
}  // namespace idto
