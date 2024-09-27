#pragma once

#include <pinocchio/fwd.hpp>  // always include it before any other header

#include <Types.h>

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

namespace idto {
namespace optimizer {

/**
 * A container for scratch variables that we use in various intermediate
 * computations. Allows us to avoid extra allocations when speed is important.
 */
template <typename T>
struct TrajectoryOptimizerWorkspace {
  // Construct a workspace with size matching the given plant.
  TrajectoryOptimizerWorkspace(const int num_steps, const pinocchio::ModelTpl<T>& model, const pinocchio::DataTpl<T>& data) {
    const int nq = model.nq;
    const int nv = model.nv;
    const int num_vars = nq * (num_steps + 1);

    fext = pinocchio::container::aligned_vector<pinocchio::ForceTpl<T>>(model.njoints, pinocchio::ForceTpl<T>::Zero());

    // Get number of unactuated DoFs
    int num_unactuated = 6;
    const int num_eq_cons = num_unactuated * num_steps;

    // Set vector sizes
    q_size_tmp1.resize(nq);
    q_size_tmp2.resize(nq);
    q_size_tmp3.resize(nq);
    q_size_tmp4.resize(nq);

    v_size_tmp1.resize(nv);
    v_size_tmp2.resize(nv);
    v_size_tmp3.resize(nv);
    v_size_tmp4.resize(nv);
    v_size_tmp5.resize(nv);
    v_size_tmp6.resize(nv);
    v_size_tmp7.resize(nv);
    v_size_tmp8.resize(nv);

    tau_size_tmp1.resize(nv);
    tau_size_tmp2.resize(nv);
    tau_size_tmp3.resize(nv);
    tau_size_tmp4.resize(nv);
    tau_size_tmp5.resize(nv);
    tau_size_tmp6.resize(nv);
    tau_size_tmp7.resize(nv);
    tau_size_tmp8.resize(nv);
    tau_size_tmp9.resize(nv);
    tau_size_tmp10.resize(nv);
    tau_size_tmp11.resize(nv);
    tau_size_tmp12.resize(nv);

    a_size_tmp1.resize(nv);
    a_size_tmp2.resize(nv);
    a_size_tmp3.resize(nv);
    a_size_tmp4.resize(nv);
    a_size_tmp5.resize(nv);
    a_size_tmp6.resize(nv);
    a_size_tmp7.resize(nv);
    a_size_tmp8.resize(nv);
    a_size_tmp9.resize(nv);
    a_size_tmp10.resize(nv);
    a_size_tmp11.resize(nv);
    a_size_tmp12.resize(nv);

    num_vars_size_tmp1.resize(num_vars);
    num_vars_size_tmp2.resize(num_vars);
    num_vars_size_tmp3.resize(num_vars);

    num_vars_by_num_eq_cons_tmp.resize(num_vars, num_eq_cons);
    mass_matrix_size_tmp.resize(nv, nv);

    // Allocate sequences
    q_sequence_tmp1.assign(num_steps, ocs2::vector_t::Zero(nq));
    q_sequence_tmp2.assign(num_steps, ocs2::vector_t::Zero(nq));
  }

  // Storage for multibody forces
  pinocchio::container::aligned_vector<pinocchio::ForceTpl<T>> fext;

  // Storage of size nq
  ocs2::vector_s_t<T> q_size_tmp1;
  ocs2::vector_s_t<T> q_size_tmp2;
  ocs2::vector_s_t<T> q_size_tmp3;
  ocs2::vector_s_t<T> q_size_tmp4;

  // Storage of size nv
  // These are named v, tau, and a, but this distinction is just for
  // convienience.
  ocs2::vector_s_t<T> v_size_tmp1;
  ocs2::vector_s_t<T> v_size_tmp2;
  ocs2::vector_s_t<T> v_size_tmp3;
  ocs2::vector_s_t<T> v_size_tmp4;
  ocs2::vector_s_t<T> v_size_tmp5;
  ocs2::vector_s_t<T> v_size_tmp6;
  ocs2::vector_s_t<T> v_size_tmp7;
  ocs2::vector_s_t<T> v_size_tmp8;

  ocs2::vector_s_t<T> tau_size_tmp1;
  ocs2::vector_s_t<T> tau_size_tmp2;
  ocs2::vector_s_t<T> tau_size_tmp3;
  ocs2::vector_s_t<T> tau_size_tmp4;
  ocs2::vector_s_t<T> tau_size_tmp5;
  ocs2::vector_s_t<T> tau_size_tmp6;
  ocs2::vector_s_t<T> tau_size_tmp7;
  ocs2::vector_s_t<T> tau_size_tmp8;
  ocs2::vector_s_t<T> tau_size_tmp9;
  ocs2::vector_s_t<T> tau_size_tmp10;
  ocs2::vector_s_t<T> tau_size_tmp11;
  ocs2::vector_s_t<T> tau_size_tmp12;

  ocs2::vector_s_t<T> a_size_tmp1;
  ocs2::vector_s_t<T> a_size_tmp2;
  ocs2::vector_s_t<T> a_size_tmp3;
  ocs2::vector_s_t<T> a_size_tmp4;
  ocs2::vector_s_t<T> a_size_tmp5;
  ocs2::vector_s_t<T> a_size_tmp6;
  ocs2::vector_s_t<T> a_size_tmp7;
  ocs2::vector_s_t<T> a_size_tmp8;
  ocs2::vector_s_t<T> a_size_tmp9;
  ocs2::vector_s_t<T> a_size_tmp10;
  ocs2::vector_s_t<T> a_size_tmp11;
  ocs2::vector_s_t<T> a_size_tmp12;

  // Storage of sequence of q
  std::vector<ocs2::vector_s_t<T>> q_sequence_tmp1;
  std::vector<ocs2::vector_s_t<T>> q_sequence_tmp2;

  // Vector of all decision variables
  ocs2::vector_s_t<T> num_vars_size_tmp1;
  ocs2::vector_s_t<T> num_vars_size_tmp2;
  ocs2::vector_s_t<T> num_vars_size_tmp3;

  // Matrix of size (number of variables) * (number of equality constraints)
  ocs2::matrix_s_t<T> num_vars_by_num_eq_cons_tmp;

  // Matrix of size nv x nv, used to store the mass matrix
  ocs2::matrix_s_t<T> mass_matrix_size_tmp;
};

template struct TrajectoryOptimizerWorkspace<double>;

}  // namespace optimizer
}  // namespace idto
