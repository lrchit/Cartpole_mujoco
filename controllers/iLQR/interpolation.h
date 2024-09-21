
#pragma once

#include <chrono>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <boost/math/interpolators/cubic_b_spline.hpp>

struct Timer {
  using Clock = std::chrono::high_resolution_clock;
  using TimePoint = Clock::time_point;

  TimePoint start_time;

  double getCurrentTime() const { return std::chrono::duration<double>(Clock::now() - start_time).count(); }
  void reset() { start_time = Clock::now(); }
};

class TrajectoryInterpolation {
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  TrajectoryInterpolation(int nx, int nu, double dt) : nx_(nx), nu_(nu), dt_(dt) {
    state_spline_.resize(nx_);
    input_spline_.resize(nu_);
  }
  ~TrajectoryInterpolation() {}

  void updateTrajectory(const std::vector<Eigen::VectorXd>& state_nodes, const std::vector<Eigen::VectorXd>& input_nodes, const double current_time) {
    if (state_nodes.size() - 1 != input_nodes.size()) {
      throw std::invalid_argument("Size of state solutions and input solutions should match!");
    }

    // interpolate through classical cubic sline
    for (int k = 0; k < nx_; ++k) {
      std::vector<double> state;
      for (int i = 0; i < state_nodes.size(); ++i) {
        state.push_back(state_nodes[i][k]);
      }
      state_spline_[k] = boost::math::cubic_b_spline<double>(state.begin(), state.end(), current_time, dt_);
    }
    for (int k = 0; k < nu_; ++k) {
      std::vector<double> input;
      for (int i = 0; i < input_nodes.size(); ++i) {
        input.push_back(input_nodes[i][k]);
      }
      input_spline_[k] = boost::math::cubic_b_spline<double>(input.begin(), input.end(), current_time, dt_);
    }
  }

  Eigen::VectorXd getRealTimeState(double query_time) const {
    Eigen::VectorXd result;
    result.resize(nx_);
    for (int k = 0; k < nx_; ++k) {
      result[k] = state_spline_[k](query_time);
    }
    return result;
  }
  Eigen::VectorXd getRealTimeInput(double query_time) const {
    Eigen::VectorXd result;
    result.resize(nu_);
    for (int k = 0; k < nu_; ++k) {
      result[k] = input_spline_[k](query_time);
    }
    return result;
  }

  private:
  int nx_;
  int nu_;
  double dt_;
  std::vector<boost::math::cubic_b_spline<double>> state_spline_;
  std::vector<boost::math::cubic_b_spline<double>> input_spline_;
};