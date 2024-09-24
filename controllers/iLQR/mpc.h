
#pragma once

#include <iostream>
#include <vector>
#include <mutex>
#include <thread>

#include <iLQR.h>
#include <interpolation.h>

class MpcController {
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  MpcController(YAML::Node config, std::unique_ptr<iLQR_Solver> iLQR) : iLQR_(std::move(iLQR)) {
    nx_ = config["nx"].as<int>();
    nu_ = config["nu"].as<int>();
    mpcfrequency_ = config["mpcFrequency"].as<double>();
    mrtFrequency_ = config["mrtFrequency"].as<double>();
    useFeedbackPolicy_ = config["useFeedbackPolicy"].as<bool>();
    double dt = config["dt"].as<double>();
    int Nt = config["horizon"].as<int>() + 1;

    timer.reset();
    interpolator_.reset(new TrajectoryInterpolation(nx_, nu_, dt));

    // reset xcur_, xref_, K_
    xcur_.setZero(nx_);
    for (int k = 0; k < Nt; ++k) {
      xref_.push_back(ocs2::vector_t::Zero(nx_));
    }
    K_.setZero(nu_, nx_);

    // launch mpc and mrt thread
    mpc_thread_ = std::thread(&MpcController::callMpcSolver, this);
    mrt_thread_ = std::thread(&MpcController::calSolution, this);
  }
  ~MpcController() {
    stop_thread_ = true;  // signal the thread to stop
    if (mpc_thread_.joinable()) {
      mpc_thread_.join();
    }
    if (mrt_thread_.joinable()) {
      mrt_thread_.join();
    }
  }

  void callMpcSolver() {
    while (!stop_thread_) {
      const double start_time = timer.getCurrentTime();
      if (start_time > mpcWaitingTime_) {
        // call iLQR
        iLQR_->iLQR_algorithm(xcur_, xref_);

        // update interpolator
        {
          const std::vector<ocs2::vector_t> xtraj = iLQR_->getStateTrajectory();
          const std::vector<ocs2::vector_t> utraj = iLQR_->getInputTrajectory();
          K_ = iLQR_->getFeedBackMatrix();
          // std::unique_lock<std::mutex> lock(mutex_);  // Manually lock the mutex
          interpolator_->updateTrajectory(xtraj, utraj, start_time);
          // lock.unlock();  // Manually unlock the mutex
          if (!firstMpcSolved_) {
            firstMpcSolved_ = true;
          }
        }

        // sleep if mpc is too fast
        const double duration_time = timer.getCurrentTime() - start_time;
        if (duration_time < 1 / mpcfrequency_) {
          const std::chrono::duration<double> interval(1.0 / mpcfrequency_ - duration_time);
          std::this_thread::sleep_for(interval);
        }  // compute for next solution immediately if it's too slow
      }
    }
  }

  void resetProblem(const ocs2::vector_t& xcur, const std::vector<ocs2::vector_t>& xref) {
    xcur_ = xcur;
    xref_ = xref;
  }

  void calSolution() {
    while (!stop_thread_) {
      const double start_time = timer.getCurrentTime();

      if (firstMpcSolved_) {
        const double current_time = timer.getCurrentTime();
        const ocs2::vector_t xdes = interpolator_->getRealTimeState(current_time);
        const ocs2::vector_t udes = interpolator_->getRealTimeInput(current_time);
        if (useFeedbackPolicy_) {
          std::unique_lock<std::mutex> lock(mutex_);  // Manually lock the mutex
          udes_ = udes + K_ * (xdes - xcur_);
          lock.unlock();  // Manually unlock the mutex
        } else {
          udes_ = udes;
        }
      } else {
        udes_ = ocs2::vector_t::Zero(nu_);
      }

      // sleep if mrt is too fast
      const double duration_time = timer.getCurrentTime() - start_time;
      if (duration_time < 1 / mrtFrequency_) {
        const std::chrono::duration<double> interval(1.0 / mrtFrequency_ - duration_time);
        std::this_thread::sleep_for(interval);
      }  // compute for next solution immediately if it's too slow
      // std::cerr << "mrt time = " << timer.getCurrentTime() - start_time << std::endl;
    }
  }

  ocs2::vector_t getCommand() { return udes_; }

  private:
  int nx_;
  int nu_;
  double mpcfrequency_;
  double mrtFrequency_;
  bool useFeedbackPolicy_;

  bool firstMpcSolved_ = false;

  ocs2::vector_t xcur_;
  std::vector<ocs2::vector_t> xref_;
  ocs2::matrix_t K_;
  ocs2::vector_t udes_;

  ocs2::scalar_t mpcWaitingTime_ = 1.0;

  Timer timer;
  std::unique_ptr<iLQR_Solver> iLQR_;
  std::unique_ptr<TrajectoryInterpolation> interpolator_;

  std::mutex mutex_;
  std::thread mpc_thread_;
  std::thread mrt_thread_;
  std::atomic<bool> stop_thread_;  // Atomic flag to control thread exit
};
