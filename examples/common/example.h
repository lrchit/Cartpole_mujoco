
#pragma once

#include "iLQR.h"

#include <cartpole_cost.h>
#include <cartpole_dynamics.h>
#include <mpc.h>

class Example {
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Example(std::string yaml_name) {
    YAML::Node config = YAML::LoadFile(yaml_name);
    nx = config["nx"].as<int>();
    Nt = config["horizon"].as<int>() + 1;
    double nu = config["nu"].as<int>();
    xcur.setZero(nx);
    x_goal.resize(Nt);
    for (int k = 0; k < Nt; ++k) {
      x_goal[k].setZero(nx);
    }
  }

  virtual ~Example() = default;

  virtual void get_control(mjData* d) {
    int waiting_time = 100;
    static int counter = 0;

    if (counter < waiting_time) {
      counter++;
    } else {
      computeInput(d);
    }
  }

  protected:
  virtual void computeInput(mjData* d) = 0;

  int Nt;
  int nx;

  ocs2::vector_t xcur;
  std::vector<ocs2::vector_t> x_goal;

  std::unique_ptr<mpc_controller> mpc;
};