
#pragma once

#include <mpc.h>

class Example {
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Example(std::string yaml_name) : yaml_name_(yaml_name) {
    YAML::Node config = YAML::LoadFile(yaml_name);
    nx = config["nx"].as<int>();
    nu = config["nu"].as<int>();
    Nt = config["mpc"]["horizon"].as<int>() + 1;
    xcur.setZero(nx);
    xtarget.setZero(nx);
  }

  virtual ~Example() = default;

  virtual void computeInput(mjData* d) = 0;

  virtual void load_initial_state(mjData* d) = 0;

  protected:
  int Nt;
  int nx;
  int nu;

  std::string yaml_name_;

  ocs2::vector_t xcur;
  ocs2::vector_t xtarget;

  std::unique_ptr<MpcController> mpc;
};