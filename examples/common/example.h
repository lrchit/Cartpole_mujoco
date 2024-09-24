
#pragma once

#include <mpc.h>

class Example {
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Example(std::string yaml_name) : yaml_name_(yaml_name) {
    YAML::Node config = YAML::LoadFile(yaml_name);
    nx = config["nx"].as<int>();
    nu = config["nu"].as<int>();
    Nt = config["horizon"].as<int>() + 1;
    double nu = config["nu"].as<int>();
    xcur.setZero(nx);
    xref.resize(Nt);
    for (int k = 0; k < Nt; ++k) {
      xref[k].setZero(nx);
    }
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
  std::vector<ocs2::vector_t> xref;

  std::unique_ptr<MpcController> mpc;
};