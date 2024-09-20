
#pragma once

#include "iLQR.h"
#include <cartpole_dynamics.h>

class Cartpole_Example {
  public:
  Cartpole_Example(YAML::Node config) {
    step = config["step"].as<double>();
    dt = config["dt"].as<double>();
    double Tfinal = config["Tfinal"].as<double>();
    Nt = (int)(Tfinal / dt) + 1;
    std::shared_ptr<Cartpole_Dynamics> cartpole_dynamics;
    cartpole_dynamics.reset(new Cartpole_Dynamics(dt, config));
    iLQR.reset(new iLQR_Solver(config, cartpole_dynamics));

    utraj.resize(Nt - 1);
    for (int i = 0; i < Nt - 1; ++i) {
      utraj[i].setZero(1);
    }
  }
  ~Cartpole_Example() {}

  void get_control(mjData* d) {
    int waiting_time = 100;
    static int counter = 0;
    static int index = 0;

    if (counter < waiting_time) {
      counter++;
    } else if ((counter - waiting_time) % (int)(step / 0.002) == 0) {
      // std::cout << "********** iLQR *********" << std::endl;
      ocs2::vector_t _xcur(4);
      std::vector<ocs2::vector_t> _x_goal(Nt);
      _xcur << d->sensordata[0], d->sensordata[1], d->sensordata[2], d->sensordata[3];
      for (int k = 0; k < Nt; ++k) {
        _x_goal[k].setZero(4);
      }
      utraj = iLQR->iLQR_algorithm(_xcur, _x_goal);
      index = 0;
      counter++;
    } else {
      // 设置控制力
      d->ctrl[0] = fmin(fmax(utraj[index].value(), -100), 100);
      d->ctrl[1] = 0;  // pole没有直接控制
      counter++;
      if (counter % (int)(dt / 0.002) == 0) {
        index++;
      }
    }
  }

  private:
  std::unique_ptr<iLQR_Solver> iLQR;

  double step;
  int Nt;
  double dt;

  std::vector<ocs2::vector_t> utraj;
};