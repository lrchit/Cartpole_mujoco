
#include <utils.h>
#include <quadruped_cost.h>
#include <quadruped_dynamics.h>
#include <direct_multiple_shooting.h>

#include <mpc.h>

int main() {
  YAML::Node config = YAML::LoadFile("../test/config/dms.yaml");

  int nq = config["nq"].as<int>();
  int nv = config["nv"].as<int>();
  int nu = config["nu"].as<int>();
  int nx = nq + nv;
  int horizon = config["mpc"]["horizon"].as<int>() + 1;

  // pinocchio model
  const std::string urdfFile = "../models/quadruped/urdf/a1.urdf";
  const std::vector<std::string> footName{"FL_foot", "FR_foot", "RL_foot", "RR_foot"};
  const std::vector<std::string> jointNames{"FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
      "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"};
  const pinocchio::ModelTpl<ocs2::scalar_t> model = createPinocchioModel(urdfFile, jointNames);
  pinocchio::DataTpl<ocs2::scalar_t> data = pinocchio::DataTpl<ocs2::scalar_t>(model);
  std::vector<pinocchio::FrameIndex> footId;
  for (int leg = 0; leg < 4; ++leg) {
    footId.push_back(model.getFrameId(footName[leg]));
  }

  std::shared_ptr<Quadruped_Dynamics> quadruped_dynamics = std::make_shared<Quadruped_Dynamics>(config, model, footId);
  std::shared_ptr<Quadruped_Cost> quadruped_cost = std::make_shared<Quadruped_Cost>(config, model, footId);

  ocs2::matrix_t K(nu, nx);
  std::vector<double> Kp = config["dms"]["Kp"].as<std::vector<double>>();
  std::vector<double> Kd = config["dms"]["Kd"].as<std::vector<double>>();
  for (int i = 0; i < nu; ++i) {
    K.middleCols(6, nu).diagonal()[i] = Kp[i];
    K.middleCols(6 + nq, nu).diagonal()[i] = Kd[i];
  }
  std::unique_ptr<DirectMultipleShooting> dms = std::make_unique<DirectMultipleShooting>(config, K, quadruped_dynamics, quadruped_cost);
  std::unique_ptr<MpcController> mpc = std::make_unique<MpcController>(config, std::move(dms));

  ocs2::vector_t xcur(nx);
  xcur.head(nq) << 0.05, 0.02, 0.4, 0, 0, 0, -0., 0.72, -1.44, 0., 0.72, -1.44, -0., 0.72, -1.44, 0., 0.72, -1.44;
  xcur.tail(nv).setZero();
  ocs2::vector_t xtarget(nx);
  xtarget.head(nq) << 0, 0, 0.3, 0, 0, 0, -0., 0.72, -1.44, 0., 0.72, -1.44, -0., 0.72, -1.44, 0., 0.72, -1.44;
  xtarget.tail(nv).setZero();
  std::vector<ocs2::vector_t> xref(horizon, xtarget);

  while (true) {
    mpc->resetProblem(xcur, xref);
  }
}
