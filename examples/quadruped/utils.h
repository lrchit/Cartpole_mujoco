
#pragma once

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <urdf_model/model.h>
#include <urdf_model/joint.h>
#include <urdf_parser/urdf_parser.h>

#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/multibody/model.hpp"

pinocchio::ModelTpl<ocs2::scalar_t> createPinocchioModel(const std::string& urdfFilePath, const std::vector<std::string>& jointNames) {
  using joint_pair_t = std::pair<const std::string, std::shared_ptr<::urdf::Joint>>;

  ::urdf::ModelInterfaceSharedPtr urdfTree = ::urdf::parseURDFFile(urdfFilePath);
  if (urdfTree == nullptr) {
    throw std::invalid_argument("The file " + urdfFilePath + " does not contain a valid URDF model!");
  }

  // remove extraneous joints from urdf
  ::urdf::ModelInterfaceSharedPtr newModel = std::make_shared<::urdf::ModelInterface>(*urdfTree);
  for (joint_pair_t& jointPair : newModel->joints_) {
    if (std::find(jointNames.begin(), jointNames.end(), jointPair.first) == jointNames.end()) {
      jointPair.second->type = urdf::Joint::FIXED;
    }
  }

  // add 6 DoF for the floating base
  pinocchio::JointModelComposite jointComposite(2);
  jointComposite.addJoint(pinocchio::JointModelTranslation());
  jointComposite.addJoint(pinocchio::JointModelSphericalZYX());

  pinocchio::ModelTpl<ocs2::scalar_t> model;
  pinocchio::urdf::buildModel(urdfTree, jointComposite, model);
  return model;
}