# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "stdr_msgs: 45 messages, 12 services")

set(MSG_I_FLAGS "-Istdr_msgs:/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg;-Istdr_msgs:/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg;-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg;-Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg;-Inav_msgs:/opt/ros/noetic/share/nav_msgs/cmake/../msg;-Iactionlib_msgs:/opt/ros/noetic/share/actionlib_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(stdr_msgs_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg" ""
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg" "stdr_msgs/Noise:geometry_msgs/Pose2D"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg" "stdr_msgs/Noise:geometry_msgs/Pose2D"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg" ""
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg" "geometry_msgs/Point"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg" "stdr_msgs/LaserSensorMsg:stdr_msgs/SonarSensorMsg:stdr_msgs/RfidSensorMsg:stdr_msgs/KinematicMsg:geometry_msgs/Pose2D:stdr_msgs/Noise:stdr_msgs/ThermalSensorMsg:stdr_msgs/SoundSensorMsg:geometry_msgs/Point:stdr_msgs/CO2SensorMsg:stdr_msgs/FootprintMsg"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg" "stdr_msgs/LaserSensorMsg:stdr_msgs/SonarSensorMsg:stdr_msgs/RfidSensorMsg:stdr_msgs/KinematicMsg:geometry_msgs/Pose2D:stdr_msgs/Noise:stdr_msgs/ThermalSensorMsg:stdr_msgs/SoundSensorMsg:geometry_msgs/Point:stdr_msgs/CO2SensorMsg:stdr_msgs/RobotMsg:stdr_msgs/FootprintMsg"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedVectorMsg.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedVectorMsg.msg" "stdr_msgs/LaserSensorMsg:stdr_msgs/SonarSensorMsg:stdr_msgs/RobotIndexedMsg:stdr_msgs/RfidSensorMsg:stdr_msgs/KinematicMsg:geometry_msgs/Pose2D:stdr_msgs/Noise:stdr_msgs/ThermalSensorMsg:stdr_msgs/SoundSensorMsg:geometry_msgs/Point:stdr_msgs/CO2SensorMsg:stdr_msgs/RobotMsg:stdr_msgs/FootprintMsg"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg" "geometry_msgs/Pose2D"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMeasurementMsg.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMeasurementMsg.msg" "std_msgs/Header"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTag.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTag.msg" "geometry_msgs/Pose2D"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTagVector.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTagVector.msg" "stdr_msgs/RfidTag:geometry_msgs/Pose2D"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg" "geometry_msgs/Pose2D"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMeasurementMsg.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMeasurementMsg.msg" "std_msgs/Header"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSource.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSource.msg" "geometry_msgs/Pose2D"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSourceVector.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSourceVector.msg" "stdr_msgs/SoundSource:geometry_msgs/Pose2D"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg" "geometry_msgs/Pose2D"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMeasurementMsg.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMeasurementMsg.msg" "std_msgs/Header"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSource.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSource.msg" "geometry_msgs/Pose2D"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSourceVector.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSourceVector.msg" "stdr_msgs/ThermalSource:geometry_msgs/Pose2D"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg" "geometry_msgs/Pose2D"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMeasurementMsg.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMeasurementMsg.msg" "std_msgs/Header"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2Source.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2Source.msg" "geometry_msgs/Pose2D"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SourceVector.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SourceVector.msg" "geometry_msgs/Pose2D:stdr_msgs/CO2Source"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotAction.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotAction.msg" "stdr_msgs/RegisterRobotActionGoal:stdr_msgs/KinematicMsg:stdr_msgs/ThermalSensorMsg:stdr_msgs/RobotMsg:stdr_msgs/SonarSensorMsg:stdr_msgs/RegisterRobotActionResult:actionlib_msgs/GoalID:stdr_msgs/CO2SensorMsg:stdr_msgs/RegisterRobotResult:geometry_msgs/Pose2D:stdr_msgs/SoundSensorMsg:stdr_msgs/RegisterRobotFeedback:actionlib_msgs/GoalStatus:stdr_msgs/LaserSensorMsg:std_msgs/Header:stdr_msgs/RfidSensorMsg:stdr_msgs/RegisterRobotGoal:stdr_msgs/Noise:stdr_msgs/RegisterRobotActionFeedback:geometry_msgs/Point:stdr_msgs/FootprintMsg"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionGoal.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionGoal.msg" "actionlib_msgs/GoalID:std_msgs/Header:stdr_msgs/RegisterRobotGoal"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionResult.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionResult.msg" "actionlib_msgs/GoalStatus:stdr_msgs/LaserSensorMsg:stdr_msgs/SonarSensorMsg:std_msgs/Header:stdr_msgs/RfidSensorMsg:stdr_msgs/RegisterRobotResult:stdr_msgs/KinematicMsg:geometry_msgs/Pose2D:stdr_msgs/Noise:actionlib_msgs/GoalID:stdr_msgs/ThermalSensorMsg:stdr_msgs/SoundSensorMsg:geometry_msgs/Point:stdr_msgs/CO2SensorMsg:stdr_msgs/RobotMsg:stdr_msgs/FootprintMsg"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionFeedback.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionFeedback.msg" "actionlib_msgs/GoalStatus:stdr_msgs/RegisterRobotFeedback:actionlib_msgs/GoalID:std_msgs/Header"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotGoal.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotGoal.msg" ""
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotResult.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotResult.msg" "stdr_msgs/LaserSensorMsg:stdr_msgs/SonarSensorMsg:stdr_msgs/RfidSensorMsg:stdr_msgs/KinematicMsg:geometry_msgs/Pose2D:stdr_msgs/Noise:stdr_msgs/ThermalSensorMsg:stdr_msgs/SoundSensorMsg:geometry_msgs/Point:stdr_msgs/CO2SensorMsg:stdr_msgs/RobotMsg:stdr_msgs/FootprintMsg"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotFeedback.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotFeedback.msg" ""
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotAction.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotAction.msg" "stdr_msgs/KinematicMsg:stdr_msgs/SpawnRobotActionGoal:stdr_msgs/ThermalSensorMsg:stdr_msgs/RobotMsg:stdr_msgs/SpawnRobotActionResult:stdr_msgs/SonarSensorMsg:stdr_msgs/RobotIndexedMsg:stdr_msgs/SpawnRobotGoal:actionlib_msgs/GoalID:stdr_msgs/CO2SensorMsg:stdr_msgs/SpawnRobotFeedback:geometry_msgs/Pose2D:stdr_msgs/SoundSensorMsg:stdr_msgs/SpawnRobotResult:stdr_msgs/LaserSensorMsg:actionlib_msgs/GoalStatus:std_msgs/Header:stdr_msgs/RfidSensorMsg:stdr_msgs/Noise:geometry_msgs/Point:stdr_msgs/SpawnRobotActionFeedback:stdr_msgs/FootprintMsg"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionGoal.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionGoal.msg" "stdr_msgs/LaserSensorMsg:stdr_msgs/SonarSensorMsg:std_msgs/Header:stdr_msgs/SpawnRobotGoal:stdr_msgs/RfidSensorMsg:stdr_msgs/KinematicMsg:geometry_msgs/Pose2D:stdr_msgs/Noise:actionlib_msgs/GoalID:stdr_msgs/ThermalSensorMsg:stdr_msgs/SoundSensorMsg:geometry_msgs/Point:stdr_msgs/CO2SensorMsg:stdr_msgs/RobotMsg:stdr_msgs/FootprintMsg"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionResult.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionResult.msg" "actionlib_msgs/GoalStatus:stdr_msgs/LaserSensorMsg:stdr_msgs/SonarSensorMsg:stdr_msgs/RobotIndexedMsg:std_msgs/Header:stdr_msgs/FootprintMsg:stdr_msgs/RfidSensorMsg:stdr_msgs/KinematicMsg:geometry_msgs/Pose2D:stdr_msgs/Noise:actionlib_msgs/GoalID:stdr_msgs/ThermalSensorMsg:stdr_msgs/SoundSensorMsg:geometry_msgs/Point:stdr_msgs/CO2SensorMsg:stdr_msgs/RobotMsg:stdr_msgs/SpawnRobotResult"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionFeedback.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionFeedback.msg" "actionlib_msgs/GoalStatus:stdr_msgs/SpawnRobotFeedback:actionlib_msgs/GoalID:std_msgs/Header"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotGoal.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotGoal.msg" "stdr_msgs/LaserSensorMsg:stdr_msgs/SonarSensorMsg:stdr_msgs/RfidSensorMsg:stdr_msgs/KinematicMsg:geometry_msgs/Pose2D:stdr_msgs/Noise:stdr_msgs/ThermalSensorMsg:stdr_msgs/SoundSensorMsg:geometry_msgs/Point:stdr_msgs/CO2SensorMsg:stdr_msgs/RobotMsg:stdr_msgs/FootprintMsg"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotResult.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotResult.msg" "stdr_msgs/LaserSensorMsg:stdr_msgs/SonarSensorMsg:stdr_msgs/RobotIndexedMsg:stdr_msgs/RfidSensorMsg:stdr_msgs/KinematicMsg:geometry_msgs/Pose2D:stdr_msgs/Noise:stdr_msgs/ThermalSensorMsg:stdr_msgs/SoundSensorMsg:geometry_msgs/Point:stdr_msgs/CO2SensorMsg:stdr_msgs/RobotMsg:stdr_msgs/FootprintMsg"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotFeedback.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotFeedback.msg" ""
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotAction.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotAction.msg" "stdr_msgs/DeleteRobotGoal:stdr_msgs/DeleteRobotActionResult:stdr_msgs/DeleteRobotActionGoal:actionlib_msgs/GoalStatus:std_msgs/Header:actionlib_msgs/GoalID:stdr_msgs/DeleteRobotResult:stdr_msgs/DeleteRobotActionFeedback:stdr_msgs/DeleteRobotFeedback"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionGoal.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionGoal.msg" "stdr_msgs/DeleteRobotGoal:actionlib_msgs/GoalID:std_msgs/Header"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionResult.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionResult.msg" "actionlib_msgs/GoalStatus:actionlib_msgs/GoalID:std_msgs/Header:stdr_msgs/DeleteRobotResult"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionFeedback.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionFeedback.msg" "actionlib_msgs/GoalStatus:actionlib_msgs/GoalID:std_msgs/Header:stdr_msgs/DeleteRobotFeedback"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotGoal.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotGoal.msg" ""
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotResult.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotResult.msg" ""
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotFeedback.msg" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotFeedback.msg" ""
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/LoadMap.srv" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/LoadMap.srv" ""
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/LoadExternalMap.srv" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/LoadExternalMap.srv" "std_msgs/Header:geometry_msgs/Quaternion:geometry_msgs/Pose:geometry_msgs/Point:nav_msgs/OccupancyGrid:nav_msgs/MapMetaData"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/RegisterGui.srv" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/RegisterGui.srv" "stdr_msgs/LaserSensorMsg:stdr_msgs/SonarSensorMsg:stdr_msgs/RobotIndexedMsg:stdr_msgs/RfidSensorMsg:stdr_msgs/KinematicMsg:geometry_msgs/Pose2D:stdr_msgs/Noise:stdr_msgs/ThermalSensorMsg:stdr_msgs/SoundSensorMsg:geometry_msgs/Point:stdr_msgs/CO2SensorMsg:stdr_msgs/RobotMsg:stdr_msgs/FootprintMsg"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/MoveRobot.srv" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/MoveRobot.srv" "geometry_msgs/Pose2D"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddRfidTag.srv" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddRfidTag.srv" "stdr_msgs/RfidTag:geometry_msgs/Pose2D"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteRfidTag.srv" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteRfidTag.srv" ""
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddThermalSource.srv" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddThermalSource.srv" "stdr_msgs/ThermalSource:geometry_msgs/Pose2D"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteThermalSource.srv" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteThermalSource.srv" ""
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddSoundSource.srv" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddSoundSource.srv" "stdr_msgs/SoundSource:geometry_msgs/Pose2D"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteSoundSource.srv" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteSoundSource.srv" ""
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddCO2Source.srv" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddCO2Source.srv" "geometry_msgs/Pose2D:stdr_msgs/CO2Source"
)

get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteCO2Source.srv" NAME_WE)
add_custom_target(_stdr_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "stdr_msgs" "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteCO2Source.srv" ""
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedVectorMsg.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMeasurementMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTag.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTagVector.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTag.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMeasurementMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSource.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSourceVector.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSource.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMeasurementMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSource.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSourceVector.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSource.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMeasurementMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2Source.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SourceVector.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2Source.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotAction.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionGoal.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionResult.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotResult.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotFeedback.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotGoal.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionFeedback.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionGoal.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotGoal.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionResult.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotResult.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotFeedback.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotGoal.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotResult.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotFeedback.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotAction.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionGoal.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionResult.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotGoal.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotFeedback.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotResult.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionFeedback.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionGoal.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotGoal.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionResult.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotResult.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotFeedback.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotGoal.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotResult.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotFeedback.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotAction.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotGoal.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionResult.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionGoal.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotResult.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionFeedback.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotFeedback.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionGoal.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotGoal.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionResult.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotResult.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotFeedback.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotGoal.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotResult.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotFeedback.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)

### Generating Services
_generate_srv_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/LoadMap.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_srv_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/LoadExternalMap.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/nav_msgs/cmake/../msg/OccupancyGrid.msg;/opt/ros/noetic/share/nav_msgs/cmake/../msg/MapMetaData.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_srv_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/RegisterGui.srv"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_srv_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/MoveRobot.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_srv_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddRfidTag.srv"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTag.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_srv_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteRfidTag.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_srv_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddThermalSource.srv"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSource.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_srv_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteThermalSource.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_srv_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddSoundSource.srv"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSource.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_srv_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteSoundSource.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_srv_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddCO2Source.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2Source.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)
_generate_srv_cpp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteCO2Source.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
)

### Generating Module File
_generate_module_cpp(stdr_msgs
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(stdr_msgs_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(stdr_msgs_generate_messages stdr_msgs_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedVectorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMeasurementMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTag.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTagVector.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMeasurementMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSource.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSourceVector.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMeasurementMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSource.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSourceVector.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMeasurementMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2Source.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SourceVector.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotAction.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionGoal.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionResult.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionFeedback.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotGoal.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotResult.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotFeedback.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotAction.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionGoal.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionResult.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionFeedback.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotGoal.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotResult.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotFeedback.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotAction.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionGoal.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionResult.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionFeedback.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotGoal.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotResult.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotFeedback.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/LoadMap.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/LoadExternalMap.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/RegisterGui.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/MoveRobot.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddRfidTag.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteRfidTag.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddThermalSource.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteThermalSource.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddSoundSource.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteSoundSource.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddCO2Source.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteCO2Source.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_cpp _stdr_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(stdr_msgs_gencpp)
add_dependencies(stdr_msgs_gencpp stdr_msgs_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS stdr_msgs_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedVectorMsg.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMeasurementMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTag.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTagVector.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTag.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMeasurementMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSource.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSourceVector.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSource.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMeasurementMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSource.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSourceVector.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSource.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMeasurementMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2Source.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SourceVector.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2Source.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotAction.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionGoal.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionResult.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotResult.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotFeedback.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotGoal.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionFeedback.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionGoal.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotGoal.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionResult.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotResult.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotFeedback.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotGoal.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotResult.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotFeedback.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotAction.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionGoal.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionResult.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotGoal.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotFeedback.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotResult.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionFeedback.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionGoal.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotGoal.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionResult.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotResult.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotFeedback.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotGoal.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotResult.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotFeedback.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotAction.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotGoal.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionResult.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionGoal.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotResult.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionFeedback.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotFeedback.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionGoal.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotGoal.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionResult.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotResult.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotFeedback.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotGoal.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotResult.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_msg_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotFeedback.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)

### Generating Services
_generate_srv_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/LoadMap.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_srv_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/LoadExternalMap.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/nav_msgs/cmake/../msg/OccupancyGrid.msg;/opt/ros/noetic/share/nav_msgs/cmake/../msg/MapMetaData.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_srv_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/RegisterGui.srv"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_srv_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/MoveRobot.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_srv_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddRfidTag.srv"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTag.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_srv_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteRfidTag.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_srv_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddThermalSource.srv"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSource.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_srv_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteThermalSource.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_srv_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddSoundSource.srv"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSource.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_srv_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteSoundSource.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_srv_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddCO2Source.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2Source.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)
_generate_srv_eus(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteCO2Source.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
)

### Generating Module File
_generate_module_eus(stdr_msgs
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(stdr_msgs_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(stdr_msgs_generate_messages stdr_msgs_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedVectorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMeasurementMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTag.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTagVector.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMeasurementMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSource.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSourceVector.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMeasurementMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSource.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSourceVector.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMeasurementMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2Source.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SourceVector.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotAction.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionGoal.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionResult.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionFeedback.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotGoal.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotResult.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotFeedback.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotAction.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionGoal.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionResult.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionFeedback.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotGoal.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotResult.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotFeedback.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotAction.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionGoal.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionResult.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionFeedback.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotGoal.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotResult.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotFeedback.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/LoadMap.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/LoadExternalMap.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/RegisterGui.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/MoveRobot.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddRfidTag.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteRfidTag.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddThermalSource.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteThermalSource.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddSoundSource.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteSoundSource.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddCO2Source.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteCO2Source.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_eus _stdr_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(stdr_msgs_geneus)
add_dependencies(stdr_msgs_geneus stdr_msgs_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS stdr_msgs_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedVectorMsg.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMeasurementMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTag.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTagVector.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTag.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMeasurementMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSource.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSourceVector.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSource.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMeasurementMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSource.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSourceVector.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSource.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMeasurementMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2Source.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SourceVector.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2Source.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotAction.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionGoal.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionResult.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotResult.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotFeedback.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotGoal.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionFeedback.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionGoal.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotGoal.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionResult.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotResult.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotFeedback.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotGoal.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotResult.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotFeedback.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotAction.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionGoal.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionResult.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotGoal.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotFeedback.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotResult.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionFeedback.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionGoal.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotGoal.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionResult.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotResult.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotFeedback.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotGoal.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotResult.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotFeedback.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotAction.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotGoal.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionResult.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionGoal.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotResult.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionFeedback.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotFeedback.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionGoal.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotGoal.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionResult.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotResult.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotFeedback.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotGoal.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotResult.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_msg_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotFeedback.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)

### Generating Services
_generate_srv_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/LoadMap.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_srv_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/LoadExternalMap.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/nav_msgs/cmake/../msg/OccupancyGrid.msg;/opt/ros/noetic/share/nav_msgs/cmake/../msg/MapMetaData.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_srv_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/RegisterGui.srv"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_srv_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/MoveRobot.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_srv_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddRfidTag.srv"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTag.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_srv_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteRfidTag.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_srv_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddThermalSource.srv"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSource.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_srv_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteThermalSource.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_srv_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddSoundSource.srv"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSource.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_srv_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteSoundSource.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_srv_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddCO2Source.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2Source.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)
_generate_srv_lisp(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteCO2Source.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
)

### Generating Module File
_generate_module_lisp(stdr_msgs
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(stdr_msgs_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(stdr_msgs_generate_messages stdr_msgs_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedVectorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMeasurementMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTag.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTagVector.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMeasurementMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSource.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSourceVector.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMeasurementMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSource.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSourceVector.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMeasurementMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2Source.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SourceVector.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotAction.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionGoal.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionResult.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionFeedback.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotGoal.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotResult.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotFeedback.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotAction.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionGoal.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionResult.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionFeedback.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotGoal.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotResult.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotFeedback.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotAction.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionGoal.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionResult.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionFeedback.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotGoal.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotResult.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotFeedback.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/LoadMap.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/LoadExternalMap.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/RegisterGui.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/MoveRobot.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddRfidTag.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteRfidTag.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddThermalSource.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteThermalSource.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddSoundSource.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteSoundSource.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddCO2Source.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteCO2Source.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_lisp _stdr_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(stdr_msgs_genlisp)
add_dependencies(stdr_msgs_genlisp stdr_msgs_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS stdr_msgs_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedVectorMsg.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMeasurementMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTag.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTagVector.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTag.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMeasurementMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSource.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSourceVector.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSource.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMeasurementMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSource.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSourceVector.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSource.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMeasurementMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2Source.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SourceVector.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2Source.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotAction.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionGoal.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionResult.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotResult.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotFeedback.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotGoal.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionFeedback.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionGoal.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotGoal.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionResult.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotResult.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotFeedback.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotGoal.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotResult.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotFeedback.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotAction.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionGoal.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionResult.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotGoal.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotFeedback.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotResult.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionFeedback.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionGoal.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotGoal.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionResult.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotResult.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotFeedback.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotGoal.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotResult.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotFeedback.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotAction.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotGoal.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionResult.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionGoal.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotResult.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionFeedback.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotFeedback.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionGoal.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotGoal.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionResult.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotResult.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotFeedback.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotGoal.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotResult.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_msg_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotFeedback.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)

### Generating Services
_generate_srv_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/LoadMap.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_srv_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/LoadExternalMap.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/nav_msgs/cmake/../msg/OccupancyGrid.msg;/opt/ros/noetic/share/nav_msgs/cmake/../msg/MapMetaData.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_srv_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/RegisterGui.srv"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_srv_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/MoveRobot.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_srv_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddRfidTag.srv"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTag.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_srv_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteRfidTag.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_srv_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddThermalSource.srv"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSource.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_srv_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteThermalSource.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_srv_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddSoundSource.srv"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSource.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_srv_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteSoundSource.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_srv_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddCO2Source.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2Source.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)
_generate_srv_nodejs(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteCO2Source.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
)

### Generating Module File
_generate_module_nodejs(stdr_msgs
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(stdr_msgs_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(stdr_msgs_generate_messages stdr_msgs_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedVectorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMeasurementMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTag.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTagVector.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMeasurementMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSource.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSourceVector.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMeasurementMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSource.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSourceVector.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMeasurementMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2Source.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SourceVector.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotAction.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionGoal.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionResult.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionFeedback.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotGoal.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotResult.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotFeedback.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotAction.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionGoal.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionResult.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionFeedback.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotGoal.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotResult.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotFeedback.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotAction.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionGoal.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionResult.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionFeedback.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotGoal.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotResult.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotFeedback.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/LoadMap.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/LoadExternalMap.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/RegisterGui.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/MoveRobot.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddRfidTag.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteRfidTag.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddThermalSource.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteThermalSource.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddSoundSource.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteSoundSource.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddCO2Source.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteCO2Source.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_nodejs _stdr_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(stdr_msgs_gennodejs)
add_dependencies(stdr_msgs_gennodejs stdr_msgs_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS stdr_msgs_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedVectorMsg.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMeasurementMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTag.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTagVector.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTag.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMeasurementMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSource.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSourceVector.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSource.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMeasurementMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSource.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSourceVector.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSource.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMeasurementMsg.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2Source.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SourceVector.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2Source.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotAction.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionGoal.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionResult.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotResult.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotFeedback.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotGoal.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionFeedback.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionGoal.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotGoal.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionResult.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotResult.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotFeedback.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotGoal.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotResult.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotFeedback.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotAction.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionGoal.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionResult.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotGoal.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotFeedback.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotResult.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionFeedback.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionGoal.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotGoal.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionResult.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotResult.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotFeedback.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotGoal.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotResult.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotFeedback.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotAction.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotGoal.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionResult.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionGoal.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotResult.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionFeedback.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotFeedback.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionGoal.msg"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotGoal.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionResult.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotResult.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotFeedback.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotGoal.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotResult.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_msg_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotFeedback.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)

### Generating Services
_generate_srv_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/LoadMap.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_srv_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/LoadExternalMap.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/nav_msgs/cmake/../msg/OccupancyGrid.msg;/opt/ros/noetic/share/nav_msgs/cmake/../msg/MapMetaData.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_srv_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/RegisterGui.srv"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_srv_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/MoveRobot.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_srv_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddRfidTag.srv"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTag.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_srv_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteRfidTag.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_srv_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddThermalSource.srv"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSource.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_srv_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteThermalSource.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_srv_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddSoundSource.srv"
  "${MSG_I_FLAGS}"
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSource.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_srv_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteSoundSource.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_srv_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddCO2Source.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2Source.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)
_generate_srv_py(stdr_msgs
  "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteCO2Source.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
)

### Generating Module File
_generate_module_py(stdr_msgs
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(stdr_msgs_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(stdr_msgs_generate_messages stdr_msgs_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/Noise.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/LaserSensorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SonarSensorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/KinematicMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/FootprintMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RobotIndexedVectorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidSensorMeasurementMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTag.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/RfidTagVector.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSensorMeasurementMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSource.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/SoundSourceVector.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSensorMeasurementMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSource.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/ThermalSourceVector.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SensorMeasurementMsg.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2Source.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/msg/CO2SourceVector.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotAction.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionGoal.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionResult.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotActionFeedback.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotGoal.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotResult.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/RegisterRobotFeedback.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotAction.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionGoal.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionResult.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotActionFeedback.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotGoal.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotResult.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/SpawnRobotFeedback.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotAction.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionGoal.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionResult.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotActionFeedback.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotGoal.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotResult.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/devel/share/stdr_msgs/msg/DeleteRobotFeedback.msg" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/LoadMap.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/LoadExternalMap.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/RegisterGui.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/MoveRobot.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddRfidTag.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteRfidTag.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddThermalSource.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteThermalSource.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddSoundSource.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteSoundSource.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/AddCO2Source.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/cc/ee106b/sp25/class/ee106b-aaj/workspace/project2/src/stdr_simulator/stdr_msgs/srv/DeleteCO2Source.srv" NAME_WE)
add_dependencies(stdr_msgs_generate_messages_py _stdr_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(stdr_msgs_genpy)
add_dependencies(stdr_msgs_genpy stdr_msgs_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS stdr_msgs_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/stdr_msgs
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(stdr_msgs_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()
if(TARGET geometry_msgs_generate_messages_cpp)
  add_dependencies(stdr_msgs_generate_messages_cpp geometry_msgs_generate_messages_cpp)
endif()
if(TARGET nav_msgs_generate_messages_cpp)
  add_dependencies(stdr_msgs_generate_messages_cpp nav_msgs_generate_messages_cpp)
endif()
if(TARGET actionlib_msgs_generate_messages_cpp)
  add_dependencies(stdr_msgs_generate_messages_cpp actionlib_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/stdr_msgs
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(stdr_msgs_generate_messages_eus std_msgs_generate_messages_eus)
endif()
if(TARGET geometry_msgs_generate_messages_eus)
  add_dependencies(stdr_msgs_generate_messages_eus geometry_msgs_generate_messages_eus)
endif()
if(TARGET nav_msgs_generate_messages_eus)
  add_dependencies(stdr_msgs_generate_messages_eus nav_msgs_generate_messages_eus)
endif()
if(TARGET actionlib_msgs_generate_messages_eus)
  add_dependencies(stdr_msgs_generate_messages_eus actionlib_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/stdr_msgs
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(stdr_msgs_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()
if(TARGET geometry_msgs_generate_messages_lisp)
  add_dependencies(stdr_msgs_generate_messages_lisp geometry_msgs_generate_messages_lisp)
endif()
if(TARGET nav_msgs_generate_messages_lisp)
  add_dependencies(stdr_msgs_generate_messages_lisp nav_msgs_generate_messages_lisp)
endif()
if(TARGET actionlib_msgs_generate_messages_lisp)
  add_dependencies(stdr_msgs_generate_messages_lisp actionlib_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/stdr_msgs
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(stdr_msgs_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()
if(TARGET geometry_msgs_generate_messages_nodejs)
  add_dependencies(stdr_msgs_generate_messages_nodejs geometry_msgs_generate_messages_nodejs)
endif()
if(TARGET nav_msgs_generate_messages_nodejs)
  add_dependencies(stdr_msgs_generate_messages_nodejs nav_msgs_generate_messages_nodejs)
endif()
if(TARGET actionlib_msgs_generate_messages_nodejs)
  add_dependencies(stdr_msgs_generate_messages_nodejs actionlib_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs)
  install(CODE "execute_process(COMMAND \"/usr/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/stdr_msgs
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(stdr_msgs_generate_messages_py std_msgs_generate_messages_py)
endif()
if(TARGET geometry_msgs_generate_messages_py)
  add_dependencies(stdr_msgs_generate_messages_py geometry_msgs_generate_messages_py)
endif()
if(TARGET nav_msgs_generate_messages_py)
  add_dependencies(stdr_msgs_generate_messages_py nav_msgs_generate_messages_py)
endif()
if(TARGET actionlib_msgs_generate_messages_py)
  add_dependencies(stdr_msgs_generate_messages_py actionlib_msgs_generate_messages_py)
endif()
