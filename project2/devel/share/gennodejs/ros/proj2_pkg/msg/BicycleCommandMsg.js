// Auto-generated. Do not edit!

// (in-package proj2_pkg.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class BicycleCommandMsg {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.linear_velocity = null;
      this.steering_rate = null;
    }
    else {
      if (initObj.hasOwnProperty('linear_velocity')) {
        this.linear_velocity = initObj.linear_velocity
      }
      else {
        this.linear_velocity = 0.0;
      }
      if (initObj.hasOwnProperty('steering_rate')) {
        this.steering_rate = initObj.steering_rate
      }
      else {
        this.steering_rate = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type BicycleCommandMsg
    // Serialize message field [linear_velocity]
    bufferOffset = _serializer.float64(obj.linear_velocity, buffer, bufferOffset);
    // Serialize message field [steering_rate]
    bufferOffset = _serializer.float64(obj.steering_rate, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type BicycleCommandMsg
    let len;
    let data = new BicycleCommandMsg(null);
    // Deserialize message field [linear_velocity]
    data.linear_velocity = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [steering_rate]
    data.steering_rate = _deserializer.float64(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 16;
  }

  static datatype() {
    // Returns string type for a message object
    return 'proj2_pkg/BicycleCommandMsg';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '98601f97a39c5d728d155ceb909428fe';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # The commands to a bicycle model robot (v, phi)
    float64 linear_velocity
    float64 steering_rate
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new BicycleCommandMsg(null);
    if (msg.linear_velocity !== undefined) {
      resolved.linear_velocity = msg.linear_velocity;
    }
    else {
      resolved.linear_velocity = 0.0
    }

    if (msg.steering_rate !== undefined) {
      resolved.steering_rate = msg.steering_rate;
    }
    else {
      resolved.steering_rate = 0.0
    }

    return resolved;
    }
};

module.exports = BicycleCommandMsg;
