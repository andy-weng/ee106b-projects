; Auto-generated. Do not edit!


(cl:in-package proj2_pkg-msg)


;//! \htmlinclude BicycleCommandMsg.msg.html

(cl:defclass <BicycleCommandMsg> (roslisp-msg-protocol:ros-message)
  ((linear_velocity
    :reader linear_velocity
    :initarg :linear_velocity
    :type cl:float
    :initform 0.0)
   (steering_rate
    :reader steering_rate
    :initarg :steering_rate
    :type cl:float
    :initform 0.0))
)

(cl:defclass BicycleCommandMsg (<BicycleCommandMsg>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <BicycleCommandMsg>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'BicycleCommandMsg)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name proj2_pkg-msg:<BicycleCommandMsg> is deprecated: use proj2_pkg-msg:BicycleCommandMsg instead.")))

(cl:ensure-generic-function 'linear_velocity-val :lambda-list '(m))
(cl:defmethod linear_velocity-val ((m <BicycleCommandMsg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader proj2_pkg-msg:linear_velocity-val is deprecated.  Use proj2_pkg-msg:linear_velocity instead.")
  (linear_velocity m))

(cl:ensure-generic-function 'steering_rate-val :lambda-list '(m))
(cl:defmethod steering_rate-val ((m <BicycleCommandMsg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader proj2_pkg-msg:steering_rate-val is deprecated.  Use proj2_pkg-msg:steering_rate instead.")
  (steering_rate m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <BicycleCommandMsg>) ostream)
  "Serializes a message object of type '<BicycleCommandMsg>"
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'linear_velocity))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'steering_rate))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <BicycleCommandMsg>) istream)
  "Deserializes a message object of type '<BicycleCommandMsg>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'linear_velocity) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'steering_rate) (roslisp-utils:decode-double-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<BicycleCommandMsg>)))
  "Returns string type for a message object of type '<BicycleCommandMsg>"
  "proj2_pkg/BicycleCommandMsg")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'BicycleCommandMsg)))
  "Returns string type for a message object of type 'BicycleCommandMsg"
  "proj2_pkg/BicycleCommandMsg")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<BicycleCommandMsg>)))
  "Returns md5sum for a message object of type '<BicycleCommandMsg>"
  "98601f97a39c5d728d155ceb909428fe")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'BicycleCommandMsg)))
  "Returns md5sum for a message object of type 'BicycleCommandMsg"
  "98601f97a39c5d728d155ceb909428fe")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<BicycleCommandMsg>)))
  "Returns full string definition for message of type '<BicycleCommandMsg>"
  (cl:format cl:nil "# The commands to a bicycle model robot (v, phi)~%float64 linear_velocity~%float64 steering_rate~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'BicycleCommandMsg)))
  "Returns full string definition for message of type 'BicycleCommandMsg"
  (cl:format cl:nil "# The commands to a bicycle model robot (v, phi)~%float64 linear_velocity~%float64 steering_rate~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <BicycleCommandMsg>))
  (cl:+ 0
     8
     8
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <BicycleCommandMsg>))
  "Converts a ROS message object to a list"
  (cl:list 'BicycleCommandMsg
    (cl:cons ':linear_velocity (linear_velocity msg))
    (cl:cons ':steering_rate (steering_rate msg))
))
