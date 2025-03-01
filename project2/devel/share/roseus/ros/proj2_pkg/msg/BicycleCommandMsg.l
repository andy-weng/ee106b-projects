;; Auto-generated. Do not edit!


(when (boundp 'proj2_pkg::BicycleCommandMsg)
  (if (not (find-package "PROJ2_PKG"))
    (make-package "PROJ2_PKG"))
  (shadow 'BicycleCommandMsg (find-package "PROJ2_PKG")))
(unless (find-package "PROJ2_PKG::BICYCLECOMMANDMSG")
  (make-package "PROJ2_PKG::BICYCLECOMMANDMSG"))

(in-package "ROS")
;;//! \htmlinclude BicycleCommandMsg.msg.html


(defclass proj2_pkg::BicycleCommandMsg
  :super ros::object
  :slots (_linear_velocity _steering_rate ))

(defmethod proj2_pkg::BicycleCommandMsg
  (:init
   (&key
    ((:linear_velocity __linear_velocity) 0.0)
    ((:steering_rate __steering_rate) 0.0)
    )
   (send-super :init)
   (setq _linear_velocity (float __linear_velocity))
   (setq _steering_rate (float __steering_rate))
   self)
  (:linear_velocity
   (&optional __linear_velocity)
   (if __linear_velocity (setq _linear_velocity __linear_velocity)) _linear_velocity)
  (:steering_rate
   (&optional __steering_rate)
   (if __steering_rate (setq _steering_rate __steering_rate)) _steering_rate)
  (:serialization-length
   ()
   (+
    ;; float64 _linear_velocity
    8
    ;; float64 _steering_rate
    8
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float64 _linear_velocity
       (sys::poke _linear_velocity (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _steering_rate
       (sys::poke _steering_rate (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float64 _linear_velocity
     (setq _linear_velocity (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _steering_rate
     (setq _steering_rate (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;;
   self)
  )

(setf (get proj2_pkg::BicycleCommandMsg :md5sum-) "98601f97a39c5d728d155ceb909428fe")
(setf (get proj2_pkg::BicycleCommandMsg :datatype-) "proj2_pkg/BicycleCommandMsg")
(setf (get proj2_pkg::BicycleCommandMsg :definition-)
      "# The commands to a bicycle model robot (v, phi)
float64 linear_velocity
float64 steering_rate
")



(provide :proj2_pkg/BicycleCommandMsg "98601f97a39c5d728d155ceb909428fe")


