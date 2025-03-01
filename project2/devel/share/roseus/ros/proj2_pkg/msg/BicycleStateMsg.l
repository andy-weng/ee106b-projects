;; Auto-generated. Do not edit!


(when (boundp 'proj2_pkg::BicycleStateMsg)
  (if (not (find-package "PROJ2_PKG"))
    (make-package "PROJ2_PKG"))
  (shadow 'BicycleStateMsg (find-package "PROJ2_PKG")))
(unless (find-package "PROJ2_PKG::BICYCLESTATEMSG")
  (make-package "PROJ2_PKG::BICYCLESTATEMSG"))

(in-package "ROS")
;;//! \htmlinclude BicycleStateMsg.msg.html


(defclass proj2_pkg::BicycleStateMsg
  :super ros::object
  :slots (_x _y _theta _phi ))

(defmethod proj2_pkg::BicycleStateMsg
  (:init
   (&key
    ((:x __x) 0.0)
    ((:y __y) 0.0)
    ((:theta __theta) 0.0)
    ((:phi __phi) 0.0)
    )
   (send-super :init)
   (setq _x (float __x))
   (setq _y (float __y))
   (setq _theta (float __theta))
   (setq _phi (float __phi))
   self)
  (:x
   (&optional __x)
   (if __x (setq _x __x)) _x)
  (:y
   (&optional __y)
   (if __y (setq _y __y)) _y)
  (:theta
   (&optional __theta)
   (if __theta (setq _theta __theta)) _theta)
  (:phi
   (&optional __phi)
   (if __phi (setq _phi __phi)) _phi)
  (:serialization-length
   ()
   (+
    ;; float64 _x
    8
    ;; float64 _y
    8
    ;; float64 _theta
    8
    ;; float64 _phi
    8
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float64 _x
       (sys::poke _x (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _y
       (sys::poke _y (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _theta
       (sys::poke _theta (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _phi
       (sys::poke _phi (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float64 _x
     (setq _x (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _y
     (setq _y (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _theta
     (setq _theta (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _phi
     (setq _phi (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;;
   self)
  )

(setf (get proj2_pkg::BicycleStateMsg :md5sum-) "7aab3456a3691108bb0b44c570431f4c")
(setf (get proj2_pkg::BicycleStateMsg :datatype-) "proj2_pkg/BicycleStateMsg")
(setf (get proj2_pkg::BicycleStateMsg :definition-)
      "# The state of a bicycle model robot (x, y, theta, phi)
float64 x
float64 y
float64 theta
float64 phi
")



(provide :proj2_pkg/BicycleStateMsg "7aab3456a3691108bb0b44c570431f4c")


