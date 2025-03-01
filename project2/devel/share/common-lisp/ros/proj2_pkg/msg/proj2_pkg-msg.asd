
(cl:in-package :asdf)

(defsystem "proj2_pkg-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "BicycleCommandMsg" :depends-on ("_package_BicycleCommandMsg"))
    (:file "_package_BicycleCommandMsg" :depends-on ("_package"))
    (:file "BicycleStateMsg" :depends-on ("_package_BicycleStateMsg"))
    (:file "_package_BicycleStateMsg" :depends-on ("_package"))
  ))