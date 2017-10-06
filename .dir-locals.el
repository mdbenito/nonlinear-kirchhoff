((nil . (
         ;; Number of jobs (-j nn) should already be set with MAKEFLAGS in .bashrc
         (compile-command . "cd build && make -j 12 -k")
         ;; Use a wrapper around cmake which sets FEniCS' environment variables
         (cmake-ide-cmake-command . "/home/fenics/bin/fenics-cmake")
         (cmake-ide-build-dir . "build")
         (c-basic-offset . 2)
         (c-default-style . "linux"))))
