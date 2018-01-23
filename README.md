# Nonlinear Kirchhoff plate model

This is a C++ implementation in [FEniCS](www.fenicsproject.org) of the
nonlinear Kirchhoff model using a non-conforming implementation with
Discrete Kirchhoff Triangles as described in [1].

Note that there is a typo in the discrete equations in p. 521: the top
block of the right hand side should read $ - \alpha T^t S T Y^n + F$.

## To do

1. The problem with skewed solutions has reappeared! This means
   that **solutions are wrong**.
2. Parallel operation across all classes. Check `IsometryConstraint`,
   `KirchhoffAssembler` and `DKTGradient` in particular.

Remember to check `DiscreteOperators.h` in the dolfin sources for some
ideas.

## Dependencies

These are bundled in the docker image which I haven't pushed to
dockerhub yet because it needs cleaning and fine tuning. Basically it
amounts to:

* The base fenics-dev image.
* My implementations of Hermite and DKT elements for FIAT and FFC.

## Building and running

The usual:
```
mkdir -p build && cd build && cmake .. && make -j <num_processes>
```

Then check the options with
```
./nonlinear_kirchhoff --help

Nonlinear Kirchhoff model on the unit square.

-h,           --help  Show this help
-v,        --verbose  Debug verbosity level, 0: no output, ... 3: output vect-
                      ors and matrices [0]
-m,   --num_vertical  number of vertical subdivisions [4]
-n, --num_horizontal  number of horizontal subdivisions [4]
-d,       --diagonal  Direction of diagonals: "left", "right", "left/right", -
                      "crossed" [right]
-a,          --alpha  Constant scaling of the bending energy [1]
-t,            --tau  Scaling of the time step wrt. minimal cell size [0.7]
-x,      --max_steps  Maximum number of time steps [24]
-c,     --checkpoint  Output solution data every so many steps. [25]
-s,          --scale  Scale time step at checkpoints by this amount. [0.99]
-e,       --eps_stop  Stopping threshold. [1e-06]
-p,          --pause  Pause each worker for so many seconds in order to attac-
                      h a debugger [0]
-q,           --test  Run the specified test (dofs, blockvector, dkt) [none]
```


## License

All code released under
the [GNU Lesser General Public License](http://www.gnu.org/licenses).

## References

[1] S. Bartels, "Approximation of Large Bending Isometries with
    Discrete Kirchhoff Triangles", SIAM J. Numer. Anal., vol. 51, no. 1,
    pp. 516–525, Jan. 2013.
