# Nonlinear Kirchhoff plate model

This is a C++ implementation in [FEniCS](www.fenicsproject.org) of the
nonlinear Kirchhoff model using a non-conforming implementation with
Discrete Kirchhoff Triangles as described in [1].

Note that there is a typo in the discrete equations in p. 521: the top
block of the right hand side should read $ - \alpha T^t S T Y^n + F$.

## To do

1. The problem with skewed solutions has reappeared!!!
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

## License

All code released under
the [GNU Lesser General Public License](http://www.gnu.org/licenses).

## References

[1] S. Bartels, "Approximation of Large Bending Isometries with
    Discrete Kirchhoff Triangles", SIAM J. Numer. Anal., vol. 51, no. 1,
    pp. 516–525, Jan. 2013.
