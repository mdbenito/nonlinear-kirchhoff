# Nonlinear Kirchhoff plate model

This is a C++ implementation in [FEniCS](www.fenicsproject.org) of the
nonlinear Kirchhoff model using a non-conforming implementation with
Discrete Kirchhoff Triangles as described in [1].

## To do

For now this is little more than a stub. The **plan** is:

1. Finish `IsometryConstraint`.  I should try and make this
   parallelizable and be careful with the initialisation of the
   sparsity pattern. Use DiscreteOperators.h in the dolfin sources for
   a guide.
1. Hack together the gradient flow and solution.
1. Improve the hackish implementation of `DKTGradient`. In particular
   ensure parallel operation is possible.
1. Improve the hackish implementation of `KirchhoffAssembler`. In
   particular ensure parallel operation is possible.

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
