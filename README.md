# Linear Kirchhoff model in FEniCS

A C++ implementation in FEniCS of the nonlinear Kirchhoff model using
non-conforming Discrete Kirchhoff Triangles as described in [1].

This serves as another test for DKT elements and a first
implementation of a discrete gradient flow using FEniCS.

## To do

For now this is little more than a stub. The **plan** is:

1. Implement the discrete isometry constraint. See
   `IsometryConstraint`.  I should try and make this parallelizable,
   and be careful with the initialisation of the sparsity pattern. See
   DiscreteOperators.h in the dolfin sources for a guide.
1. Piece together a `BlockMatrix` for the linear system using the
   system matrix from `LinearKirchhoff` and the `IsometryConstraint`.
1. Hack together the gradient flow and solution.
1. Improve the hackish implementation of `DKTGradient`. In particular
   ensure parallel operation is possible.
1. Improve the hackish implementation of `KirchhoffAssembler`. In
   particular ensure parallel operation is possible. Ditto for
   `HermiteDirichletBC`.

## Dependencies

Bundled in the docker image...

## License

GNU GPL v3, most likely. Still have to decide. Nobody reads this
anyway.

## References

[1] S. Bartels, “Approximation of Large Bending Isometries with
    Discrete Kirchhoff Triangles,” SIAM J. Numer. Anal., vol. 51, no. 1,
    pp. 516–525, Jan. 2013.
