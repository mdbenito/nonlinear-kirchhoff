#ifndef __KIRCHHOFFASSEMBLER_H
#define __KIRCHHOFFASSEMBLER_H

#include <vector>
#include <array>
#include <cmath>
#include <memory>
#include <dolfin/fem/AssemblerBase.h>
#include "DKTGradient.h"

namespace dolfin
{

  // Forward declarations
  class GenericMatrix;
  class Form;
  class UFC;
  class Cell;
  template<typename T> class MeshFunction;

  /// This class provides a system assembler for vector valued P_2
  /// forms which applies a discrete gradient operator to 

  class KirchhoffAssembler : public AssemblerBase
  {
    DKTGradient grad;
    
  public:

    /// Constructor
    KirchhoffAssembler() {}    

    /// Assemble tensor from given form
    ///
    /// *Arguments*
    ///     A (_GenericMatrix_)
    ///         The tensor to assemble.
    ///     a (_Form_)
    ///         The P_3^{red} form to assemble the tensor from
    ///     p26form (_Form_)
    ///         The (\nabla u, \nabla v) P_2^6 form to assemble the
    ///         tensor from.
    void assemble(GenericMatrix& A, const Form& a, const Form& p26form);

    void assemble_cells(GenericMatrix& A, const Form& a, UFC& ufc,
                        std::shared_ptr<const MeshFunction<std::size_t>> domains,
                        std::vector<double>* values);

    void assemble_exterior_facets(GenericMatrix& A, const Form& a,
                                  UFC& ufc,
                                  std::shared_ptr<const MeshFunction<std::size_t>> domains,
                                  std::vector<double>* values);

    void assemble_interior_facets(GenericMatrix& A, const Form& a,
                                  UFC& ufc,
                                  std::shared_ptr<const MeshFunction<std::size_t>> domains,
                                  std::shared_ptr<const MeshFunction<std::size_t>> cell_domains,
                                  std::vector<double>* values);

    /// Assemble tensor from given form over vertices. This function is
    /// provided for users who wish to build a customized assembler.
    void assemble_vertices(GenericMatrix& A, const Form& a, UFC& ufc,
                           std::shared_ptr<const MeshFunction<std::size_t>> domains);

  };

}

#endif
