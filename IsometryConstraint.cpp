#include "IsometryConstraint.h"

#include <memory>
#include <dolfin.h>
#include <fem_utils.h>

namespace dolfin {

  IsometryConstraint::IsometryConstraint(const FunctionSpace& W)
    : _v2d(vertex_to_dof_map(W)),
      _B(std::make_shared<Matrix>()), _Bt(std::make_shared<Matrix>())
  {
    const Mesh& mesh = *(W.mesh());

    // TODO Check that W is a DKT vector space with 3 components
    if (W.dim() != 9*mesh.size_global(0))
    {
      dolfin_error("IsometryConstraint.cpp",
                   "initialise discrete isometry constraint",
                   "function space has wrong dimension");
    }
    if (W.num_sub_spaces() != 3)
    {
      dolfin_error("IsometryConstraint.cpp",
                   "initialise discrete isometry constraint",
                   "function space has wrong number of components");
    }
      
    // What's the difference with vertex_to_dofmap() ??
    // const auto& v2d = W.dofmap()->dofs(mesh, 0);

    {
      auto tensor_layout = _B->factory().create_layout(2);  // 2 is the rank
      dolfin_assert(tensor_layout);  // what for?

      // FIXME: is this ok? every process should own a 4xN block
      auto row_index_map = std::make_shared<IndexMap>(mesh.mpi_comm(), 4, 1);
      row_index_map.set_local_to_global(WHAT HERE);

    
      std::vector<std::shared_ptr<const IndexMap> > index_maps
      { row_index_map, W.dofmap()->index_map() };

      auto local_column_range = W.dofmap()->ownership_range();

      tensor_layout->init(mesh.mpi_comm(), index_maps,
                          TensorLayout::Ghosts::UNGHOSTED);

      SparsityPattern& pattern = *tensor_layout->sparsity_pattern();
      pattern.init(mesh.mpi_comm(), index_maps);
    
      // _Build sparsity pattern
      if (tensor_layout->sparsity_pattern())
      {
        std::size_t dofs[3];  // in order: point eval, dx, dy
        for (VertexIterator v(mesh); !edge.end(); ++edge)
        {
          for (int sub = 0; sub < 3; ++sub)  // iterate over the 3 subspaces
          {
            dofs[0] = _v2d[9*v.index()+3*i];
            dofs[1] = _v2d[9*v.index()+3*i+1];
            dofs[2] = _v2d[9*v.index()+3*i+2];
        
            pattern.insert_global(0, dofs[1]);
            pattern.insert_global(1, dofs[1]);
            pattern.insert_global(1, dofs[2]);
            pattern.insert_global(2, dofs[1]);
            pattern.insert_global(2, dofs[2]);
            pattern.insert_global(3, dofs[2]);
          }
        }
        pattern.apply();
      }
      _B->init(*tensor_layout); 
    }

    // This is me being lazy and sloppy...
    
    {
      auto tensor_layout = _Bt->factory().create_layout(2);  // 2 is the rank
      dolfin_assert(tensor_layout);  // what for?

      // FIXME: is this ok? every process should own a Nx4 block
      auto col_index_map = std::make_shared<IndexMap>(mesh.mpi_comm(), 4, 1);
      col_index_map.set_local_to_global(WHAT HERE);

    
      std::vector<std::shared_ptr<const IndexMap> > index_maps
      { W.dofmap()->index_map(), col_index_map };

      auto local_row_range = W.dofmap()->ownership_range();

      tensor_layout->init(mesh.mpi_comm(), index_maps,
                          TensorLayout::Ghosts::UNGHOSTED);

      SparsityPattern& pattern = *tensor_layout->sparsity_pattern();
      pattern.init(mesh.mpi_comm(), index_maps);
    
      // _Build sparsity pattern
      if (tensor_layout->sparsity_pattern())
      {
        std::size_t dofs[3];
        for (VertexIterator v(mesh); !edge.end(); ++edge)
        {
          for (int sub = 0; sub < 3; ++sub)  // iterate over the 3 subspaces
          {
            dofs[0] = _v2d[9*v.index()+3*i];
            dofs[1] = _v2d[9*v.index()+3*i+1];
            dofs[2] = _v2d[9*v.index()+3*i+2];

            pattern.insert_global(dofs[1], 0);
            pattern.insert_global(dofs[1], 1);
            pattern.insert_global(dofs[1], 2);
            pattern.insert_global(dofs[2], 1);
            pattern.insert_global(dofs[2], 2);
            pattern.insert_global(dofs[2], 3);
          }
        }
        pattern.apply();
      }
      _Bt->init(*tensor_layout);
    }
  }
  
  void
  IsometryConstraint::update(const Function& Y)
  {
    std::size_t dofs[3];
    std::size_t rows[4] = {0, 1, 2, 3};
    double values[4*3] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    
    auto yy = Y.vector();
    assert(yy->local_range() == _B.local_range());
    
    for (VertexIterator v(mesh); !edge.end(); ++edge)
    {
      dofs[0] = _v2d[v.index()];
      dofs[1] = _v2d[v.index()+1];
      dofs[2] = _v2d[v.index()+2];

      // Copy the values of Y into the 4x3 chunk:
      /* 0 */     values[1] = 2*yy[dofs[1]];      /* 0 */
      /* 0 */     values[4] =   yy[dofs[2]];      values[5] =   yy[dofs[1]];
      /* 0 */     values[7] =   yy[dofs[2]];      values[8] =   yy[dofs[1]];
      /* 0 */     /* 0 */                        values[11] = 2*yy[dofs[2]];

      B->set(values, 4, rows, 3, dofs);  // set() uses global indices
    }

    B->apply("insert");
  }
  
}
