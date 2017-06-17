#include "IsometryConstraint.h"

#include <dolfin.h>
#include <dolfin/fem/fem_utils.h>

namespace dolfin {

  IsometryConstraint::IsometryConstraint(const FunctionSpace& W,
                                         const VertexFunction<bool>& boundary_marker)
    : _v2d(vertex_to_dof_map(W)),
      _B(std::make_shared<Matrix>()), _Bt(std::make_shared<Matrix>())
  {
    const Mesh& mesh = *(W.mesh());

    // TODO Check that W is a DKT vector space with 3 components
    if (W.dim() != 9 * mesh.size_global(0))
    {
      dolfin_error("IsometryConstraint.cpp",
                   "initialise discrete isometry constraint",
                   "function space has wrong dimension");
    }
    
    if (W.element()->num_sub_elements() != 3)
    {
      dolfin_error("IsometryConstraint.cpp",
                   "initialise discrete isometry constraint",
                   "function space has wrong number of components");
    }
      
    // What's the difference with vertex_to_dofmap() ??
    // const auto& v2d = W.dofmap()->dofs(mesh, 0);

    {
      _B_tensor_layout = _B->factory().create_layout(2);  // 2 is the rank
      dolfin_assert(_B_tensor_layout);  // what for?

      // FIXME: is this ok? every process should own a 4xN block
      auto row_index_map = std::make_shared<IndexMap>(mesh.mpi_comm(), 4, 1);
      // row_index_map->set_local_to_global(vector of global indices beyond local range);

    
      std::vector<std::shared_ptr<const IndexMap>> index_maps
        { row_index_map, W.dofmap()->index_map() };

      auto local_column_range = W.dofmap()->ownership_range();

      _B_tensor_layout->init(mesh.mpi_comm(), index_maps,
                             TensorLayout::Ghosts::UNGHOSTED);

      auto pattern = _B_tensor_layout->sparsity_pattern();
      dolfin_assert(pattern);   // when can the pattern be null?

      pattern->init(mesh.mpi_comm(), index_maps);
      // Build sparsity pattern
      std::size_t dofs[3];  // in order: point eval, dx, dy
      for (VertexIterator v(mesh); !v.end(); ++v)
      {
        if (boundary_marker[*v])
          continue;
        for (int sub = 0; sub < 3; ++sub)  // iterate over the 3 subspaces
        {
          // dofs[0] = _v2d[9*v->index() + 3*sub];
          dofs[1] = _v2d[9*v->index() + 3*sub + 1];
          dofs[2] = _v2d[9*v->index() + 3*sub + 2];

          pattern->insert_global(0, dofs[1]);
          pattern->insert_global(1, dofs[1]);
          pattern->insert_global(1, dofs[2]);
          pattern->insert_global(2, dofs[1]);
          pattern->insert_global(2, dofs[2]);
          pattern->insert_global(3, dofs[2]);
        }
      }
      pattern->apply();
      _B->init(*_B_tensor_layout);
      // _B->apply("insert");
      std::cout << "Initialised B with size " << _B->size(0) << " x " << _B->size(1) << "\n";
      // std::cout << "Pattern:\n" << pattern->str(true) << "\n";
    }

    // This is me being lazy and sloppy...
    
    {
      _Bt_tensor_layout = _Bt->factory().create_layout(2);  // 2 is the rank
      dolfin_assert(_Bt_tensor_layout);  // what for?

      // FIXME: is this ok? every process should own a Nx4 block
      auto col_index_map = std::make_shared<IndexMap>(mesh.mpi_comm(), 4, 1);
      // col_index_map->set_local_to_global(vector of global indices beyond local range);

    
      std::vector<std::shared_ptr<const IndexMap>> index_maps
      { W.dofmap()->index_map(), col_index_map };

      auto local_row_range = W.dofmap()->ownership_range();

      _Bt_tensor_layout->init(mesh.mpi_comm(), index_maps,
                              TensorLayout::Ghosts::UNGHOSTED);

      auto pattern = _Bt_tensor_layout->sparsity_pattern();
      dolfin_assert(pattern)
      pattern->init(mesh.mpi_comm(), index_maps);
    
      // _Build sparsity pattern
      std::size_t dofs[3];
      for (VertexIterator v(mesh); !v.end(); ++v)
      {
        for (int sub = 0; sub < 3; ++sub)   // iterate over the 3 subspaces
        {
          // dofs[0] = _v2d[9*v->index() + 3*sub];
          dofs[1] = _v2d[9*v->index() + 3*sub + 1];
          dofs[2] = _v2d[9*v->index() + 3*sub + 2];

          pattern->insert_global(dofs[1], 0);
          pattern->insert_global(dofs[1], 1);
          pattern->insert_global(dofs[1], 2);
          pattern->insert_global(dofs[2], 1);
          pattern->insert_global(dofs[2], 2);
          pattern->insert_global(dofs[2], 3);
        }
      }
      pattern->apply();
      _Bt->init(*_Bt_tensor_layout);
      std::cout << "Initialised Bt with size " << _Bt->size(0) << " x " << _Bt->size(1) << "\n";
      std::cout << "Pattern:\n" << pattern->str(true) << "\n";      
    }
  }
  
  void
  IsometryConstraint::update_with(const Function& y)
  {
    la_index dofs[3] = {-1, -1, -1};
    la_index rows[4] = {0, 1, 2, 3};
    double values[4*3] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    const auto& mesh = *(y.function_space()->mesh());
    const auto& Y = *(y.vector());
    // assert(Y.local_range() == _B.local_range());   // WTF??
    
    for (VertexIterator v(mesh); !v.end(); ++v)
    {
      // std::cout << "\nVertex " << v->index() ":\n\n";
      for (int sub = 0; sub < 3; ++sub)   // iterate over the 3 subspaces
      {
        // dofs[0] = _v2d[9*v->index() + 3*sub];
        dofs[1] = _v2d[9*v->index() + 3*sub + 1];
        dofs[2] = _v2d[9*v->index() + 3*sub + 2];

        // std::cout << "\nSubspace " << sub << ":\n";
      
        // Copy the values of y into the 4x3 chunk:
        /* values[0] = 0.0; */    values[1]  = 2*Y[dofs[1]]; /*  values[2] =          0.0; */
        /* values[3] = 0.0; */    values[4]  =   Y[dofs[2]];     values[5] =   Y[dofs[1]];
        /* values[6] = 0.0; */    values[7]  =   Y[dofs[2]];     values[8] =   Y[dofs[1]];
        /* values[9] = 0.0; */ /* values[10] =          0.0; */  values[11] = 2*Y[dofs[2]];

        // std::cout << "Updating row " << rows[0] << " with " << values[1]
        //           << " at dof " << dofs[1] << " for vertex " << v->index() << "\n";
        _B->set(&(values[1]), 1, &(rows[0]), 1, &(dofs[1]));
        // std::cout << "Updating row " << rows[1] << " with " << values[4] << ", " << values[5]
        //           << " at dofs " << dofs[1] << ", " << dofs[2] << " for vertex " << v->index() << "\n";
        _B->set(&(values[4]), 1, &(rows[1]), 2, &(dofs[1]));
        // std::cout << "Updating row " << rows[2] << " with " << values[7] << ", " << values[8]
        //           << " at dofs " << dofs[1] << ", " << dofs[2] << " for vertex " << v->index() << "\n";
        _B->set(&(values[7]), 1, &(rows[2]), 2, &(dofs[1]));
        // std::cout << "Updating row " << rows[3] << " with " << values[11]
        //           << " at dof " << dofs[2] << " for vertex " << v->index() << "\n";
        _B->set(&(values[11]), 1, &(rows[3]), 1, &(dofs[2]));


        // Now transposed
        // std::cout << "Transpose:\n";
        // Copy the values of y into the 3x4 chunk:
        /* values[0] = 0.0;          values[1] = 0.0;         values[2] = 0.0;          values[3] = 0.0; */
        values[4] = 2*Y[dofs[1]]; values[5] = Y[dofs[2]];  values[6] = Y[dofs[1]];   /* values[7] = 0.0; */
        /* values[8] = 0.0; */    values[9] = Y[dofs[2]];  values[10] = Y[dofs[1]]; values[11] = 2*Y[dofs[2]];
      
        // std::cout << "Updating row " << dofs[1] << " with " << values[4] << ", " << values[5] << ", " << values[6]
        //           << " at col " << rows[0] << " for vertex " << v->index() << "\n";
        _Bt->set(&(values[4]), 1, &(dofs[1]), 3, &(rows[0]));
        // std::cout << "Updating row " << dofs[2] << " with " << values[9] << ", " << values[10] << ", " << values[11]
        //           << " at col " << rows[1] << " for vertex " << v->index() << "\n";
        _Bt->set(&(values[9]), 1, &(dofs[2]), 3, &(rows[1]));
      }
    }
    _B->apply("insert");
    _Bt->apply("insert");
  }
}
