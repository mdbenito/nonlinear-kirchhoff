#include "IsometryConstraint.h"
#include "output.h"
#include <dolfin.h>
#include <dolfin/fem/fem_utils.h>
#include <dolfin/la/PETScMatrix.h>

namespace dolfin {

#define TEST_PETSC_ERROR(__ierr, __funname)                             \
  if (__ierr != 0) PETScObject::petsc_error(__ierr, __FILE__, __funname);

  IsometryConstraint::IsometryConstraint(const FunctionSpace& W)
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
      _B_tensor_layout = _B->factory().create_layout(2);    // rank 2 tensor
      dolfin_assert(_B_tensor_layout);                      // when can that fail?
      _Bt_tensor_layout = _Bt->factory().create_layout(2);
      dolfin_assert(_Bt_tensor_layout);

      // std::cout << "\t\tCHECK ME: IndexMap init ok?\n";
      auto nvert = mesh.num_vertices();
      // std::cout << "\t\tnvert = " << nvert << "\n";
      auto row_index_map_B = std::make_shared<IndexMap>(mesh.mpi_comm(), nvert*4, 1);
      // row_index_map->set_local_to_global(std::vector<std::size_t>());  // How?
      auto col_index_map_Bt = std::make_shared<IndexMap>(mesh.mpi_comm(), nvert*4, 1);
      // col_index_map_Bt->set_local_to_global(std::vector<std::size_t>());  // How?

      std::vector<std::shared_ptr<const IndexMap>> index_maps_B
        { row_index_map_B, W.dofmap()->index_map() };
      _B_tensor_layout->init(mesh.mpi_comm(), index_maps_B,
                             TensorLayout::Ghosts::UNGHOSTED);
      
      auto local_col_range_B = W.dofmap()->ownership_range();
      auto pattern_B = _B_tensor_layout->sparsity_pattern();
      dolfin_assert(pattern_B);   // when can the pattern be null?
      pattern_B->init(mesh.mpi_comm(), index_maps_B);

      std::vector<std::shared_ptr<const IndexMap>> index_maps_Bt
      { W.dofmap()->index_map(), col_index_map_Bt };
      _Bt_tensor_layout->init(mesh.mpi_comm(), index_maps_Bt,
                              TensorLayout::Ghosts::UNGHOSTED);

      auto local_row_range_Bt = W.dofmap()->ownership_range();
      auto pattern_Bt = _Bt_tensor_layout->sparsity_pattern();
      dolfin_assert(pattern_Bt)
      pattern_Bt->init(mesh.mpi_comm(), index_maps_Bt);
    
      // Build sparsity pattern
      la_index dofs[3];  // in order: point eval, dx, dy
      for (VertexIterator v(mesh); !v.end(); ++v)
      {
        // iterate over the 3 subspaces
        for (int sub = 0; sub < 3; ++sub)
        {
          // Thw following should be a process-local index
          auto idx = static_cast<la_index>(v->index());

          dofs[0] = _v2d[9*idx + 3*sub];
          dofs[1] = _v2d[9*idx + 3*sub + 1];
          dofs[2] = _v2d[9*idx + 3*sub + 2];
          // B
          std::vector<ArrayView<const la_index>> entries;
          if (dofs[1] >= local_col_range_B.first && dofs[1]
              < local_col_range_B.second)
          {
            la_index arr_i[] = {4*idx+0, 4*idx+1, 4*idx+2};
            la_index arr_j[] = {dofs[1], dofs[1], dofs[1]};
            auto map_i = ArrayView<const la_index>(3, arr_i);
            auto map_j = ArrayView<const la_index>(3, arr_j);
            entries.push_back(map_i);
            entries.push_back(map_j);
            pattern_B->insert_local(entries);
            // pattern_B->insert_local(4*idx+0, dofs[1]);
            // pattern_B->insert_local(4*idx+1, dofs[1]);
            // pattern_B->insert_local(4*idx+2, dofs[1]);
          }
          if (dofs[2] >= local_col_range_B.first && dofs[2]
              < local_col_range_B.second)
          {
            la_index arr_i[] = {4*idx+1, 4*idx+2, 4*idx+3};
            la_index arr_j[] = {dofs[2], dofs[2], dofs[2]};
            auto map_i = ArrayView<const la_index>(3, arr_i);
            auto map_j = ArrayView<const la_index>(3, arr_j);
            entries.push_back(map_i);
            entries.push_back(map_j);
            pattern_B->insert_local(entries);
            // pattern_B->insert_local(4*idx+1, dofs[2]);
            // pattern_B->insert_local(4*idx+2, dofs[2]);
            // pattern_B->insert_local(4*idx+3, dofs[2]);
          }
          // Bt
          if (dofs[1] >= local_row_range_Bt.first && dofs[1]
              < local_row_range_Bt.second)
          {
            la_index arr_i[] = {dofs[1], dofs[1], dofs[1]};
            la_index arr_j[] = {4*idx+0, 4*idx+1, 4*idx+2};
            auto map_i = ArrayView<const la_index>(3, arr_i);
            auto map_j = ArrayView<const la_index>(3, arr_j);
            entries.push_back(map_i);
            entries.push_back(map_j);
            pattern_Bt->insert_local(entries);
            // pattern_Bt->insert_global(dofs[1], 4*idx+0);
            // pattern_Bt->insert_global(dofs[1], 4*idx+1);
            // pattern_Bt->insert_global(dofs[1], 4*idx+2);
          }
          if (dofs[2] >= local_row_range_Bt.first && dofs[2]
              < local_row_range_Bt.second)
          {
            la_index arr_i[] = {dofs[2], dofs[2], dofs[2]};
            la_index arr_j[] = {4*idx+1, 4*idx+2, 4*idx+3};
            auto map_i = ArrayView<const la_index>(3, arr_i);
            auto map_j = ArrayView<const la_index>(3, arr_j);
            entries.push_back(map_i);
            entries.push_back(map_j);
            pattern_Bt->insert_local(entries);
            // pattern_Bt->insert_global(dofs[2], 4*idx+1);
            // pattern_Bt->insert_global(dofs[2], 4*idx+2);
            // pattern_Bt->insert_global(dofs[2], 4*idx+3);
          }
        }
      }
      pattern_B->apply();
      _B->init(*_B_tensor_layout);
      // std::cout << "Initialised B with size " << _B->size(0) << " x " << _B->size(1) << "\n";
      // std::cout << "Pattern:\n" << pattern_B->str(true) << "\n";
      pattern_Bt->apply();
      _Bt->init(*_Bt_tensor_layout);
      // std::cout << "Initialised Bt with size " << _Bt->size(0) << " x " << _Bt->size(1) << "\n";
      // std::cout << "Pattern:\n" << pattern_Bt->str(true) << "\n";      
    }

    // std::cout << "FIXME! IsometryConstraint: apply() at construction causes PETSc err out of bounds later. ";
    // _B->apply("insert");
    // _Bt->apply("insert");
  }
  
  void
  IsometryConstraint::update_with(const Function& y)
  {
    
    la_index dofs[3] = {-1, -1, -1};
    double values[4*3] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    const auto& mesh = *(y.function_space()->mesh());
    const auto& Y = *(y.vector());
    assert(Y.local_range() == _B.local_range());   // WTF??
    
    for (VertexIterator v(mesh); !v.end(); ++v)
    {
      auto idx = static_cast<la_index>(v->index());   // should be local index
      la_index rows[4] = {4*idx, 4*idx+1, 4*idx+2, 4*idx+3};
      // std::cout << "\nVertex " << idx << ":\n\n";
      for (int sub = 0; sub < 3; ++sub)   // iterate over the 3 subspaces
      {
        dofs[1] = _v2d[9*idx + 3*sub + 1];
        dofs[2] = _v2d[9*idx + 3*sub + 2];

        // std::cout << "\nSubspace " << sub << ":\n";
      
        // Copy the values of y into the 4x3 chunk:
        /* values[0] = 0.0; */    values[1]  = 2*Y[dofs[1]]; /*  values[2] =          0.0; */
        /* values[3] = 0.0; */    values[4]  =   Y[dofs[2]];     values[5] =   Y[dofs[1]];
        /* values[6] = 0.0; */    values[7]  =   Y[dofs[2]];     values[8] =   Y[dofs[1]];
        /* values[9] = 0.0; */ /* values[10] =          0.0; */ values[11] = 2*Y[dofs[2]];

        // std::cout << "Updating row " << rows[0] << " with " << values[1]
                  // << " at dof " << dofs[1] << " for vertex " << v->index() << "\n";
        _B->set(&(values[1]), 1, &(rows[0]), 1, &(dofs[1]));
        // std::cout << "Updating row " << rows[1] << " with " << values[4] << ", " << values[5]
                  // << " at dofs " << dofs[1] << ", " << dofs[2] << " for vertex " << v->index() << "\n";
        _B->set(&(values[4]), 1, &(rows[1]), 2, &(dofs[1]));
        // std::cout << "Updating row " << rows[2] << " with " << values[7] << ", " << values[8]
                  // << " at dofs " << dofs[1] << ", " << dofs[2] << " for vertex " << v->index() << "\n";
        _B->set(&(values[7]), 1, &(rows[2]), 2, &(dofs[1]));
        // std::cout << "Updating row " << rows[3] << " with " << values[11]
                  // << " at dof " << dofs[2] << " for vertex " << v->index() << "\n";
        _B->set(&(values[11]), 1, &(rows[3]), 1, &(dofs[2]));


        // Now transposed
        // std::cout << "Transpose:\n";
        // Copy the values of y into the 3x4 chunk:
        /* values[0] = 0.0;          values[1] = 0.0;         values[2] = 0.0;          values[3] = 0.0; */
        values[4] = 2*Y[dofs[1]]; values[5] = Y[dofs[2]];  values[6] = Y[dofs[2]];   /* values[7] = 0.0; */
        /* values[8] = 0.0; */    values[9] = Y[dofs[1]];  values[10] = Y[dofs[1]]; values[11] = 2*Y[dofs[2]];
      
        // std::cout << "Updating row " << dofs[1] << " with " << values[4] << ", " << values[5] << ", " << values[6]
                  // << " at col " << rows[0] << " for vertex " << v->index() << "\n";
        _Bt->set(&(values[4]), 1, &(dofs[1]), 3, &(rows[0]));
        // std::cout << "Updating row " << dofs[2] << " with " << values[9] << ", " << values[10] << ", " << values[11]
                  // << " at col " << rows[1] << " for vertex " << v->index() << "\n";
        _Bt->set(&(values[9]), 1, &(dofs[2]), 3, &(rows[1]));
      }
    }
    _B->apply("insert");
    _Bt->apply("insert");
  }

  std::shared_ptr<GenericMatrix>
  IsometryConstraint::get_padding()
  {
    // HACK: I really don't know how to create an empty n x n
    // dolfin::Matrix, so I use PETSc... duh
    Mat mat;
    PetscErrorCode ierr;
    int N = _B->size(0);
    ierr = MatCreate(MPI_COMM_WORLD, &mat);
    TEST_PETSC_ERROR(ierr, "MatCreate");
    // MATAIJ fails to pick MPIAIJ in parallel (??)
    ierr = MatSetType(mat, MATMPIAIJ);
    TEST_PETSC_ERROR(ierr, "MatSetType");
    ierr = MatSetSizes(mat, PETSC_DECIDE, PETSC_DECIDE, N, N);
    TEST_PETSC_ERROR(ierr, "MatSetSizes");
    ierr = MatMPIAIJSetPreallocation(mat, 1, NULL, 0, NULL);
    TEST_PETSC_ERROR(ierr, "MatMPIAIJSetPreallocation");
    ierr = MatSeqAIJSetPreallocation(mat, 1, NULL);
    TEST_PETSC_ERROR(ierr, "MatSeqAIJSetPreallocation");
    // MatSetUp(mat);
    MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);
    // Careful to shift after assembly or it won't have any effect
    MatShift(mat, 1.0);

    PetscInt lm, ln, gm, gn;
    MatGetLocalSize(mat, &lm, &ln);
    MatGetSize(mat, &gm, &gn);
    std::cout << "Padding has local size " << lm << " x " << ln
              << " and global size " << gm << " x " << gn << "\n";
    return std::make_shared<PETScMatrix>(mat);
  }

}

