// This file is heavily based on Assembler.cpp fron DOLFIN's code
// Copyright (C) 2007-2015 Anders Logg with modifications by Garth
// N. Wells 2007-2009, Ola Skavhaug 2007-2009, Kent-Andre Mardal 2008,
// Joachim B Haga 2012, Martin Alnaes 2013-2015
#include <cassert>
#include <algorithm>

#include <dolfin/log/log.h>
#include <dolfin/log/Progress.h>
#include <dolfin/common/Array.h>
#include <dolfin/common/Timer.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/MeshData.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/SubDomain.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/UFC.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/AssemblerBase.h>

#include "KirchhoffAssembler.h"
#include "output.h"

using namespace dolfin;

/* Remember that a is a Form in a VectorFunctionSpace<DKT> and p2form
 * is a form in a TensorFunctionSpace<P2> with shape (3,2)
 */
void
KirchhoffAssembler::assemble(GenericMatrix& A,
                             const Form& a,
                             const Form& p22form)
{
  // Get cell, exterior / interior facet and vertex domains
  // (std::shared_ptr<const MeshFunction<std::size_t>>)
  auto cell_domains = a.cell_domains();
  auto exterior_facet_domains = a.exterior_facet_domains();
  auto interior_facet_domains = a.interior_facet_domains();
  auto vertex_domains = a.vertex_domains();

  // Check form
  AssemblerBase::check(a);

  // Create data structure for local assembly data
  UFC ufc(p22form);

  // Off-process coefficients
  auto coefficients = a.coefficients();

  init_global_tensor(A, a);

  // Assemble
  assemble_cells(A, a, ufc, cell_domains, NULL);
  assemble_exterior_facets(A, a, ufc, exterior_facet_domains, NULL);
  assemble_interior_facets(A, a, ufc, interior_facet_domains,
                           cell_domains, NULL);
  assemble_vertices(A, a, ufc, vertex_domains);

  if (finalize_tensor)
    A.apply("add");
}

//-----------------------------------------------------------------------------

/// WARNING! I'm using the fact that the local P22 tensor has block
/// diagonal structure because there are no interactions among
/// different components.
void KirchhoffAssembler::assemble_cells(
  GenericMatrix& A,
  const Form& a,         /* The P_3^red form*/
  UFC& ufc,
  std::shared_ptr<const MeshFunction<std::size_t>> domains,
  std::vector<double>* values)
{
  // Skip assembly if there are no cell integrals
  if (!ufc.form.has_cell_integrals())
    return;

  Timer timer("Assemble cells");

  dolfin_assert(a.mesh());
  const Mesh& mesh = *(a.mesh());

  // Superfluous if we are only accepting matrices
  // auto form_rank = ufc.form.rank();
  // assert(form_rank == a.rank());
  // assert(form_rank == 2);
  
  const int range_dim = 3;  // MBD HACK: DKTGradient assumes dim = 3
  
  // Collect pointers to dof maps
  // MBD We use the DKT form 'a' here, so this is ok
  std::vector<const GenericDofMap*> rowdofmaps;
  std::vector<const GenericDofMap*> coldofmaps;
  for (auto j = 0; j < range_dim; ++j)
  {
    rowdofmaps.push_back(a.function_space(0)->sub(j)->dofmap().get());
    coldofmaps.push_back(a.function_space(1)->sub(j)->dofmap().get());
  }
  // Vectors to hold row dofmaps for a cell
  std::vector<ArrayView<const dolfin::la_index>> rowdofs(range_dim);
  std::vector<ArrayView<const dolfin::la_index>> coldofs(range_dim);

  // Cell integral for p22
  auto integral = ufc.default_cell_integral.get();

  // Check whether integral is domain-dependent
  bool use_domains = domains && !domains->empty();

  // Assemble over cells
  ufc::cell ufc_cell;
  std::vector<double> coordinate_dofs;
  
  // MBD: For each cell tensor T, D stores Mt * T_ii * M for
  // i=1,2,3. Recall that T has diagonal block structure
  DKTGradient::P3Tensor D;

  Progress p(AssemblerBase::progress_message(A.rank(), "cells"),
             mesh.num_cells());

  // std::cout << "Form rank is " << form_rank << ".\n";
  
  // auto dm0 = a.function_space(0)->dofmap().get();
  // auto dm1 = a.function_space(1)->dofmap().get();
  
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Get integral for sub domain (if any)
    if (use_domains)
      integral = ufc.get_cell_integral((*domains)[*cell]);

    // Skip if no integral on current domain
    if (!integral)
      continue;

    // Check that cell is not a ghost
    dolfin_assert(!cell->is_ghost());

    // Update to current cell
    cell->get_cell_data(ufc_cell);
    cell->get_coordinate_dofs(coordinate_dofs);
    ufc.update(*cell, coordinate_dofs, ufc_cell,
               integral->enabled_coefficients());
    grad.update(*cell);    // MBD Update DKTGradient operator
    
    // Get local-to-global dof maps for cell
    // MBD: we need to use the dofmaps for DKT
    bool empty_dofmap = false;
    for(auto j = 0; j < range_dim; ++j)
    {
      rowdofs[j] = rowdofmaps[j]->cell_dofs(cell->index());
      coldofs[j] = coldofmaps[j]->cell_dofs(cell->index());
      empty_dofmap = empty_dofmap || rowdofs[j].size() == 0 || coldofs[j].size() == 0;

      // std::cout << "Subdofmap " << j << ": ";
      // for (int i=0;i<rowdofs[j].size();++i)
      //   std::cout << "(" << rowdofs[j][i] << ", " << coldofs[j][i] << ") ";
      // std::cout << ".\n";
    }
    // std::cout << "\nDofmap: ";
    // ArrayView<const la_index> c0, c1;
    // c0 = dm0->cell_dofs(cell->index());
    // c1 = dm1->cell_dofs(cell->index());
    // for (int i=0;i<c0.size();++i)
    //   std::cout << "(" << c0[i] << ", " << c1[i] << ") ";
    // std::cout << ".\n";

    // Skip if at least one dofmap is empty MBD: why would it?
    if (empty_dofmap) 
      continue;

    // Tabulate cell tensor
    integral->tabulate_tensor(ufc.A.data(), ufc.w(),
                              coordinate_dofs.data(),
                              ufc_cell.orientation);

    // dump_raw_matrix(ufc.A, 36, 36, 4, "P22", true, true);

    // split ufc.A in three blocks A_ii
    const double* data = ufc.A.data();
    for(auto j = 0; j < range_dim; ++j)
    {
      // std::cout << "Applying DKT gradient to component " << j << "... ";

      // We use an Eigen::OuterStride<range_dim*12> in DKTGradient.apply()
      // to map chunks of the local tensor data into 12x12 matrices to
      // multiply by.
      grad.apply(data + j*12*12*range_dim + j*12, D);

      // std::cout << "done.\n";
      // std::cout << "Adding dofs to global tensor... ";
      //Use dkt local->global dofs for insertion into global tensor
      A.add_local(D.data(), 9, rowdofs[j].data(), 9, coldofs[j].data());
      // std::cout << "done.\n";
    }
    p++;
  }
}
//-----------------------------------------------------------------------------
void KirchhoffAssembler::assemble_exterior_facets(
  GenericMatrix& A,
  const Form& a,
  UFC& ufc,
  std::shared_ptr<const MeshFunction<std::size_t>> domains,
  std::vector<double>* values)
{
  // Skip assembly if there are no exterior facet integrals
  if (!ufc.form.has_exterior_facet_integrals())
    return;

  // Set timer
  Timer timer("Assemble exterior facets");

  // Extract mesh
  dolfin_assert(a.mesh());
  const Mesh& mesh = *(a.mesh());

  // Form rank
  const std::size_t form_rank = ufc.form.rank();

  // Collect pointers to dof maps
  std::vector<const GenericDofMap*> dofmaps;
  for (std::size_t i = 0; i < form_rank; ++i)
    dofmaps.push_back(a.function_space(i)->dofmap().get());

  // Vector to hold dof map for a cell
  std::vector<ArrayView<const dolfin::la_index>> dofs(form_rank);

  // Exterior facet integral
  const ufc::exterior_facet_integral* integral
    = ufc.default_exterior_facet_integral.get();

  // Check whether integral is domain-dependent
  bool use_domains = domains && !domains->empty();

  // Compute facets and facet - cell connectivity if not already computed
  const std::size_t D = mesh.topology().dim();
  mesh.init(D - 1);
  mesh.init(D - 1, D);
  dolfin_assert(mesh.ordered());

  // Assemble over exterior facets (the cells of the boundary)
  ufc::cell ufc_cell;
  std::vector<double> coordinate_dofs;
  Progress p(AssemblerBase::progress_message(A.rank(), "exterior facets"),
             mesh.num_facets());
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    // Only consider exterior facets
    if (!facet->exterior())
    {
      p++;
      continue;
    }

    // Get integral for sub domain (if any)
    if (use_domains)
      integral = ufc.get_exterior_facet_integral((*domains)[*facet]);

    // Skip integral if zero
    if (!integral)
      continue;

    // Get mesh cell to which mesh facet belongs (pick first, there is
    // only one)
    dolfin_assert(facet->num_entities(D) == 1);
    Cell mesh_cell(mesh, facet->entities(D)[0]);

    // Check that cell is not a ghost
    dolfin_assert(!mesh_cell.is_ghost());

    // Get local index of facet with respect to the cell
    const std::size_t local_facet = mesh_cell.index(*facet);

    // Update UFC cell
    mesh_cell.get_cell_data(ufc_cell, local_facet);
    mesh_cell.get_coordinate_dofs(coordinate_dofs);

    // Update UFC object
    ufc.update(mesh_cell, coordinate_dofs, ufc_cell,
               integral->enabled_coefficients());

    // Get local-to-global dof maps for cell
    for (std::size_t i = 0; i < form_rank; ++i)
      dofs[i] = dofmaps[i]->cell_dofs(mesh_cell.index());

    // Tabulate exterior facet tensor
    integral->tabulate_tensor(ufc.A.data(),
                              ufc.w(),
                              coordinate_dofs.data(),
                              local_facet,
                              ufc_cell.orientation);

    // Add entries to global tensor
    A.add_local(ufc.A.data(), dofs);

    p++;
  }
}
//-----------------------------------------------------------------------------
void KirchhoffAssembler::assemble_interior_facets(
  GenericMatrix& A,
  const Form& a,
  UFC& ufc,
  std::shared_ptr<const MeshFunction<std::size_t>> domains,
  std::shared_ptr<const MeshFunction<std::size_t>> cell_domains,
  std::vector<double>* values)
{
  // Skip assembly if there are no interior facet integrals
  if (!ufc.form.has_interior_facet_integrals())
    return;

  // Set timer
  Timer timer("Assemble interior facets");

  // Extract mesh and coefficients
  dolfin_assert(a.mesh());
  const Mesh& mesh = *(a.mesh());

  // MPI rank
  const int my_mpi_rank = MPI::rank(mesh.mpi_comm());

  // Form rank
  const std::size_t form_rank = ufc.form.rank();

  // Collect pointers to dof maps
  std::vector<const GenericDofMap*> dofmaps;
  for (std::size_t i = 0; i < form_rank; ++i)
    dofmaps.push_back(a.function_space(i)->dofmap().get());

  // Vector to hold dofs for cells, and a vector holding pointers to same
  std::vector<std::vector<dolfin::la_index>> macro_dofs(form_rank);
  std::vector<ArrayView<const dolfin::la_index>> macro_dof_ptrs(form_rank);

  // Interior facet integral
  const ufc::interior_facet_integral* integral
    = ufc.default_interior_facet_integral.get();

  // Check whether integral is domain-dependent
  bool use_domains = domains && !domains->empty();
  bool use_cell_domains = cell_domains && !cell_domains->empty();

  // Compute facets and facet - cell connectivity if not already computed
  const std::size_t D = mesh.topology().dim();
  mesh.init(D - 1);
  mesh.init(D - 1, D);
  dolfin_assert(mesh.ordered());

  // Assemble over interior facets (the facets of the mesh)
  ufc::cell ufc_cell[2];
  std::vector<double> coordinate_dofs[2];
  Progress p(AssemblerBase::progress_message(A.rank(), "interior facets"),
             mesh.num_facets());
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    if (facet->num_entities(D) == 1)
      continue;

    // Check that facet is not a ghost
    dolfin_assert(!facet->is_ghost());

    // Get integral for sub domain (if any)
    if (use_domains)
      integral = ufc.get_interior_facet_integral((*domains)[*facet]);

    // Skip integral if zero
    if (!integral)
      continue;

    // Get cells incident with facet (which is 0 and 1 here is arbitrary)
    dolfin_assert(facet->num_entities(D) == 2);
    std::size_t cell_index_plus = facet->entities(D)[0];
    std::size_t cell_index_minus = facet->entities(D)[1];

    if (use_cell_domains && (*cell_domains)[cell_index_plus]
        < (*cell_domains)[cell_index_minus])
    {
      std::swap(cell_index_plus, cell_index_minus);
    }

    // The convention '+' = 0, '-' = 1 is from ffc
    const Cell cell0(mesh, cell_index_plus);
    const Cell cell1(mesh, cell_index_minus);

    // Get local index of facet with respect to each cell
    std::size_t local_facet0 = cell0.index(*facet);
    std::size_t local_facet1 = cell1.index(*facet);

    // Update to current pair of cells
    cell0.get_cell_data(ufc_cell[0], local_facet0);
    cell0.get_coordinate_dofs(coordinate_dofs[0]);
    cell1.get_cell_data(ufc_cell[1], local_facet1);
    cell1.get_coordinate_dofs(coordinate_dofs[1]);

    ufc.update(cell0, coordinate_dofs[0], ufc_cell[0],
               cell1, coordinate_dofs[1], ufc_cell[1],
               integral->enabled_coefficients());

    // Tabulate dofs for each dimension on macro element
    for (std::size_t i = 0; i < form_rank; i++)
    {
      // Get dofs for each cell
      const ArrayView<const dolfin::la_index> cell_dofs0
        = dofmaps[i]->cell_dofs(cell0.index());
      const ArrayView<const dolfin::la_index> cell_dofs1
        = dofmaps[i]->cell_dofs(cell1.index());

      // Create space in macro dof vector
      macro_dofs[i].resize(cell_dofs0.size() + cell_dofs1.size());

      // Copy cell dofs into macro dof vector
      std::copy(cell_dofs0.data(), cell_dofs0.data() + cell_dofs0.size(),
                macro_dofs[i].begin());
      std::copy(cell_dofs1.data(), cell_dofs1.data() + cell_dofs1.size(),
                macro_dofs[i].begin() + cell_dofs0.size());
      macro_dof_ptrs[i].set(macro_dofs[i]);
    }

    // Tabulate interior facet tensor on macro element
    integral->tabulate_tensor(ufc.macro_A.data(),
                              ufc.macro_w(),
                              coordinate_dofs[0].data(),
                              coordinate_dofs[1].data(),
                              local_facet0,
                              local_facet1,
                              ufc_cell[0].orientation,
                              ufc_cell[1].orientation);

    if (cell0.is_ghost() != cell1.is_ghost())
    {
      int ghost_rank = -1;
      if (cell0.is_ghost())
        ghost_rank = cell0.owner();
      else
        ghost_rank = cell1.owner();

      dolfin_assert(my_mpi_rank != ghost_rank);
      dolfin_assert(ghost_rank != -1);
      if (ghost_rank < my_mpi_rank)
        continue;
    }

    // Add entries to global tensor
    A.add_local(ufc.macro_A.data(), macro_dof_ptrs);

    p++;
  }
}
//-----------------------------------------------------------------------------
void KirchhoffAssembler::assemble_vertices(
  GenericMatrix& A,
  const Form& a,
  UFC& ufc,
  std::shared_ptr<const MeshFunction<std::size_t>> domains)
{
  // Skip assembly if there are no point integrals
  if (!ufc.form.has_vertex_integrals())
    return;

  // Set timer
  Timer timer("Assemble vertices");

  // Extract mesh
  dolfin_assert(a.mesh());
  const Mesh& mesh = *(a.mesh());

  // Compute cell and vertex - cell connectivity if not already
  // computed
  const std::size_t D = mesh.topology().dim();
  mesh.init(0);
  mesh.init(0, D);
  dolfin_assert(mesh.ordered());

  // Logics for shared vertices
  const bool has_shared_vertices = mesh.topology().have_shared_entities(0);
  const std::map<std::int32_t, std::set<unsigned int>>&
    shared_vertices = mesh.topology().shared_entities(0);

  // Form rank
  const std::size_t form_rank = ufc.form.rank();

  // Collect pointers to dof maps
  std::vector<const GenericDofMap*> dofmaps(form_rank);

  // Create a vector for storying local to local map for vertex entity
  // dofs
  std::vector<std::vector<std::size_t>> local_to_local_dofs(form_rank);

  // Create a values vector to be used to fan out local tabulated
  // values to the global tensor
  std::vector<double> local_values(1);

  // Vector to hold local dof map for a vertex
  std::vector<std::vector<dolfin::la_index>> global_dofs(form_rank);
  std::vector<ArrayView<const dolfin::la_index>> global_dofs_p(form_rank);
  std::vector<dolfin::la_index> local_dof_size(form_rank);
  for (std::size_t i = 0; i < form_rank; ++i)
  {
    dofmaps[i] = a.function_space(i)->dofmap().get();

    // Check that the test and trial space as dofs on the vertices
    if (dofmaps[i]->num_entity_dofs(0) == 0)
    {
      dolfin_error("Assembler.cpp",
                   "assemble form over vertices",
                   "Expecting test and trial spaces to have dofs on "\
                   "vertices for point integrals");
    }

    // Check that the test and trial spaces do not have dofs other
    // than on vertices
    for (std::size_t j = 1; j <= D; j++)
    {
      if (dofmaps[i]->num_entity_dofs(j)!=0)
      {
        dolfin_error("Assembler.cpp",
                     "assemble form over vertices",
                     "Expecting test and trial spaces to only have dofs on " \
                     "vertices for point integrals");
      }
    }

    // Resize local values so it can hold dofs on one vertex
    local_values.resize(local_values.size()*dofmaps[i]->num_entity_dofs(0));

    // Resize local to local map according to the number of vertex
    // entities dofs
    local_to_local_dofs[i].resize(dofmaps[i]->num_entity_dofs(0));

    // Resize local dof map vector
    global_dofs[i].resize(dofmaps[i]->num_entity_dofs(0));

    // Get size of local dofs
    local_dof_size[i] = dofmaps[i]->ownership_range().second
      - dofmaps[i]->ownership_range().first;

    // Get pointer to global dofs
    global_dofs_p[i].set(global_dofs[i]);
  }

  // Vector to hold dof map for a cell
  std::vector<ArrayView<const dolfin::la_index>> dofs(form_rank);

  // Exterior point integral
  const ufc::vertex_integral* integral
    = ufc.default_vertex_integral.get();

  // Check whether integral is domain-dependent
  bool use_domains = domains && !domains->empty();

  // MPI rank
  const unsigned int my_mpi_rank = MPI::rank(mesh.mpi_comm());

  // Assemble over vertices
  ufc::cell ufc_cell;
  std::vector<double> coordinate_dofs;
  Progress p(AssemblerBase::progress_message(A.rank(), "vertices"),
             mesh.num_vertices());
  for (VertexIterator vert(mesh); !vert.end(); ++vert)
  {
    // Get integral for sub domain (if any)
    if (use_domains)
      integral = ufc.get_vertex_integral((*domains)[*vert]);

    // Skip integral if zero
    if (!integral)
      continue;

    // Check if assembling a scalar and a vertex is shared
    if (form_rank == 0 && has_shared_vertices)
    {
      // Find shared processes for this global vertex
      std::map<std::int32_t, std::set<unsigned int>>::const_iterator e;
      e = shared_vertices.find(vert->index());

      // If vertex is shared and this rank is not the lowest do not
      // include the contribution from this vertex to scalar sum
      if (e != shared_vertices.end())
      {
        bool skip_vertex = false;
        std::set<unsigned int>::const_iterator it;
        for (it = e->second.begin(); it != e->second.end(); it++)
        {
          // Check if a shared vertex has a lower process rank
          if (*it < my_mpi_rank)
          {
            skip_vertex = true;
            break;
          }
        }

        if (skip_vertex)
          continue;
      }
    }

    // Get mesh cell to which mesh vertex belongs (pick first)
    Cell mesh_cell(mesh, vert->entities(D)[0]);

    // Check that cell is not a ghost
    dolfin_assert(!mesh_cell.is_ghost());

    // Get local index of vertex with respect to the cell
    const std::size_t local_vertex = mesh_cell.index(*vert);

    // Update UFC cell
    mesh_cell.get_cell_data(ufc_cell);
    mesh_cell.get_coordinate_dofs(coordinate_dofs);

    // Update UFC object
    ufc.update(mesh_cell, coordinate_dofs, ufc_cell,
               integral->enabled_coefficients());

    // Tabulate vertex tensor
    integral->tabulate_tensor(ufc.A.data(),
                              ufc.w(),
                              coordinate_dofs.data(),
                              local_vertex,
                              ufc_cell.orientation);

    // For rank 1 and 2 tensors we need to check if tabulated dofs for
    // the test space is within the local range
    bool owns_all_dofs = true;
    for (std::size_t i = 0; i < form_rank; ++i)
    {
      // Get local-to-global dof maps for cell
      dofs[i] = dofmaps[i]->cell_dofs(mesh_cell.index());

      // Get local dofs of the local vertex
      dofmaps[i]->tabulate_entity_dofs(local_to_local_dofs[i], 0, local_vertex);

      // Copy cell dofs to local dofs and check owner ship range
      for (std::size_t j = 0; j < local_to_local_dofs[i].size(); ++j)
      {
        global_dofs[i][j] = dofs[i][local_to_local_dofs[i][j]];

        // It is the dofs for the test space that determines if a dof
        // is owned by a process, therefore i==0
        if (i == 0 && global_dofs[i][j] >= local_dof_size[i])
        {
          owns_all_dofs = false;
          break;
        }
      }
    }

    // If not owning all dofs
    if (!owns_all_dofs)
      continue;

    // Scalar
    if (form_rank == 0)
    {
      // Add entries to global tensor
      A.add_local(ufc.A.data(), dofs);
    }
    else if (form_rank == 1)
    {
      // Copy tabulated tensor to local value vector
      for (std::size_t i = 0; i < local_to_local_dofs[0].size(); ++i)
        local_values[i] = ufc.A[local_to_local_dofs[0][i]];

      // Add local entries to global tensor
      A.add_local(local_values.data(), global_dofs_p);
    }
    else
    {
      // Copy tabulated tensor to local value vector
      const std::size_t num_cols = dofs[1].size();
      for (std::size_t i = 0; i < local_to_local_dofs[0].size(); ++i)
      {
        for (std::size_t j = 0; j < local_to_local_dofs[1].size(); ++j)
        {
          local_values[i*local_to_local_dofs[1].size() + j]
            = ufc.A[local_to_local_dofs[0][i]*num_cols
                    + local_to_local_dofs[1][j]];
        }
      }

      // Add local entries to global tensor
      A.add_local(local_values.data(), global_dofs_p);
    }

    p++;
  }
}
//-----------------------------------------------------------------------------
