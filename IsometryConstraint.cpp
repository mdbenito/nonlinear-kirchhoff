#include "IsometryConstraint.h"

#include <memory>
#include <dolfin.h>
#include <fem_utils.h>

namespace dolfin {

  IsometryConstraint::IsometryConstraint(const FunctionSpace& W)
    : _v2d(vertex_to_dof_map(W)),
      B(std::make_shared<Matrix>()), Bt(std::make_shared<Matrix>())
  {
    // Initialise B
    // ...

    const Mesh& mesh = *(W.mesh());

    // What's the difference with vertex_to_dofmap() ??
    // const auto& v2d = W.dofmap()->dofs(mesh, 0);

    auto tensor_layout = B->factory().create_layout(2);
    dolfin_assert(tensor_layout);  // what for?

    std::vector<std::shared_ptr<const IndexMap> > index_maps(2);
    index_maps.push(W.dofmap()->index_map());
    index_maps.push();

    auto local_range = W.dofmap()->ownership_range();

    tensor_layout->init(mesh.mpi_comm(), index_maps,
                        TensorLayout::Ghosts::UNGHOSTED);

    mesh.init(1, 0);  // Initialize edge -> vertex connections
  }
  
  void IsometryConstraint::update(const Function& Y)
  {
    double dofs[3];
    
    for (auto::size_t vid = 0; vid < _num_vertices; ++vid) {
      dofs[0] = _v2d[vid]; dofs[1] = _v2d[vid+1]; dofs[2] = _v2d[vid+2];
      // do stuff
      // ...
    }
  }
}
