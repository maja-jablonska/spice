import dataclasses
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from .utils import cast_to_los, cast_normalized_to_los, cast_to_normal_plane
from .mesh_model import MeshModel
from geometry import clip, polygon_area
from geometry.utils import get_cast_areas, cast_indexes, polygon_areas, append_value_to_last_nan
from typing import Any, NamedTuple
from functools import partial
from jax.tree_util import register_pytree_node_class


def register_pytree_node_dataclass(cls):
  _flatten = lambda obj: jax.tree_flatten(dataclasses.asdict(obj))
  _unflatten = lambda d, children: cls(**d.unflatten(children))
  jax.tree_util.register_pytree_node(cls, _flatten, _unflatten)
  return cls


@register_pytree_node_class
class Grid:
    
    def __init__(self,
                 x: ArrayLike,
                 y: ArrayLike,
                 x_span: float,
                 y_span: float,
                 grid: ArrayLike,
                 n_cells: int,
                 n_centers_m1: int,
                 n_centers_m2: int):
        self.x = x
        self.y = y
        self.x_span = x_span
        self.y_span = y_span
        self.grid = grid
        self.n_cells = n_cells
        self.n_centers_m1 = n_centers_m1
        self.n_centers_m2 = n_centers_m2
        
    @classmethod
    def construct(cls, m1: MeshModel, m2: MeshModel, n_cells: int):
        vs1, vs2 = m1.cast_vertices[m1.faces.astype(int)], m2.cast_vertices[m2.faces.astype(int)]
        vs1, vs2 = vs1[cast_indexes(vs1)], vs2[cast_indexes(vs2)]
        
        x_range = jnp.linspace(jnp.min(jnp.concatenate([vs1[:, :, 0], vs2[:, :, 0]])), jnp.max(jnp.concatenate([vs1[:, :, 0], vs2[:, :, 0]])), n_cells)
        y_range = jnp.linspace(jnp.min(jnp.concatenate([vs1[:, :, 1], vs2[:, :, 1]])), jnp.max(jnp.concatenate([vs1[:, :, 1], vs2[:, :, 1]])), n_cells)

        nx, ny = jnp.meshgrid(x_range, y_range)
        x, y = jnp.meshgrid(x_range, y_range, sparse=True)
        return cls(
            x.flatten(), y.flatten(),
            x_range[1]-x_range[0],
            y_range[1]-y_range[0],
            jnp.vstack([nx.ravel(), ny.ravel()]).T,
            n_cells,
            m1.cast_centers.shape[0],
            m2.cast_centers.shape[0]
        )
        
    def __hash__(self) -> int:
        return self.n_cells

    def __eq__(self, __value: object) -> bool:
        return self.__hash__()==__value.__hash__()
    
    def __repr__(self) -> str:
        return "Grid(n_cells={})".format(self.n_cells)
    
    def tree_flatten(self):
        return (self.grid, self.x, self.y, self.x_span, self.y_span, self.n_cells, self.n_centers_m1, self.n_centers_m2), None
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    
    
def some_hash_function(x):
  return int(jnp.sum(x))
    
    
class HashableArrayWrapper:
  def __init__(self, val):
    self.val = val
  def __hash__(self):
    return some_hash_function(self.val)
  def __eq__(self, other):
    return (isinstance(other, HashableArrayWrapper) and
            jnp.all(jnp.equal(self.val, other.val)))

def gnool_jit(fun, static_array_argnums=()):
  @partial(jax.jit, static_argnums=static_array_argnums)
  def callee(*args):
    args = list(args)
    for i in static_array_argnums:
      args[i] = args[i].val
    return fun(*args)

  def caller(*args):
    args = list(args)
    for i in static_array_argnums:
      args[i] = HashableArrayWrapper(args[i])
    return callee(*args)

  return caller


@jax.jit
def get_mesh_view(mesh: MeshModel, los_vector: ArrayLike) -> MeshModel:
    """Cast 3D vectors of centers and center velocities to the line-of-sight

    Args:
        mesh (MeshModel): Properties to be casted (n, 3)
        los_vector (ArrayLike): LOS vector (3,)

    Returns:
        MeshModel: mesh with updated los_vector, mus, and los_velocities
    """
    cast_vertices=cast_to_normal_plane(mesh.vertices, los_vector)
    return mesh._replace(
        los_vector = los_vector,
        los_z=cast_to_los(mesh.center+mesh.d_centers, los_vector),
        cast_vertices = cast_vertices,
        cast_centers = cast_to_normal_plane(mesh.centers, los_vector),
        mus = cast_normalized_to_los(mesh.d_centers, los_vector),
        los_velocities = cast_to_los(mesh.velocities, los_vector),
        cast_areas=get_cast_areas(cast_vertices[mesh.faces.astype(int)])
    )

@jax.jit
def visible_area(vertices1: ArrayLike, vertices2: ArrayLike) -> ArrayLike:
    clipped = jnp.nan_to_num(clip(vertices1, vertices2))
    return polygon_area(clipped[:, 0], clipped[:, 1])

total_visible_area = jax.jit(jax.vmap(visible_area, in_axes=(None, 0)))

visibility_areas = jax.jit(jax.vmap(total_visible_area, in_axes=(0, None)))

@jax.jit
def construct_grid(m1: MeshModel, m2: MeshModel) -> Grid:
    N_GRID = 20
    vs1, vs2 = m1.cast_vertices[m1.faces.astype(int)], m2.cast_vertices[m2.faces.astype(int)]
    vs1, vs2 = vs1[cast_indexes(vs1)], vs2[cast_indexes(vs2)]
    
    x_range = jnp.linspace(jnp.min(jnp.concatenate([vs1[:, :, 0], vs2[:, :, 0]])), jnp.max(jnp.concatenate([vs1[:, :, 0], vs2[:, :, 0]])), N_GRID)
    y_range = jnp.linspace(jnp.min(jnp.concatenate([vs1[:, :, 1], vs2[:, :, 1]])), jnp.max(jnp.concatenate([vs1[:, :, 1], vs2[:, :, 1]])), N_GRID)

    nx, ny = jnp.meshgrid(x_range, y_range)
    x, y = jnp.meshgrid(x_range, y_range, sparse=True)
    x, y = x.flatten(), y.flatten()
    x_span = x_range[1]-x_range[0]
    y_span = y_range[1]-y_range[0]
    return Grid(**{
            'grid': jnp.vstack([nx.ravel(), ny.ravel()]).T,
            'x': x,
            'y': y,
            'x_span': x_span,
            'y_span': y_span,
            'n_cells': N_GRID,
            'n_centers_m1': m1.cast_centers.shape[0],
            'n_centers_m2': m2.cast_centers.shape[0]
        })

@jax.jit
def get_grid_index(grid: Grid, cast_point: ArrayLike) -> ArrayLike:
    x_grid = jnp.where(
            jnp.isclose((cast_point[0]-grid.x)//grid.x_span, 0.),
            cast_point[0],
            jnp.nan
        )
    y_grid = jnp.where(
            jnp.isclose((cast_point[1]-grid.y)//grid.y_span, 0.),
            cast_point[1],
            jnp.nan
        )
    return (
        jnp.max(jnp.argwhere(~jnp.isnan(x_grid), size=x_grid.shape[0], fill_value=0)),
        jnp.max(jnp.argwhere(~jnp.isnan(y_grid), size=y_grid.shape[0], fill_value=0))
    )
    
    
@jax.jit
def assign_element_to_grid(i: int, carry: ArrayLike, m: MeshModel):
    def assign_element_to_grid1_pos_mu(i: int, carry):
        grid, grid_points, reverse_grid = carry
        grid_index_x, grid_index_y = get_grid_index(grid, m.cast_centers[i, 1:])
        grid_points = grid_points.at[grid_index_x, grid_index_y].set(append_value_to_last_nan(grid_points[grid_index_x, grid_index_y], i))
        reverse_grid = reverse_grid.at[i].set(jnp.array([grid_index_x, grid_index_y]))
        return grid, grid_points, reverse_grid
    def assign_element_to_grid1_neg_mu(i: int, carry):
        grid, grid_points, reverse_grid = carry
        return grid, grid_points, reverse_grid.at[i].set(jnp.array([jnp.nan, jnp.nan]))
    return jax.lax.cond(jnp.all(m.mus[i]>0),
                        assign_element_to_grid1_pos_mu,
                        assign_element_to_grid1_neg_mu,
                        i, carry)
    
@partial(jax.jit, static_argnums=(2,))
def create_grid_dictionaries(m1: MeshModel, m2: MeshModel, grid: Grid):
    # N_POINTS = jnp.ceil(4*(m1.cast_centers.shape[0]/(grid_cells*grid_cells))).astype(int)
    # N_POINTS2 = jnp.ceil(4*(m2.cast_centers.shape[0]/(grid_cells*grid_cells))).astype(int)
    grid_cells = grid.n_cells
    grids_m1 = jnp.nan*jnp.ones((grid_cells, grid_cells, grid_cells))
    reverse_grids_m1 = jnp.nan*jnp.ones((grid.n_centers_m1, 2))
    grids_m2 = jnp.nan*jnp.ones((grid_cells, grid_cells, grid_cells))
    reverse_grids_m2 = jnp.nan*jnp.ones((grid.n_centers_m2, 2))

    return (
        jax.lax.fori_loop(0, len(m1.cast_centers), lambda i, c: assign_element_to_grid(i, c, m1), (grid, grids_m1, reverse_grids_m1)),
        jax.lax.fori_loop(0, len(m2.cast_centers), lambda i, c: assign_element_to_grid(i, c, m2), (grid, grids_m2, reverse_grids_m2))
    )
    

@jax.jit
def get_neighbouring(x: int, y: int):
    
    neighbours = jnp.array([[x, y], [x-1, y-1], [x-1, y], [x-1, y+1],
                            [x, y-1], [x, y+1],
                            [x+1, y-1], [x+1, y], [x+1, y+1]])
    return neighbours


@partial(jax.jit, static_argnums=(3,))
def resolve_occlusion_for_face(m1: MeshModel, m2: MeshModel, face_index: int, grid: Grid):
    (_, _, reverse_grids_m1), (_, grids_m2, _) = create_grid_dictionaries(m1, m2, grid)
    ix, iy = jnp.nan_to_num(reverse_grids_m1[face_index], reverse_grids_m1.shape[0]+1).astype(int)
    grid_neighbours = get_neighbouring(ix, iy)
    points_in_grid = (grids_m2[grid_neighbours[:, 0], grid_neighbours[:, 1]]).flatten()
    points_mask = jnp.where(jnp.isnan(points_in_grid), 0., 1.)*(m1.mus[face_index]>0).astype(float)
    return jnp.sum(total_visible_area(m1.cast_vertices[m1.faces[face_index].astype(int)][:, [1, 2]],
                                      m2.cast_vertices[m2.faces[points_in_grid.astype(int)].astype(int)][:, :, [1, 2]])*points_mask)
    
v_resolve_occlusion_for_face = jax.jit(jax.vmap(resolve_occlusion_for_face, in_axes=(None, None, 0, None)), static_argnums=(3,))
    
@partial(jax.jit, static_argnums=(2,))
def resolve_occlusion(m1: MeshModel, m2: MeshModel, grid: Grid) -> ArrayLike:
    cast_mesh2_center = cast_to_normal_plane(m2.center, m1.los_vector)
    cast_mesh2_center = cast_mesh2_center[0, cast_indexes(cast_mesh2_center)]
    nonzero_indexer = cast_indexes(m1.cast_centers)
    mean_separation = jnp.mean(jnp.sqrt(2*m1.areas))
    o = v_resolve_occlusion_for_face(m1, m2, jnp.arange(len(m1.faces)), grid)
    jax.debug.print("{a}", a = (jnp.sqrt(jnp.sum(jnp.square(m1.cast_vertices[m1.faces.astype(int)][:, :, nonzero_indexer]-cast_mesh2_center), axis=2))).shape)
    #return (jnp.sqrt(jnp.sum(jnp.square(m1.cast_centers[:, nonzero_indexer]-cast_mesh2_center), axis=1))<(m2.radius-mean_separation)).astype(float)
    return m1._replace(
        cast_areas=jnp.where(
            jnp.sqrt(jnp.sum(jnp.square(m1.cast_centers[:, nonzero_indexer]-cast_mesh2_center), axis=1))<(m2.radius-1.1*mean_separation),
            0.,
            m1.cast_areas-o
        ))
    # cast_mesh2_center = cast_to_normal_plane(mesh2.center, mesh1.los_vector)
    # cast_mesh2_center = cast_mesh2_center[0, cast_indexes(cast_mesh2_center)]
    # mean_separation = jnp.mean(jnp.sqrt(2*mesh1.areas))

    # face_vertices1 = mesh1.cast_vertices[mesh1.faces.astype(int)]
    # face_vertices2 = mesh2.cast_vertices[mesh2.faces.astype(int)]
    # nonzero_indexer = cast_indexes(mesh1.cast_vertices)
    # cast_vertices1 = face_vertices1[:, :, nonzero_indexer]
    # cast_vertices2 = face_vertices2[:, :, nonzero_indexer]
    # occluded = jnp.where(mesh2.mus>0,
    #                              visibility_areas(cast_vertices1, cast_vertices2,
    #                                               cast_mesh2_center, mesh2.radius, mean_separation),
    #                              0.)
    # return mesh1._replace(
    #         cast_areas=jnp.where(mesh1.mus>0, mesh1.cast_areas-occluded, 0)
    #     )


# @jax.jit
# def resolve_occlusion(mesh1: MeshModel, mesh2: MeshModel) -> ArrayLike:
#     cast_mesh2_center = cast_to_normal_plane(mesh2.center, mesh1.los_vector)
#     cast_mesh2_center = cast_mesh2_center[0, cast_indexes(cast_mesh2_center)]
#     mean_separation = jnp.mean(jnp.sqrt(2*mesh1.areas))

#     face_vertices1 = mesh1.cast_vertices[mesh1.faces.astype(int)]
#     face_vertices2 = mesh2.cast_vertices[mesh2.faces.astype(int)]
#     nonzero_indexer = cast_indexes(mesh1.cast_vertices)
#     cast_vertices1 = face_vertices1[:, :, nonzero_indexer]
#     cast_vertices2 = face_vertices2[:, :, nonzero_indexer]
#     occluded = jnp.where(mesh2.mus>0,
#                                  visibility_areas(cast_vertices1, cast_vertices2,
#                                                   cast_mesh2_center, mesh2.radius, mean_separation),
#                                  0.)
#     return mesh1._replace(
#             cast_areas=jnp.where(mesh1.mus>0, mesh1.cast_areas-occluded, 0)
#         )
