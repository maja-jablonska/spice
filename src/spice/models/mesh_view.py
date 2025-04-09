from typing import Tuple, Any

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from .mesh_model import MeshModel
from spice.geometry import clip, polygon_area
from spice.geometry.utils import append_value_to_last_nan
from functools import partial
from jax.tree_util import register_pytree_node_class

from jaxtyping import Array, Float


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
                 n_centers_m2: int,
                 n_points_m1: int,
                 n_points_m2: int):
        self.x = x
        self.y = y
        self.x_span = x_span
        self.y_span = y_span
        self.grid = grid
        self.n_cells = n_cells
        self.n_centers_m1 = n_centers_m1
        self.n_centers_m2 = n_centers_m2
        self.n_points_m1 = n_points_m1
        self.n_points_m2 = n_points_m2

    @classmethod
    def construct(cls, m1: MeshModel, m2: MeshModel, n_cells: int):
        vs1, vs2 = m1.cast_vertices[m1.faces.astype(int)], m2.cast_vertices[m2.faces.astype(int)]

        x_range = jnp.linspace(jnp.min(jnp.concatenate([vs1[:, 0], vs2[:, 0]])),
                               jnp.max(jnp.concatenate([vs1[:, 0], vs2[:, 0]])), n_cells)
        y_range = jnp.linspace(jnp.min(jnp.concatenate([vs1[:, 1], vs2[:, 1]])),
                               jnp.max(jnp.concatenate([vs1[:, 1], vs2[:, 1]])), n_cells)

        nx, ny = jnp.meshgrid(x_range, y_range)
        x, y = jnp.meshgrid(x_range, y_range, sparse=True)
        x, y = x.flatten(), y.flatten()
        return cls(
            x=x,
            y=y,
            x_span=x[1] - x[0],
            y_span=y[1] - y[0],
            grid=jnp.vstack([nx.ravel(), ny.ravel()]).T,
            n_cells=n_cells,
            n_centers_m1=m1.cast_centers.shape[0],
            n_centers_m2=m2.cast_centers.shape[0],
            n_points_m1=jnp.ceil(4 * m1.cast_centers.shape[0] / (n_cells * n_cells)).astype(int).item(),
            n_points_m2=jnp.ceil(4 * m2.cast_centers.shape[0] / (n_cells * n_cells)).astype(int).item()
        )

    def __hash__(self) -> int:
        return self.n_cells

    def __eq__(self, __value: object) -> bool:
        return self.__hash__() == __value.__hash__()

    def __repr__(self) -> str:
        return "Grid(n_cells={})".format(self.n_cells)

    def tree_flatten(self):
        return (self.x, self.y, self.x_span, self.y_span, self.grid, self.n_cells,
                self.n_centers_m1, self.n_centers_m2, self.n_points_m1, self.n_points_m2), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
    
    
@jax.jit
def get_grid_spans(m1, m2, n_cells_array):
    """Calculate grid cell spans for different grid sizes.
    
    For each number of cells in n_cells_array, calculates the span (width/height) of grid cells
    that would cover the projected area of both meshes. Returns the minimum of x and y spans
    to ensure square grid cells.
    
    Args:
        m1 (MeshModel): First mesh model with cast_vertices and faces
        m2 (MeshModel): Second mesh model with cast_vertices and faces 
        n_cells_array (ArrayLike): Array of different grid cell counts to try
        
    Returns:
        ArrayLike: Array of grid cell spans corresponding to each n_cells value
    """
    vs1, vs2 = m1.cast_vertices[m1.faces.astype(int)], m2.cast_vertices[m2.faces.astype(int)]

    x_min = jnp.min(jnp.concatenate([vs1[:, 0], vs2[:, 0]]))
    x_max = jnp.max(jnp.concatenate([vs1[:, 0], vs2[:, 0]]))
    y_min = jnp.min(jnp.concatenate([vs1[:, 1], vs2[:, 1]]))
    y_max = jnp.max(jnp.concatenate([vs1[:, 1], vs2[:, 1]]))
    
    x_spans = (x_max - x_min) / n_cells_array
    y_spans = (y_max - y_min) / n_cells_array
    
    return jnp.minimum(x_spans, y_spans)


@jax.jit
def get_mesh_view(mesh: MeshModel, los_vector: Float[Array, "3"]) -> MeshModel:
    """Cast 3D vectors of centers and center velocities to the line-of-sight

    Args:
        mesh (MeshModel): Properties to be cast (n, 3)
        los_vector (Float[Array, "3"]): LOS vector (3,)

    Returns:
        MeshModel: mesh with updated los_vector, mus, and los_velocities
    """
    return mesh._replace(
        los_vector=los_vector
    )


@jax.jit
def visible_area(vertices1: Float[Array, "n_vertices 3"], vertices2: Float[Array, "n_vertices 3"]) -> Float[Array, "n_vertices"]:
    clipped = clip(vertices1, vertices2)
    
    # Create mask for valid vertices (1=valid, 0=NaN)
    mask = ~jnp.any(jnp.isnan(clipped), axis=1)
    
    # Replace NaN values with zeros for safe calculations
    clipped_safe = jnp.where(jnp.isnan(clipped), 0.0, clipped)
    
    # Get coordinates and handle wrap-around
    x = clipped_safe[:, 0]
    y = clipped_safe[:, 1]
    next_idx = jnp.roll(jnp.arange(x.shape[0]), -1)  # Circular next index
    
    # Calculate contribution for each edge pair, masked by validity
    terms = (x * y[next_idx] - x[next_idx] * y) * mask * mask[next_idx]
    
    return 0.5 * jnp.abs(jnp.sum(terms))


total_visible_area = jax.jit(jax.vmap(visible_area, in_axes=(None, 0)))

visibility_areas = jax.jit(jax.vmap(total_visible_area, in_axes=(0, None)))


@jax.jit
def get_grid_index(grid: Grid, cast_point: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    x_grid = jnp.where(
        jnp.isclose((cast_point[0] - grid.x) // grid.x_span, 0.),
        cast_point[0],
        jnp.nan
    )
    y_grid = jnp.where(
        jnp.isclose((cast_point[1] - grid.y) // grid.y_span, 0.),
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
        grid_index_x, grid_index_y = get_grid_index(grid, m.cast_centers[i])
        grid_points = grid_points.at[grid_index_x, grid_index_y].set(
            append_value_to_last_nan(grid_points[grid_index_x, grid_index_y], i))
        reverse_grid = reverse_grid.at[i].set(jnp.array([grid_index_x, grid_index_y]))
        return grid, grid_points, reverse_grid

    def assign_element_to_grid1_neg_mu(i: int, carry):
        grid, grid_points, reverse_grid = carry
        return grid, grid_points, reverse_grid.at[i].set(jnp.array([jnp.nan, jnp.nan]))

    return jax.lax.cond(jnp.all(m.mus[i] > 0),
                        assign_element_to_grid1_pos_mu,
                        assign_element_to_grid1_neg_mu,
                        i, carry)


@partial(jax.jit, static_argnums=(2,))
def create_grid_dictionaries(m1: MeshModel, m2: MeshModel, grid: Grid):
    grids_m1 = jnp.nan * jnp.ones((grid.x.shape[0], grid.y.shape[0], grid.n_points_m1))
    reverse_grids_m1 = jnp.nan * jnp.ones((grid.n_centers_m1, 2))
    grids_m2 = jnp.nan * jnp.ones((grid.x.shape[0], grid.y.shape[0], grid.n_points_m2))
    reverse_grids_m2 = jnp.nan * jnp.ones((grid.n_centers_m2, 2))

    return (
        jax.lax.fori_loop(0, len(m1.cast_centers), lambda i, c: assign_element_to_grid(i, c, m1),
                          (grid, grids_m1, reverse_grids_m1)),
        jax.lax.fori_loop(0, len(m2.cast_centers), lambda i, c: assign_element_to_grid(i, c, m2),
                          (grid, grids_m2, reverse_grids_m2))
    )


@jax.jit
def get_neighbouring(x: int, y: int):
    neighbours = jnp.array([[x, y], [x - 1, y - 1], [x - 1, y], [x - 1, y + 1],
                            [x, y - 1], [x, y + 1],
                            [x + 1, y - 1], [x + 1, y], [x + 1, y + 1]])
    return neighbours


@partial(jax.jit, static_argnums=(3,))
def resolve_occlusion_for_face(m1: MeshModel, m2: MeshModel, face_index: int, grid: Grid):
    (_, _, reverse_grids_m1), (_, grids_m2, _) = create_grid_dictionaries(m1, m2, grid)
    ix, iy = jnp.nan_to_num(reverse_grids_m1[face_index], reverse_grids_m1.shape[0] + 1).astype(int)
    grid_neighbours = get_neighbouring(ix, iy)
    points_in_grid = (grids_m2[grid_neighbours[:, 0], grid_neighbours[:, 1]]).flatten()
    points_mask = jnp.where(jnp.isnan(points_in_grid), 0., 1.) * (m1.mus[face_index] > 0).astype(float)
    # Calculate occlusion for each neighboring point
    occlusions = total_visible_area(m1.cast_vertices[m1.faces[face_index].astype(int)],
                                  m2.cast_vertices[m2.faces[points_in_grid.astype(int)].astype(int)]) * points_mask
    
    total_occlusion = jnp.clip(jnp.sum(occlusions), 0., m1.cast_areas[face_index])
    return total_occlusion


v_resolve_occlusion_for_face = jax.jit(jax.vmap(resolve_occlusion_for_face, in_axes=(None, None, 0, None)),
                                       static_argnums=(3,))


@partial(jax.jit, static_argnums=(2,))
def resolve_occlusion(m1: MeshModel, m2: MeshModel, grid: Grid) -> MeshModel:
    """Calculate the occlusion of m1 by m2

    Args:
        m1 (MeshModel): occluded mesh model
        m2 (MeshModel): occluding mesh model
        grid (Grid): grid for calculation optimization

    Returns:
        MeshModel: m1 with updated visible areas
    """
    o = v_resolve_occlusion_for_face(m1, m2, jnp.arange(len(m1.faces)), grid)
    return m1._replace(
        occluded_areas=o
    )
