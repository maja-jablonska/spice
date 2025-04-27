import warnings
from typing import Tuple, Any
import pkgutil
import pickle
import numpy as np
import jax.numpy as jnp
import jax

# Define ArrayLike type alias since jax.typing.ArrayLike is causing issues
ArrayLike = Any

# https://sinestesia.co/blog/tutorials/python-icospheres/

PHI: float = (1 + jnp.sqrt(5)) / 2


def vertex(x: float, y: float, z: float) -> float:
    """ Return vertex coordinates fixed to the unit sphere """
    length = jnp.sqrt(jnp.power(x, 2) + jnp.power(y, 2) + jnp.power(z, 2))
    coords = jnp.array([x, y, z])
    return (coords)/length

# Number of vertices and faces, solved in wolfram

# 12 + 3*(faces-1) = 
# faces: 4*(faces-1)

# faces: f(n)=5*4^(n+1)
# vertices: v(n) = 4*(5*4^n-2)

# 0: f(0) = 4*5 = 20
# 0: v(0) = 4*(5-2) = 12

# 1: f(1) = 5*4^2 = 80
# 1: v(1) = 4*(20-2) = 72


def init_vertices_faces(subdiv: int) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:

    init_verts = jnp.array([
          vertex(-1,  PHI, 0),
          vertex( 1,  PHI, 0),
          vertex(-1, -PHI, 0),
          vertex( 1, -PHI, 0),

          vertex(0, -1, PHI),
          vertex(0,  1, PHI),
          vertex(0, -1, -PHI),
          vertex(0,  1, -PHI),

          vertex( PHI, 0, -1),
          vertex( PHI, 0,  1),
          vertex(-PHI, 0, -1),
          vertex(-PHI, 0,  1),
        ])

    init_faces = jnp.array([
             # 5 faces around point 0
             [0, 11, 5],
             [0, 5, 1],
             [0, 1, 7],
             [0, 7, 10],
             [0, 10, 11],

             # Adjacent faces
             [1, 5, 9],
             [5, 11, 4],
             [11, 10, 2],
             [10, 7, 6],
             [7, 1, 8],

             # 5 faces around 3
             [3, 9, 4],
             [3, 4, 2],
             [3, 2, 6],
             [3, 6, 8],
             [3, 8, 9],

             # Adjacent faces
             [4, 9, 5],
             [2, 4, 11],
             [6, 2, 10],
             [8, 6, 7],
             [9, 8, 1],
    ])

    verts_total_size = (4*(5*jnp.power(4, subdiv)-2)).astype(int)
    verts = jnp.zeros((verts_total_size, 3))
    verts = verts.at[:12].set(init_verts)
    verts_mask = jnp.concatenate([jnp.ones(12,), jnp.zeros(verts_total_size-12,)]).astype(jnp.int32)

    faces_total_size = int(5*4**(subdiv+1))
    faces = -1.*jnp.ones((faces_total_size, 3))
    faces = faces.at[::int(faces_total_size/20)].set(init_faces)

    return verts, verts_mask, faces


@jax.jit
def middle_point(point_1: int,
                 point_2: int,
                 verts: ArrayLike,
                 verts_mask: ArrayLike,
                 middle_point_cache: ArrayLike) -> Tuple[float,
                                                         ArrayLike,
                                                         ArrayLike,
                                                         ArrayLike]:
    """ Find a middle point and project to the unit sphere """
    # We check if we have already cut this edge first
    # to avoid duplicated verts
    smaller_index, greater_index = jax.lax.cond(point_1<point_2,
                                                lambda: (point_1.astype(int), point_2.astype(int)),
                                                lambda: (point_2.astype(int), point_1.astype(int)))

    def expand_verts(verts, verts_mask, point_1, point_2, middle_point_cache):
        # If it's not in cache, then we can cut it
        loc_verts = jnp.array([verts[point_1.astype(int)],
                               verts[point_2.astype(int)]]).T
        middle = jnp.sum(loc_verts, axis=1)/2

        index = jnp.sum(verts_mask)
        verts = verts.at[index].set(vertex(*middle))
        verts_mask = verts_mask.at[index].set(1)

        smaller_index, greater_index = jax.lax.cond(point_1<point_2,
                                                lambda: (point_1.astype(int), point_2.astype(int)),
                                                lambda: (point_2.astype(int), point_1.astype(int)))

        middle_point_cache = middle_point_cache.at[smaller_index, greater_index].set(index)

        return index, verts, verts_mask, middle_point_cache

    maybe_index = middle_point_cache[smaller_index, greater_index]
    return jax.lax.cond(maybe_index>0,
                        lambda: (maybe_index, verts, verts_mask, middle_point_cache),
                        lambda: expand_verts(verts, verts_mask, point_1, point_2, middle_point_cache))


@jax.jit
def subdivide_trie(carry: Tuple[ArrayLike, ArrayLike, ArrayLike], tri: ArrayLike) -> Tuple[ArrayLike,
                                                                                           ArrayLike,
                                                                                           ArrayLike,
                                                                                           ArrayLike]:
    verts, verts_mask, middle_point_cache = carry

    def _subdivide(verts, verts_mask, middle_point_cache, tri):
        v1, verts, verts_mask, middle_point_cache = middle_point(tri[0],
                                                                     tri[1],
                                                                     verts,
                                                                     verts_mask,
                                                                     middle_point_cache)
        v2, verts, verts_mask, middle_point_cache = middle_point(tri[1],
                                                                     tri[2],
                                                                     verts,
                                                                     verts_mask,
                                                                     middle_point_cache)
        v3, verts, verts_mask, middle_point_cache = middle_point(tri[2],
                                                                     tri[0],
                                                                     verts,
                                                                     verts_mask,
                                                                     middle_point_cache)

        return (verts, verts_mask, middle_point_cache), jnp.array([
            [tri[0], v1, v3],
            [tri[1], v2, v1],
            [tri[2], v3, v2],
            [v1, v2, v3]])

    return jax.lax.cond(jnp.all(tri==-1),
                        lambda: ((verts, verts_mask, middle_point_cache), -1.*jnp.ones((4, 3))),
                        lambda: _subdivide(verts, verts_mask, middle_point_cache, tri[0]))


def fill_placeholders(x: jnp.array, chunk_size):
    new_x = -1.*jnp.ones((chunk_size, 3))
    return new_x.at[0].set(x)


relax = jax.vmap(lambda x, n: fill_placeholders(x, n), in_axes=(0, None))


@jax.jit
def subdivide(faces: ArrayLike, verts: ArrayLike, verts_mask: ArrayLike) -> Tuple[ArrayLike,
                                                                                  ArrayLike,
                                                                                  ArrayLike]:
    verts_total_size = verts.shape[0]
    keys_cache = (-1*jnp.ones((verts_total_size, verts_total_size))).astype(int)
    (verts, verts_mask, _), new_faces = jax.lax.scan(subdivide_trie,
                                                     (verts, verts_mask, keys_cache),
                                                     faces.reshape((-1, 4, 3)))
    return new_faces.reshape((-1, 3)), verts, verts_mask


relax_all = lambda chunks, n: jax.vmap(lambda x: relax(x[:4], int(n/4)), in_axes=(0,))(chunks.reshape((-1, n, 3))).reshape((-1, 3))


@jax.jit
def face_center(verts: ArrayLike, face: ArrayLike) -> Tuple[ArrayLike, float]:
    a, b, c = verts[face[0]], verts[face[1]], verts[face[2]]
    ab = b-a
    ac = c-a
    A = jnp.linalg.norm(jnp.cross(ab, ac))/2
    return A, (a+b+c)/3


def _icosphere(subdiv: int) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    verts, verts_mask, faces = init_vertices_faces(subdiv)

    for s in range(subdiv):
        faces, verts, verts_mask = subdivide(faces, verts, verts_mask)
        current_s = int(faces.shape[0]/(5*4**(s+1)))
        faces = relax_all(faces, current_s)

    areas, centers = jax.jit(jax.vmap(face_center, in_axes=(None, 0)))(verts, faces.astype(jnp.int32))
    return verts, faces, areas, centers


def icosphere(points: int, use_cache: bool = True) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Create an icosphere_cache with at least that n points.

    Args:
        points (int): Minimal number of vertices in the icosphere_cache
        use_cache (bool): Read icospheres from pre-generated pickle files (should speed the process up)

    Returns:
        Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]: vertices (n, 3), faces (n, 3), triangle areas (n,), centers (n, 3)
    """
    subdivs = jnp.ceil(.5*jnp.log2(points/5)-1).astype(int)

    if use_cache:
        try:
            return pickle.loads(pkgutil.get_data('spice', f"icosphere_cache/icosphere_{subdivs}.pickle"))
        except FileNotFoundError:
            warnings.warn('No .pickle file for requested subdivisions found.')

    verts, faces, areas, centers = _icosphere(subdivs)
    return verts, faces, areas, centers
