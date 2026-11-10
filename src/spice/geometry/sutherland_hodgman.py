import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from spice.geometry.utils import inside, last_non_nan, append_to_last_nan, repeat_last, sort_xy


@jax.jit
def compute_intersection(p1: ArrayLike, p2: ArrayLike, q1: ArrayLike, q2: ArrayLike) -> ArrayLike:
    """Calculate the intersection of two lines

    Uses the parametric form ``x = (a (x3 - x4) - (x1 - x2) b) / d`` with
    ``a = x1 y2 - y1 x2``, ``b = x3 y4 - y3 x4`` and
    ``d = (x1 - x2)(y3 - y4) - (y1 - y2)(x3 - x4)``, which needs no special
    case for vertical edges. Parallel lines and any nan input give a nan
    point, which the clipper treats as "no vertex to add".

    The arithmetic only ever sees finite numbers: nan inputs are replaced by
    zero and the parallel denominator by one *before* the divisions, and the
    nan result is produced by a select afterwards. That keeps the reverse
    pass finite -- the clipper walks edges into its nan padding rows and,
    under ``vmap``, evaluates every ``lax.cond`` branch, so a nan or division
    by zero anywhere in this function used to leak nan cotangents into the
    real vertices and made eclipse light curves non-differentiable.

    Args:
        p1 (ArrayLike): point 1 of line 1
        p2 (ArrayLike): point 2 of line 1
        q1 (ArrayLike): point 1 of line 2
        q2 (ArrayLike): point 2 of line 2

    Returns:
        ArrayLike: coordinates of the intersection of two lines, shape (1, 2)
    """
    pts = jnp.stack([p1[:2], p2[:2], q1[:2], q2[:2]])
    bad = jnp.any(jnp.isnan(pts))
    pts = jnp.where(jnp.isnan(pts), 0.0, pts)
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = pts
    t1 = (x1 - x2) * (y3 - y4)
    t2 = (y1 - y2) * (x3 - x4)
    d = t1 - t2
    parallel = jnp.abs(d) <= 1e-12 * (jnp.abs(t1) + jnp.abs(t2))
    d_safe = jnp.where(parallel, 1.0, d)
    a = x1 * y2 - y1 * x2
    b = x3 * y4 - y3 * x4
    x = (a * (x3 - x4) - (x1 - x2) * b) / d_safe
    y = (a * (y3 - y4) - (y1 - y2) * b) / d_safe
    return jnp.where(bad | parallel, jnp.nan, jnp.array([[x, y]]))


def s_only_edge_start_inside(final_polygon, c_edge_start, c_edge_end, s_edge_start, s_edge_end):
    intersection = compute_intersection(s_edge_start,
                                        s_edge_end,
                                        c_edge_start,
                                        c_edge_end)
    return append_to_last_nan(final_polygon, intersection)

def s_only_edge_end_inside(final_polygon, c_edge_start, c_edge_end, s_edge_start, s_edge_end):
    intersection = compute_intersection(s_edge_start,
                                        s_edge_end,
                                        c_edge_start,
                                        c_edge_end)
    polygon_with_intersection = append_to_last_nan(final_polygon, intersection)
    return append_to_last_nan(polygon_with_intersection, s_edge_end)

def s_both_start_and_end_inside(final_polygon, c_edge_start, c_edge_end, s_edge_start, s_edge_end):
    return append_to_last_nan(final_polygon, s_edge_end)


@jax.jit
def clip(subject_polygon: ArrayLike, clipping_polygon: ArrayLike) -> ArrayLike:
    """Calculate the intersection of the triangles subject_polygon and clipping_polygon

    Args:
        subject_polygon (ArrayLike): x and y coordinates (3, 2)
        clipping_polygon (ArrayLike): x and y coordinates (3, 2)

    Returns:
        ArrayLike: intersection between two triangles (6, 2)
    """ 
    subject_polygon = sort_xy(subject_polygon)
    clipping_polygon = sort_xy(clipping_polygon)
    final_polygon = repeat_last(jnp.concatenate([jnp.copy(subject_polygon), jnp.nan*jnp.ones((12-subject_polygon.shape[0], 2))], axis=0))

    def outer_iteration(i, final_polygon):
        next_polygon = jnp.copy(final_polygon)

        final_polygon = jnp.nan*jnp.ones((12, 2))

        c_edge_start = jax.lax.cond(i==0,
                                    lambda: last_non_nan(clipping_polygon),
                                    lambda: clipping_polygon[i-1])
        c_edge_end = clipping_polygon[i]

        def inner_iteration(j, final_polygon):
            s_edge_start = jax.lax.cond(j==0,
                                        lambda: last_non_nan(next_polygon),
                                        lambda: next_polygon[j-1])
            s_edge_end = next_polygon[j]

            return jax.lax.cond(
                jnp.any(inside(c_edge_start, c_edge_end, s_edge_end)),
                lambda: jax.lax.cond(
                    jnp.any(inside(c_edge_start, c_edge_end, s_edge_start)),
                    lambda: s_both_start_and_end_inside(
                        final_polygon,
                        c_edge_start, c_edge_end,
                        s_edge_start, s_edge_end
                    ),
                    lambda: s_only_edge_end_inside(
                        final_polygon,
                        c_edge_start, c_edge_end,
                        s_edge_start, s_edge_end
                    ),
                ),
                lambda: jax.lax.cond(
                    jnp.any(inside(c_edge_start, c_edge_end, s_edge_start)),
                    lambda: s_only_edge_start_inside(
                        final_polygon,
                        c_edge_start, c_edge_end,
                        s_edge_start, s_edge_end
                    ),
                    lambda: final_polygon
                )
            )
        
        return jax.lax.fori_loop(0, next_polygon.shape[0], inner_iteration, final_polygon)
    return repeat_last(jax.lax.fori_loop(0, clipping_polygon.shape[0], outer_iteration, final_polygon))
