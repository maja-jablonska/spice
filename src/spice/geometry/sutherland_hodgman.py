import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from spice.geometry.utils import inside, last_non_nan, append_to_last_nan, repeat_last, sort_xy


@jax.jit
def x_y_first_line_vertical(p1, p2, q1, q2):
    x = p1[0]
    m2 = (q2[1]-q1[1])/(q2[0]-q1[0])
    b2 = q1[1]-m2*q1[0]
    return x, m2*x+b2

@jax.jit
def x_y_second_line_vertical(p1, p2, q1, q2):
    x = q1[0]
    m1 = (p2[1]-p1[1])/(p2[0]-p1[0])
    b1 = p1[1]-m1*p1[0]
    return x, m1*x+b1

@jax.jit
def x_y_no_vertical(p1, p2, q1, q2):
    m1 = (p2[1]-p1[1])/(p2[0]-p1[0])
    b1 = p1[1]-m1*p1[0]

    m2 = (q2[1]-q1[1])/(q2[0]-q1[0])
    b2 = q1[1]-m2*q1[0]

    x = (b2-b1)/(m1-m2)
    return x, m1*x+b1


@jax.jit
def compute_intersection(p1: ArrayLike, p2: ArrayLike, q1: ArrayLike, q2: ArrayLike) -> ArrayLike:
    """Calculate the intersection of two lines

    Args:
        p1 (ArrayLike): point 1 of line 1
        p2 (ArrayLike): point 2 of line 1
        q1 (ArrayLike): point 1 of line 2
        q2 (ArrayLike): point 2 of line 2

    Returns:
        ArrayLike: coordinates of the intersection of two lines
    """
    x, y = jax.lax.cond(jnp.isclose(p1[0]-p2[0], 0),
                        x_y_first_line_vertical,
                        lambda a, b, c, d: jax.lax.cond(
                            jnp.isclose(c[0]-d[0], 0),
                            x_y_second_line_vertical,
                            x_y_no_vertical,
                            a, b, c, d),
                        p1, p2, q1, q2)
    
    return jnp.array([[x, y]])

def s_only_edge_start_inside(final_polygon, c_edge_start, c_edge_end, s_edge_start, s_edge_end):
    intersection = compute_intersection(s_edge_start,
                                        s_edge_end,
                                        c_edge_start,
                                        c_edge_end)
    intersection = jax.lax.cond(jnp.any(jnp.isnan(intersection)),
                                lambda: jnp.nan*intersection,
                                lambda: intersection)
    return append_to_last_nan(final_polygon, intersection)

def s_only_edge_end_inside(final_polygon, c_edge_start, c_edge_end, s_edge_start, s_edge_end):
    intersection = compute_intersection(s_edge_start,
                                        s_edge_end,
                                        c_edge_start,
                                        c_edge_end)
    intersection = jax.lax.cond(jnp.any(jnp.isnan(intersection)),
                            lambda: jnp.nan*intersection,
                            lambda: intersection)
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

        #c_edge_start = clipping_polygon[((i-1)%(clipping_polygon.shape[0])).astype(int)]
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
