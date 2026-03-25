import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def sort_xy(points: ArrayLike) -> ArrayLike:
    """Sort points clockwise

    Args:
        points (ArrayLike): x, y coordinates (n, 2)

    Returns:
        ArrayLike: points with x, y coordinates sorted clockwise
    """
    x, y = points[:, 0], points[:, 1]
    x0 = jnp.mean(x)
    y0 = jnp.mean(y)

    r = jnp.sqrt((x-x0)**2 + (y-y0)**2)

    angles = jnp.where((y-y0) > 0, jnp.arccos((x-x0)/r), 2*jnp.pi-jnp.arccos((x-x0)/r))

    mask = jnp.argsort(angles)

    return points[mask[::-1], :] # clockwise order



def inside(p1: ArrayLike, p2: ArrayLike, q: ArrayLike) -> bool:
    """Check if the point is inside the polygon edge

    Args:
        p1 (ArrayLike): Polygon edge point 1
        p2 (ArrayLike): Polygon edge point 2
        q (ArrayLike): Point to be checked

    Returns:
        bool: Point is inside polygon
    """
    cross = (p2[0]-p1[0])*(q[1]-p1[1]) - (p2[1]-p1[1])*(q[0]-p1[0])
    return cross <= 0  # clockwise


def last_non_nan_arg(arr: ArrayLike) -> ArrayLike:
    """Return the last non-nan index of the array

    Args:
        arr (ArrayLike): Array

    Returns:
        ArrayLike: Last non-nan index of the array
    """    
    return jnp.max(jnp.argwhere(jnp.all(~jnp.isnan(arr), axis=1), size=arr.shape[0], fill_value=0)).astype(int)


def last_non_nan(arr: ArrayLike, axis=1) -> ArrayLike:
    """Return the last non-nan value of the array

    Args:
        arr (ArrayLike): Array

    Returns:
        ArrayLike: Last non-nan row of the array
    """    
    return arr[jnp.max(jnp.argwhere(jnp.all(~jnp.isnan(arr), axis=axis), size=arr.shape[0], fill_value=0))]


def append_to_last_nan(arr: ArrayLike, to_add: ArrayLike) -> ArrayLike:
    """Append the row to the first nan row

    Args:
        arr (ArrayLike): Array to add the row to
        to_add (ArrayLike): The row to be added

    Returns:
        ArrayLike: Array with an appended row
    """    
    next_ind = jnp.min(jnp.argwhere(jnp.all(jnp.isnan(arr), axis=1), size=arr.shape[0], fill_value=arr.shape[0]-1))
    return arr.at[next_ind].set(to_add.flatten())


def append_value_to_last_nan(arr: ArrayLike, to_add: ArrayLike) -> ArrayLike:
    """Append a single value to the first nan in array

    Args:
        arr (ArrayLike): Array to add the value to
        to_add (ArrayLike): The value to be appended

    Returns:
        ArrayLike: Array with an appended value
    """    
    next_ind = jnp.min(jnp.argwhere(jnp.isnan(arr), size=arr.shape[0], fill_value=arr.shape[0]-1))
    return arr.at[next_ind].set(to_add)


def append_value_to_last_nan_2d(arr: ArrayLike, to_add: ArrayLike) -> ArrayLike:
    """Append a single value to the first nan in array

    Args:
        arr (ArrayLike): Array to add the value to
        to_add (ArrayLike): The value to be appended

    Returns:
        ArrayLike: Array with an appended value
    """
    next_ind = jnp.min(jnp.argwhere(jnp.all(jnp.isnan(arr), axis=1), size=arr.shape[0], fill_value=arr.shape[0]-1))
    return arr.at[next_ind].set(to_add)


def repeat_last(arr: ArrayLike) -> ArrayLike:
    """Repeat the last non-nan row to fill the whole array

    Args:
        arr (ArrayLike):

    Returns:
        ArrayLike: Array with nan values filled with last non-nan values
    """    
    return jax.lax.cond(jnp.any(jnp.all(jnp.isnan(arr), axis=1)),
                        lambda: jax.lax.fori_loop(0, jnp.sum(jnp.all(jnp.isnan(arr), axis=1)),
                                                  lambda i, _arr: append_to_last_nan(_arr, last_non_nan(_arr)),
                                                  arr),
                        lambda: arr)


def wrap(arr: ArrayLike) -> ArrayLike:
    """Copy the first row and concatenate to the first array

    Args:
        arr (ArrayLike):

    Returns:
        ArrayLike: Array with the copy of the first row appended
    """    
    return jnp.concatenate([arr, arr[0].reshape((-1, arr.shape[1]))])


def polygon_area(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """Calculate a polygon's surface area from its x and y coordinates

    Args:
        x (ArrayLike): x coordinates of the polygon
        y (ArrayLike): y coordinates of the polygon

    Returns:
        ArrayLike: polygon's surface area
    """    
    return 0.5*jnp.abs(jnp.dot(x,jnp.roll(y,1))-jnp.dot(y,jnp.roll(x,1)))


polygon_areas = jax.jit(jax.vmap(polygon_area, in_axes=(0, 0)))

def cast_indexes(cast_vertices: ArrayLike) -> ArrayLike:
    """When casting 3D vectors to 2D, get the indices of non-reduced dimensions

    Args:
        cast_vertices (ArrayLike): vertices after 3D to 2D casting

    Returns:
        ArrayLike: non-reduced (non-zero) dimensions
    """
    zeroth_ind = jnp.argmax(jnp.all(jnp.isclose(cast_vertices, 0), axis=1))

    return jax.lax.cond(zeroth_ind == 0,
                        lambda: jnp.array([1, 2]),
                        lambda: jax.lax.cond(zeroth_ind == 1,
                                     lambda: jnp.array([0, 2]),
                                     lambda: jnp.array([0, 1])))


def get_cast_areas(face_vertices: ArrayLike) -> ArrayLike:
    """Get visible, 2D-casted polygon areas

    Args:
        face_vertices (ArrayLike): polygons' vertices after 3D to 2D casting

    Returns:
        ArrayLike: areas of 2D-casted polygons
    """
    return polygon_areas(face_vertices[:, :, 0],
                         face_vertices[:, :, 1])
