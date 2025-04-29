"""
Mesh view utilities for occlusion detection using KD-trees.

This module provides utilities for mesh occlusion detection using KD-trees instead of a grid-based
approach. KD-trees offer better performance for spatial queries compared to a uniform grid,
especially for non-uniform distributions of mesh elements.
"""

from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from spice.models.mesh_model import MeshModel
from spice.geometry import clip
from functools import partial
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class KDTree:
    """KD-tree for efficient spatial queries on 2D projected vertices."""
    
    def __init__(
        self,
        points: Float[Array, "n_points 2"],
        indices: Float[Array, "n_points"],
        node_indices: Float[Array, "n_nodes 2"],  # start_idx, end_idx
        node_split_dims: Float[Array, "n_nodes"],
        node_split_values: Float[Array, "n_nodes"],
        node_children: Float[Array, "n_nodes 2"],  # left_child, right_child
        leaf_size: int = 10
    ):
        self.points = points
        self.indices = indices
        self.node_indices = node_indices
        self.node_split_dims = node_split_dims
        self.node_split_values = node_split_values
        self.node_children = node_children
        self.leaf_size = leaf_size
        
    @classmethod
    def construct(cls, points: Float[Array, "n_points 2"], indices: Optional[Float[Array, "n_points"]] = None, leaf_size: int = 10):
        """Construct a KD-tree from a set of 2D points.
        
        Args:
            points: Array of 2D points
            indices: Optional array of indices corresponding to the points
            leaf_size: Maximum number of points in a leaf node
            
        Returns:
            KDTree: A constructed KD-tree
        """
        n_points = points.shape[0]
        
        if indices is None:
            indices = jnp.arange(n_points)
            
        # Pre-allocate for worst-case tree size (completely unbalanced)
        max_nodes = 2 * n_points // leaf_size + 1
        
        # Initialize arrays for tree storage
        node_indices = jnp.zeros((max_nodes, 2), dtype=jnp.int32)
        node_split_dims = jnp.zeros(max_nodes, dtype=jnp.int32)
        node_split_values = jnp.zeros(max_nodes, dtype=jnp.float32)
        node_children = jnp.full((max_nodes, 2), -1, dtype=jnp.int32)  # -1 indicates no child
        
        # Build tree recursively
        def _build_tree(start_idx, end_idx, node_idx, depth, state):
            """Recursive tree building function"""
            node_indices, node_split_dims, node_split_values, node_children, next_node_idx = state
            
            # Update current node's range
            node_indices = node_indices.at[node_idx].set(jnp.array([start_idx, end_idx]))
            
            # If leaf node, return
            if end_idx - start_idx <= leaf_size:
                return node_indices, node_split_dims, node_split_values, node_children, next_node_idx
            
            # Choose split dimension (alternate dimensions)
            split_dim = depth % 2
            node_split_dims = node_split_dims.at[node_idx].set(split_dim)
            
            # Sort points along split dimension
            point_indices = indices[start_idx:end_idx]
            split_values = points[point_indices, split_dim]
            sorted_indices = point_indices[jnp.argsort(split_values)]
            indices = indices.at[start_idx:end_idx].set(sorted_indices)
            
            # Find median for split
            median_idx = start_idx + (end_idx - start_idx) // 2
            split_value = points[indices[median_idx], split_dim]
            node_split_values = node_split_values.at[node_idx].set(split_value)
            
            # Create child nodes
            left_child = next_node_idx
            right_child = left_child + 1
            node_children = node_children.at[node_idx].set(jnp.array([left_child, right_child]))
            next_node_idx += 2
            
            # Recursively build left and right subtrees
            state = node_indices, node_split_dims, node_split_values, node_children, next_node_idx
            state = _build_tree(start_idx, median_idx, left_child, depth + 1, state)
            state = _build_tree(median_idx, end_idx, right_child, depth + 1, state)
            
            return state
        
        # Build the tree
        init_state = (node_indices, node_split_dims, node_split_values, node_children, 1)  # Start with node_idx 0
        final_state = jax.lax.fori_loop(
            0, 1,  # Just one iteration to build the root
            lambda _, state: _build_tree(0, n_points, 0, 0, state),
            init_state
        )
        
        node_indices, node_split_dims, node_split_values, node_children, next_node_idx = final_state
        
        # Trim arrays to actual size
        node_indices = node_indices[:next_node_idx]
        node_split_dims = node_split_dims[:next_node_idx]
        node_split_values = node_split_values[:next_node_idx]
        node_children = node_children[:next_node_idx]
        
        return cls(
            points=points,
            indices=indices,
            node_indices=node_indices,
            node_split_dims=node_split_dims,
            node_split_values=node_split_values,
            node_children=node_children,
            leaf_size=leaf_size
        )
    
    def query_radius(self, query_point: Float[Array, "2"], radius: float) -> Float[Array, "n_neighbors"]:
        """Find all points within a given radius of the query point.
        
        Args:
            query_point: The 2D query point
            radius: Search radius
            
        Returns:
            Array of indices of the points within the radius
        """
        def _search_node(node_idx, neighbors):
            """Recursive search function"""
            # Check if node is a leaf
            is_leaf = node_children[node_idx, 0] == -1
            
            def _process_leaf():
                # Get indices for this leaf
                start_idx, end_idx = node_indices[node_idx]
                leaf_point_indices = indices[start_idx:end_idx]
                
                # Compute distances to query point
                leaf_points = points[leaf_point_indices]
                distances = jnp.sqrt(jnp.sum((leaf_points - query_point) ** 2, axis=1))
                
                # Add points within radius to neighbors list
                mask = distances <= radius
                valid_indices = leaf_point_indices[mask]
                
                # Append valid indices to neighbors
                n_valid = jnp.sum(mask)
                return neighbors.at[:n_valid].set(valid_indices)
            
            def _process_internal():
                # Get split information
                split_dim = node_split_dims[node_idx]
                split_value = node_split_values[node_idx]
                
                # Calculate distance to splitting hyperplane
                dist_to_plane = query_point[split_dim] - split_value
                
                # Determine primary and secondary child based on which side the query point falls
                primary, secondary = jnp.where(
                    dist_to_plane <= 0,
                    node_children[node_idx],
                    node_children[node_idx][::-1]
                )
                
                # Always search primary child
                neighbors = _search_node(primary, neighbors)
                
                # Search secondary child only if necessary
                needs_secondary = jnp.abs(dist_to_plane) <= radius
                return jnp.where(
                    needs_secondary,
                    _search_node(secondary, neighbors),
                    neighbors
                )
            
            return jnp.where(is_leaf, _process_leaf(), _process_internal())
        
        # Initialize array to store neighbor indices (with a reasonable max size)
        max_neighbors = jnp.minimum(self.points.shape[0], 100)  # Adjust as needed
        neighbors = jnp.full(max_neighbors, -1, dtype=jnp.int32)
        
        # Search starting from root
        points = self.points
        indices = self.indices
        node_indices = self.node_indices
        node_split_dims = self.node_split_dims
        node_split_values = self.node_split_values
        node_children = self.node_children
        
        neighbors = _search_node(0, neighbors)
        
        # Filter out -1 values (unused slots)
        return neighbors[neighbors >= 0]
    
    def tree_flatten(self):
        return (
            self.points, 
            self.indices, 
            self.node_indices, 
            self.node_split_dims, 
            self.node_split_values, 
            self.node_children
        ), self.leaf_size

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        leaf_size = aux_data
        return cls(*children, leaf_size=leaf_size)


@jax.jit
def visible_area(vertices1: Float[Array, "n_vertices 3"], vertices2: Float[Array, "n_vertices 3"]) -> Float[Array, "n_vertices"]:
    """Calculate visible area between two sets of vertices.
    
    Args:
        vertices1: First set of vertices (triangle)
        vertices2: Second set of vertices (triangle)
        
    Returns:
        Area of visible region
    """
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


@partial(jax.jit, static_argnums=(2,))
def create_kdtrees(m1: MeshModel, m2: MeshModel, leaf_size: int = 10):
    """Create KD-trees for both mesh models for efficient spatial queries.
    
    Args:
        m1: First mesh model
        m2: Second mesh model
        leaf_size: Maximum number of points in a leaf node
        
    Returns:
        Tuple of KD-trees for m1 and m2
    """
    # Create KD-tree for m1 (only include faces with positive mu)
    m1_visible_mask = m1.mus > 0
    m1_visible_indices = jnp.where(m1_visible_mask, jnp.arange(m1.mus.shape[0]), -1)
    m1_visible_indices = m1_visible_indices[m1_visible_indices >= 0]
    
    # Create KD-tree for m2 (only include faces with positive mu)
    m2_visible_mask = m2.mus > 0
    m2_visible_indices = jnp.where(m2_visible_mask, jnp.arange(m2.mus.shape[0]), -1)
    m2_visible_indices = m2_visible_indices[m2_visible_indices >= 0]
    
    # Build KD-trees
    if m1_visible_indices.shape[0] > 0:
        kdtree_m1 = KDTree.construct(m1.cast_centers[m1_visible_indices], m1_visible_indices, leaf_size)
    else:
        # Create empty KD-tree
        kdtree_m1 = KDTree(
            points=jnp.zeros((0, 2)),
            indices=jnp.zeros(0, dtype=jnp.int32),
            node_indices=jnp.zeros((1, 2), dtype=jnp.int32),
            node_split_dims=jnp.zeros(1, dtype=jnp.int32),
            node_split_values=jnp.zeros(1),
            node_children=jnp.full((1, 2), -1, dtype=jnp.int32),
            leaf_size=leaf_size
        )
    
    if m2_visible_indices.shape[0] > 0:
        kdtree_m2 = KDTree.construct(m2.cast_centers[m2_visible_indices], m2_visible_indices, leaf_size)
    else:
        # Create empty KD-tree
        kdtree_m2 = KDTree(
            points=jnp.zeros((0, 2)),
            indices=jnp.zeros(0, dtype=jnp.int32),
            node_indices=jnp.zeros((1, 2), dtype=jnp.int32),
            node_split_dims=jnp.zeros(1, dtype=jnp.int32),
            node_split_values=jnp.zeros(1),
            node_children=jnp.full((1, 2), -1, dtype=jnp.int32),
            leaf_size=leaf_size
        )
    
    return kdtree_m1, kdtree_m2


@jax.jit
def get_potential_occluders(m1: MeshModel, m2: MeshModel, search_radius_factor: float = 2.0):
    """Get potential occluders for each face in m1 from m2.
    
    Args:
        m1: Mesh model that might be occluded
        m2: Potentially occluding mesh model
        search_radius_factor: Factor to multiply triangle size for neighbor search
        
    Returns:
        Array of potential occluder indices for each face in m1
    """
    # Calculate approximate radius for each triangle in m1
    m1_face_vertices = m1.cast_vertices[m1.faces.astype(int)]
    m1_radii = jnp.sqrt(m1.cast_areas)
    
    # Get m2 faces that are visible (positive mu)
    m2_visible_mask = m2.mus > 0
    m2_visible_indices = jnp.where(m2_visible_mask, jnp.arange(m2.mus.shape[0]), -1)
    m2_visible_indices = m2_visible_indices[m2_visible_indices >= 0]
    
    # Build KD-tree for m2 visible faces
    kdtree_m2 = KDTree.construct(m2.cast_centers[m2_visible_indices], m2_visible_indices)
    
    # For each face in m1, find potential occluders in m2
    def find_occluders_for_face(i, potential_occluders):
        # Only process faces with positive mu
        def process_visible_face():
            # Get face center and radius
            center = m1.cast_centers[i]
            radius = m1_radii[i] * search_radius_factor
            
            # Query KD-tree for neighbors within radius
            neighbors = kdtree_m2.query_radius(center, radius)
            
            # Add neighbors to potential occluders
            n_neighbors = jnp.minimum(neighbors.shape[0], potential_occluders.shape[1])
            return potential_occluders.at[i, :n_neighbors].set(neighbors[:n_neighbors])
        
        # Skip faces with negative mu
        return jnp.where(m1.mus[i] > 0, process_visible_face(), potential_occluders)
    
    # Initialize array to store potential occluders (-1 indicates no occluder)
    max_occluders = 50  # Maximum number of potential occluders per face
    potential_occluders = jnp.full((m1.faces.shape[0], max_occluders), -1, dtype=jnp.int32)
    
    # Find potential occluders for each face in m1
    potential_occluders = jax.lax.fori_loop(
        0, m1.faces.shape[0],
        find_occluders_for_face,
        potential_occluders
    )
    
    return potential_occluders


@jax.jit
def resolve_occlusion_for_face(m1: MeshModel, m2: MeshModel, face_index: int, search_radius_factor: float = 2.0):
    """Calculate occlusion for a specific face in m1 by m2.
    
    Args:
        m1: Mesh model containing the face
        m2: Potentially occluding mesh model
        face_index: Index of the face to check for occlusion
        search_radius_factor: Factor to multiply face size for neighbor search
        
    Returns:
        Total occluded area for the face
    """
    # Skip calculation if face is not visible
    is_visible = m1.mus[face_index] > 0
    
    # Return 0 directly for invisible faces
    def process_invisible():
        return 0.0
    
    # Calculate occlusion for visible faces
    def process_visible():
        # Get face center and approximate radius
        center = m1.cast_centers[face_index]
        radius = jnp.sqrt(m1.cast_areas[face_index]) * search_radius_factor
        
        # Create a simple array of indices for potential occluders
        # Instead of filtering with boolean mask, we'll process all indices but ignore results for invisible faces
        potential_indices = jnp.arange(m2.faces.shape[0])
        
        # Calculate distances to center
        distances = jnp.sqrt(jnp.sum((m2.cast_centers - center[None, :]) ** 2, axis=1))
        
        # Create mask for nearby faces with positive mu
        nearby_mask = (distances <= radius) & (m2.mus > 0)
        
        # Convert to weights (1.0 for valid faces, 0.0 for invalid)
        weights = nearby_mask.astype(jnp.float32)
        
        # Calculate occlusion for all faces (will be weighted to zero for invalid faces)
        def calculate_occlusion(idx):
            # Get occluder vertices
            occluder_vertices = m2.cast_vertices[m2.faces[idx].astype(int)]
            
            # Calculate visible area
            area = visible_area(
                m1.cast_vertices[m1.faces[face_index].astype(int)],
                occluder_vertices
            )
            
            # Apply weight
            return area * weights[idx]
        
        # Calculate all occlusions
        occlusions = jax.vmap(calculate_occlusion)(potential_indices)
        
        # Sum and clip
        return jnp.clip(jnp.sum(occlusions), 0.0, m1.cast_areas[face_index])
    
    # Use lax.cond to handle the conditional logic
    return jax.lax.cond(
        is_visible,
        process_visible,
        process_invisible
    )


@jax.jit
def resolve_occlusion(m1: MeshModel, m2: MeshModel, search_radius_factor: float = 2.0) -> MeshModel:
    """Calculate the occlusion of m1 by m2 using KD-trees.
    
    Args:
        m1: Potentially occluded mesh model
        m2: Potentially occluding mesh model
        search_radius_factor: Factor to multiply face size for neighbor search
        
    Returns:
        MeshModel: m1 with updated occluded_areas
    """
    # Check if m1 is closer to the observer than m2
    # We need to compare the los_z values of both meshes
    # Only calculate occlusions if m1 is closer to the observer (smaller los_z)
    
    def calculate_occlusions():
        # Calculate occlusions for all faces in m1
        face_indices = jnp.arange(len(m1.faces))
        occlusions = jax.vmap(
            lambda idx: resolve_occlusion_for_face(m1, m2, idx, search_radius_factor)
        )(face_indices)
        return occlusions
    
    def no_occlusions():
        # If m1 is behind m2, no occlusion occurs
        return jnp.zeros_like(m1.cast_areas)
    
    # Compare los_z values - smaller values are closer to observer
    # Use mean los_z for comparison as a simple heuristic
    m1_los_z = jnp.mean(m1.los_z)
    m2_los_z = jnp.mean(m2.los_z)
    
    # Only calculate occlusions if m1 is closer to the observer than m2
    occlusions = jax.lax.cond(
        m1_los_z < m2_los_z,
        calculate_occlusions,
        no_occlusions
    )
    
    # Update occluded_areas
    return m1._replace(occluded_areas=occlusions)


def get_optimal_search_radius(m1: MeshModel, m2: MeshModel, min_factor: float = 1.5, max_factor: float = 5.0):
    """Determine optimal search radius factor for occlusion detection.
    
    Args:
        m1: First mesh model
        m2: Second mesh model
        min_factor: Minimum search radius factor to consider
        max_factor: Maximum search radius factor to consider
        
    Returns:
        Optimal search radius factor
    """
    # Get visible areas of both meshes
    m1_visible_areas = m1.cast_areas[m1.mus > 0]
    m2_visible_areas = m2.cast_areas[m2.mus > 0]
    
    # Default to 2.0 if no visible areas
    if m1_visible_areas.shape[0] == 0 or m2_visible_areas.shape[0] == 0:
        return 2.0
    
    # Get median triangle sizes
    m1_median_size = jnp.sqrt(jnp.median(m1_visible_areas))
    m2_median_size = jnp.sqrt(jnp.median(m2_visible_areas))
    
    # Calculate optimal radius factor based on mesh densities
    # Use larger factor for smaller triangles to ensure proper occlusion detection
    base_factor = 2.0
    density_factor = base_factor * (1.0 + jnp.exp(-jnp.mean(jnp.array([m1_median_size, m2_median_size])) * 5))
    
    # Clip to reasonable range
    return jnp.clip(density_factor, min_factor, max_factor)