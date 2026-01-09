import math
import jax
import jax.numpy as jnp
import time
from typing import Tuple

from beh.core.registry import *
from jax import config

# Originally the implentation was using jax with 64bit for higher precision in the BVH.
# We swapped this back to 32, because we benchmark all models in the same 32 bit style.
node_dtype = jnp.int32

def build_bvh(
  model_key : str,
  positions : jax.Array,
  max_depth : int,
  reg : CoreRegistry) -> Tuple[jax.Array, CoreRegistry] :
  """
  Build a Bounding Volume Hierarchy for 2D or 3D points.
  
  Constructs a hierarchical spatial data structure that partitions points
  using Morton-order curve discretization for efficient range queries.
  
  Args:
    positions: A JAX array of shape (N, D) containing N points in D-dimensional 
      space (D=2 or 3). Values should be normalized to the range [-1, 1].
    max_depth: The maximum depth level for tree partitioning.
    reg: A CoreRegistry object for managing resources.
  
  Returns:
    tuple:
      - jax.Array: The bounding volume tree with lower and upper bounds.
      - CoreRegistry: The updated registry object.
  """
  construction_time_t0 = time.perf_counter_ns()

  def discretize(positions, bit_depth):
    grid_size = 1 << bit_depth
    integer_positions = jnp.array((positions / 2 + 0.5) * grid_size, dtype = node_dtype)
    return integer_positions

  def morton_code(integer_positions, bit_depth, dimension):
    code = 0
    for i in range(bit_depth):
      for j in range(0, dimension):
        bit = (integer_positions[:, j] >> i) & 1
        code |= bit << (dimension * i + j)
    return code

  def partition(positions, max_depth, dimension):
    tree = jnp.empty((0, 2, dimension))
    for i in range(0, max_depth):
      samples_per_leaf = positions.shape[0] >> i
      blocked_positions = jnp.reshape(positions, (-1, samples_per_leaf, dimension))
      lower = jnp.min(blocked_positions, axis = 1)
      upper = jnp.max(blocked_positions, axis = 1)
      nodes = jnp.append(lower[:,jnp.newaxis,:], upper[:,jnp.newaxis,:], axis = 1)
      tree = jnp.append(tree, nodes, axis = 0)
    return tree

  dimension = positions.shape[-1]

  max_bit_detph = node_dtype(0).nbytes * 8
  bit_depth = max_bit_detph // dimension

  integer_positions = discretize(positions, bit_depth)
  codes = morton_code(integer_positions, bit_depth, dimension)
  sorted_positions = positions[jnp.argsort(codes)]
  tree = partition(sorted_positions, max_depth, dimension)

  tree = jnp.array(tree)

  print(f"BVH built with {tree.shape[0]} nodes and depth {int(math.log2(tree.shape[0] + 1))}")

  # Save BVH specs to Core Registry
  reg_key = model_key + core_keys['training_time']
  reg.add( reg_key, np.array((time.perf_counter_ns() - construction_time_t0) / 6e10)) # Convert to ns to min

  reg.add( model_key + core_keys['total_parameters_key'], tree.size)
  reg.add( model_key + core_keys['active_parameters_key'], tree.size)

  return tree, reg

def child(node_indices , sibling, level):
  # Breadth-first layout
  # 0112222
  old_base = (1 << (level + 0)) - 1
  new_base = (1 << (level + 1)) - 1
  node_indices -= old_base
  new_node_indices = node_indices * 2 + sibling
  new_node_indices += new_base
  return new_node_indices

def append_in_place(a, b, count):
  all_indices = jnp.arange(0, a.shape[0])

  is_second = (all_indices >= count) & (all_indices < 2 * count)
  return jnp.where(is_second, jnp.roll(b, count), a)

@jax.jit
def query_bvh(tree : jax.Array, queries : jax.Array) -> jax.Array :
  """
  Query the Bounding Volume Hierarchy.
  
  Args:
    tree: The BVH structure with shape (nodes, 2, dimension).
    queries: Points to query with shape (num_queries, dimension).
  
  Returns:
    A binary array indicating which queries hit the BVH leaf nodes.
  """
  depth = int(math.log2(tree.shape[0] + 1))

  active_count = queries.shape[0]
  max_size = 32 * active_count

  query_indices = jnp.where(jnp.arange(0, max_size) < active_count, jnp.arange(max_size, dtype = node_dtype), -1)
  node_indices = jnp.zeros(max_size, dtype = node_dtype)

  for level in range(0, depth):
    #assert(active_count < max_size)

    active_querys = queries[query_indices]
    active_nodes = tree[node_indices]
    is_above_min = jnp.all(active_querys >= active_nodes[:, 0], axis = -1)
    is_below_max = jnp.all(active_querys <= active_nodes[:, 1], axis = -1)
    is_inside = is_above_min & is_below_max
    keep = (query_indices > - 1) & is_inside

    active_count = jnp.sum(keep)

    def sort_and_clip(keep, x):
      x = jax.lax.sort_key_val(keep, x)[1]
      x = jnp.flip(x)
      x = jnp.where(jnp.arange(0, max_size) < active_count, x, -1)
      return x

    query_indices = sort_and_clip(keep, query_indices)
    node_indices = sort_and_clip(keep, node_indices)

    if level == depth - 1:
      result = jnp.zeros(queries.shape[0])
      result = result.at[query_indices].set(1)
      return result
    else:
      node_indices_left = child(node_indices, 0, level)
      node_indices_right = child(node_indices, 1, level)

      query_indices = append_in_place(query_indices, query_indices, active_count)
      node_indices = append_in_place(node_indices_left, node_indices_right, active_count)

#------------------------------------------------------------------------------------

def batch_query_bvh(bvh : jax.Array, x_batches : list,):
    ## Trim tail of x that does not fit with batchsize
    x_batched = jnp.stack(x_batches[0:-1])
    yp = jax.vmap(lambda X: query_bvh(bvh, X))(x_batched).flatten()
    ## Add tail
    x_tail = query_bvh(bvh, x_batches[-1])
    yp = jnp.concatenate((yp, x_tail.flatten()))
    return yp


def grid(shape):
  u = jnp.linspace(-1, 1, shape[1])
  v = jnp.linspace(-1, 1, shape[0])
  x, y = jnp.meshgrid(v, u)
  return jnp.stack((x, y)).T

def shrink_to_nex_power_of_two(positions : jax.Array) -> jax.Array :
    power_of_two_count = 1 << jnp.array(jnp.floor(jnp.log2(positions.shape[0])), int)
    positions = positions[0 : power_of_two_count]
    return positions

def pad_to_power_of_two(positions : jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Pad positions array to the next power of two to preserve all data.
    
    Args:
        positions: A JAX array of shape (N, D) containing N points in D-dimensional space.
    
    Returns:
        tuple:
            - Padded positions array of shape (P, D) where P is the next power of 2 >= N
            - Boolean mask of shape (P,) indicating which rows are original (True) vs padded (False)
    """
    n = positions.shape[0]
    # Calculate next power of 2
    power_of_two_size = 1 << int(jnp.ceil(jnp.log2(n)))
    
    # Pad with edge values (repeat last row)
    padding_needed = power_of_two_size - n
    positions_padded = jnp.pad(positions, ((0, padding_needed), (0, 0)), mode='edge')
    
    # Create mask for original data
    mask = jnp.arange(power_of_two_size) < n
    
    return positions_padded, mask