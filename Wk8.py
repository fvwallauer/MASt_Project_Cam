import numpy as np
import matplotlib.pyplot as plt
import random
import jax
print("JAX devices:", jax.devices())
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Computer Modern']
plt.rcParams.update({'font.size': 14})
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

# ===============================================
# PLATE PARAMETERS
# -----------------------------------------------
n_retractors = 168
r = 102.5 # mm, half of 205mm field
n_tiers = 3
n_fibres = n_retractors * n_tiers

# Retractor coordinates & constraints
angles = jnp.linspace(0, 2*jnp.pi, n_retractors, endpoint=False) # placement of fibres
retractor_x = jnp.asarray(r * jnp.cos(angles))
retractor_y = jnp.asarray(r * jnp.sin(angles))
fibre_span_allowed = jnp.asarray([r*0.4, r*0.7 , r*1.3]) # for tiers 0,1,2 (low-high)
phi_max = jnp.deg2rad(14)  # max bend angle

# Button vertices in local fibre-target coordinates
# box of 9mm lengh, 2mm width
button_vertices_local = jnp.array([
    [-1,  6],
    [ 1,  6],
    [ 1, -3],
    [-1, -3],
])

# Minimum possible target separation
min_sep = 2 # mm; min. distance of separation below which buttons overlap

# -----------------------------------------------
# FIELD PARAMETERS
# -----------------------------------------------
n_targets = 1400 # 1400

# ===============================================


# Sample random target dist., uniform over circular sky area
def rand_targets(R, n):
    """
    R = radius of disc
    n = number of targets to generate
    """
    rng = np.random.default_rng(0)
    p = rng.random(n)
    q = rng.random(n)
    rr = R * np.sqrt(p)
    theta = 2*np.pi*q
    x = rr * np.cos(theta)
    y = rr * np.sin(theta)
    return jnp.asarray(x), jnp.asarray(y)

# Indexing: assigning fibre IDs based on retractor and tier IDs
def fibre_id(retractor_id, tier_id): # Fibre ID from retractor & tier id
    if tier_id < 0 or tier_id >= n_tiers:
        raise ValueError("Invalid tier_id")
    return retractor_id * n_tiers + tier_id

# Indexing: retrieving retractor and tier IDs from fibre ID
def fibre_rt(fibre_id):
    r_id = fibre_id // n_tiers
    t_id = fibre_id % n_tiers
    return r_id, t_id # Retractor ID & tier ID from fibre ID

@jit
def TF_reach_matrix(px,py):
    """
    Vectorized function to generate matrix for allowed target-fibre pairs given reach & bend constraints.
    """
    fibre_ids = jnp.arange(n_fibres)      # (F,)
    r_ids = fibre_ids // n_tiers          # (F,)
    t_ids = fibre_ids % n_tiers           # (F,)

    rx = retractor_x[r_ids]               # (F,)
    ry = retractor_y[r_ids]               # (F,)
    tier_span = fibre_span_allowed[t_ids] # (F,) / FOR EACH FIBRE

    pos_r = jnp.stack([rx, ry], axis=1)  # (F,2) (x,y) pos. for each retractor
    pos_r_norm_inward = - pos_r / jnp.linalg.norm(pos_r, axis=1, keepdims=True) # (F,) : unit vector in retractor-origin direction

    vx = px[None, :] - rx[:, None]     # (F, T) Broadcast dimensions!! So cool!
    vy = py[None, :] - ry[:, None]     # (F, T)
    v_len = jnp.sqrt(vx**2 + vy**2) + 1e-12   # (F, T)

    ux = vx / v_len   # (F, T)
    uy = vy / v_len   # (F, T)

    # Dot product between norm (inward) from retractor & unit vector from retractor to target
    dot = (ux * pos_r_norm_inward[:, 0:1] + uy * pos_r_norm_inward[:, 1:2])     # (F, T)

    # Angle between norm @ rectractor & vector to target
    dot = jnp.clip(dot, -1.0, 1.0)  # numerical safety
    ang = jnp.arccos(dot)       # (F, T)

    # Allowed span per fibre
    span_allowed = tier_span[:, None]  # (F, 1)

    # Check reachability
    allowed = (v_len <= span_allowed) & (ang <= phi_max)  # (F, T) bool

    return allowed

# Get targets for a single fibre form *allowed*
# allowed_all[fibre] # (T,) bool

@jit
def all_button_vertices(px, py, allowed):
    """
    Compute button polygon vertices for all fibre-target combinations.

    px, py: (T,)
    allowed: (F, T) bool - True if fibre f can reach target t
    returns: vertices of shape (F, T, 4, 2)
             vertices[f, t, k, :] = [x, y] of vertex k (k=0..3)
             entries are NaN where allowed[f, t] is False.
    """
    F = allowed.shape[0]             # n_fibres
    T = allowed.shape[1]             # n_targets

    # Get fiber IDs from Retractor IDs
    fibre_ids = jnp.arange(F)              # (F,)
    r_ids = fibre_ids // n_tiers          # (F,)

    # Broadcast fibre and target positions to (F, T)
    rx = retractor_x[r_ids, None]               # (F,1)
    ry = retractor_y[r_ids, None]               # (F,1)
    # rx = retractor_x[:, None]         # (F, 1)
    # ry = retractor_y[:, None]         # (F, 1)

    tx = px[None, :]                  # (1, T)
    ty = py[None, :]                  # (1, T)

    # Direction from retractor to target
    vx = tx - rx                      # (F, T)
    vy = ty - ry                      # (F, T)
    phi = jnp.arctan2(vy, vx)         # (F, T)
    theta = phi + jnp.pi / 2          # (F, T) FIRST ATTEMPT WAS PERP.?

    c = jnp.cos(theta)                # (F, T)
    s = jnp.sin(theta)                # (F, T)

    # Local vertices: (4,2) -> separate x,y components
    bx = button_vertices_local[:, 0]  # x-components (4,)
    by = button_vertices_local[:, 1]  # y-components (4,)

    # Broadcast local vertices to (F, T, 4)
    bx_broadcast = bx[None, None, :]          # (1, 1, 4)
    by_broadcast = by[None, None, :]          # (1, 1, 4)

    # Rotate each vertex for each (fibre, target)
    # ROTATION from (x,y) to (x',y') such that : x' = c*x - s*y & y' = s*x + c*y
    # Broadcast cos & sin from (F,T,1) to get final dimensions of (F,T,4) - rotated vertices for each Fibre-Target combination
    x_rot = c[:, :, None] * bx_broadcast - s[:, :, None] * by_broadcast   # (F, T, 4)
    y_rot = s[:, :, None] * bx_broadcast + c[:, :, None] * by_broadcast  # (F, T, 4)

    # Translate vertices to target position (tx, ty) !
    x_vertex = px[None, :, None] + x_rot                  # (F, T, 4)
    y_vertex = py[None, :, None] + y_rot                  # (F, T, 4)

    # Apply to reachability matrix: only return vertices for possible T-F combinations
    mask = allowed[:, :, None]                            # (F, T, 1)
    # print(rx.shape)
    # print(bx_broadcast.shape)
    # print(c.shape)
    # print(x_rot.shape)
    # print(x_vertex.shape)
    # print(mask.shape)
    x_vertex = jnp.where(mask, x_vertex, jnp.nan) # jnp.where: vlues drawn from x_vertex if mask (reach!) is True, NaN if not
    y_vertex = jnp.where(mask, y_vertex, jnp.nan)

    # Return final array of vertices
    vertices = jnp.stack([x_vertex, y_vertex], axis=-1)   # x_vertex, y_vertex have shape (F, T, 4), stack axis=-1 creates new rightmost axis! 
                                                          # Final shape: (F, T, 4, 2)
    return vertices

@jax.jit
def compute_target_bboxes(vertices):
    # Transform vertices to per-target
    vertices_T = jnp.transpose(vertices, (1, 0, 2, 3))      # (T, F, 4, 2)
    # vertices_T = vertices_T.reshape(vertices_T.shape[0], -1, 2)   # (T, F*4, 2) : ALL button verties for each target

    # xmin = jnp.nanmin(vertices_T[..., 0], axis=1) # min of all x-coords
    # xmax = jnp.nanmax(vertices_T[..., 0], axis=1) # max  "  "     "
    # ymin = jnp.nanmin(vertices_T[..., 1], axis=1) # min of all y-coords
    # ymax = jnp.nanmax(vertices_T[..., 1], axis=1) # max  "  "     "

    # vertices: (F,T,4,2)
    xmin = jnp.nanmin(vertices[...,0], axis=(0,2)) # over fibres and 4 vertices
    xmax = jnp.nanmax(vertices[...,0], axis=(0,2))
    ymin = jnp.nanmin(vertices[...,1], axis=(0,2))
    ymax = jnp.nanmax(vertices[...,1], axis=(0,2))

    footprint_valid = jnp.isfinite(xmin) # False target has no reachable fibres at all
    return xmin, ymin, xmax, ymax, footprint_valid

@jax.jit
def compute_bbox_overlap(xmin, ymin, xmax, ymax, footprint_valid):
    # Broadcast bounds to compare pair-wise bounding box coordinates
    xmin_i = xmin[:, None];  xmin_j = xmin[None, :] # (T, 1) & (1, T) respectively
    xmax_i = xmax[:, None];  xmax_j = xmax[None, :]
    ymin_i = ymin[:, None];  ymin_j = ymin[None, :]
    ymax_i = ymax[:, None];  ymax_j = ymax[None, :]

    # Boxes overlap IF the min of one vertex is less than the max of the other in both X and Y
    overlap = (
        (xmax_i >= xmin_j) &
        (xmax_j >= xmin_i) &
        (ymax_i >= ymin_j) &
        (ymax_j >= ymin_i)
    ) # Boolean array shape (T,T)

    # Consider which targets have NO valid F-T connections
    valid_i = footprint_valid[:, None] # (T, 1)
    valid_j = footprint_valid[None, :] # (1, T)

    overlap = overlap & valid_i & valid_j # combine bool arrays - only where all are True is True returned

    # Remove diagonal entries (self-overlap)
    T = xmin.shape[0]
    overlap = overlap & (~jnp.eye(T, dtype=bool)) # Set all (i,i) entries to False

    # Symmetrize - keep only upper triangle as in Terret (2014)
    overlap = jnp.triu(overlap, k=1) # keep upper triangle
    overlap = overlap | overlap.T # mirror to lower

    return overlap

@jax.jit
def always_collide(px, py, reachable_Ts, min_sep):
    # Get target-target separation via broadcasting
    xi = px[:, None]      # (T, 1)
    yi = py[:, None]      # (T, 1)
    dx = xi - xi.T        # (T, T)
    dy = yi - yi.T        # (T, T)
    d2 = dx**2 + dy**2    # (T, T) squared distances between targets

    # Ignore self-distances
    T = px.shape[0] # n_targets
    d2 = d2 + jnp.eye(T) * jnp.inf # add identity matrix scaled by inf to distances**2

    # Check for separation-based collision
    collide = d2 < (min_sep * min_sep) # (T, T) bool

    # Limit to target pairs that are both reachable by some fibre
    both_reachable = reachable_Ts[:, None] & reachable_Ts[None, :] # BROADCAST (T, T) bool
    always_collide = collide & both_reachable # reachable pairs that always collide

    # Keep upper triangle
    always_collide = jnp.triu(always_collide, k=1) # return only upper triangle
    always_collide = always_collide | always_collide.T # force symmetry
    always_collide = jnp.where(jnp.eye(T, dtype=bool), False, always_collide) # force diagonal = False

    return always_collide

# Function to check segment-segment overlap
def seg_seg_intersect(p, r, q, s):
    # Input: p,r,q,s are (...,2) arrays representing segments p->p+r and q->q+s
    # Output: boolean array of shape (...) indicating whether segments intersect

    cross = lambda a, b: a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0] # 2D cross product

    rxs = cross(r, s)
    q_p = q - p
    q_pxr = cross(q_p, r) # is separation of q & p || to r (or s)? If yes: vectors are not just parallel but collinear

    # parallel or collinear
    parallel = jnp.isclose(rxs, 0)
    rxs_safe = jnp.where(parallel, 1.0, rxs)
    # collinear = jnp.isclose(q_pxr, 0) # Is this necessary??

    # For non-parallel, find t,u parameters at intersection. If both ∈ [0,1] POI ∈ of both segments
    t = cross(q_p, s) / rxs_safe
    u = cross(q_p, r) / rxs_safe

    # Intersect: True where NOT parallel, and t,u both ∈ [0,1]
    intersect = (~parallel) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    return intersect


@jax. jit
def compute_path_bboxes(px, py, retractor_x, retractor_y, r_id_for_fibre, allowed):
    """
    Compute bounding boxes for all fibre segments reaching each target.
    """
    F, T = allowed.shape # n_fibres, n_targets

    # Retractor positions for each fibre
    rx_f = retractor_x[r_id_for_fibre] # (F,)
    ry_f = retractor_y[r_id_for_fibre] # (F,)

    rx = rx_f[:, None] # (F, 1)
    ry = ry_f[:, None] # (F, 1)

    # All target positions for all fibres (tx[f,t] = px[t])
    tx = jnp.broadcast_to(px[None, :], (F, T))
    ty = jnp.broadcast_to(py[None, :], (F, T))

    # Segment bounding boxes for each (F,T) - either bounded by target or 
    seg_xmin = jnp.minimum(tx, rx)
    seg_xmax = jnp.maximum(tx, rx)
    seg_ymin = jnp.minimum(ty, ry)
    seg_ymax = jnp.maximum(ty, ry)

    # Mask with allowed fibres; unreachable -> ignore
    big = 1e30  # large number
    seg_xmin = jnp.where(allowed, seg_xmin, big)
    seg_ymin = jnp.where(allowed, seg_ymin, big)
    seg_xmax = jnp.where(allowed, seg_xmax, -big)
    seg_ymax = jnp.where(allowed, seg_ymax, -big)

    # Take min/max over fibres for each target
    path_xmin = jnp.min(seg_xmin, axis=0)  # (T,)
    path_ymin = jnp.min(seg_ymin, axis=0)
    path_xmax = jnp.max(seg_xmax, axis=0)
    path_ymax = jnp.max(seg_ymax, axis=0)

    return path_xmin, path_ymin, path_xmax, path_ymax

@jax.jit
def compute_fibre_segments(px, py, retractor_x, retractor_y, r_id_for_fibre, allowed):
    """
    Compute fibre segments for all fibre-target combinations.
    Return: 
    start: (F, T, 2) array of segment start points (target pos)
    vector:   (F, T, 2) vector of fibre path (from retractor to target)
    """
    F, T = allowed.shape

    # target positions: (T, 2)
    target_pos = jnp.stack([px, py], axis=-1)          # (T, 2)
    # broadcast to (F, T, 2)
    start = jnp.broadcast_to(target_pos[None, :, :], (F, T, 2)) # Target positions for all fibres # (F, T, 2)

    # retractor positions per fibre: (F,)
    rx = retractor_x[r_id_for_fibre]
    ry = retractor_y[r_id_for_fibre]

    end = jnp.stack([rx[:, None], ry[:, None]], axis=-1)  # (F, 1, 2)
    # For each fibre, end[fibre, 0, :] = (retractor_x[f], retractor_y[f])

    # Direction from 
    vector = end - start  # (F, T, 2)

    # Mask unreachable fibre–target pairs with NaNs so they never intersect
    mask = allowed[..., None]  # (F, T, 1)
    start = jnp.where(mask, start, jnp.nan)
    vector = jnp.where(mask, vector, jnp.nan)

    return start, vector

@jax.jit
def fibre_collision_block_padded(target_vertices, target_fibre_start, target_fibre_dir, target_fibre_mask, t1, t2):
    """
    Check for fibre-fibre collisions between two target blocks t1 and t2.
    target_vertices: (T, F, 4, 2) button vertices for each target-fibre pair
    target_fibre_start: (T, F, 2) fibre start points for each target-fibre pair
    target_fibre_dir: (T, F, 2) fibre direction vectors for each target-fibre pair
    target_fibre_mask: (T, F) bool mask indicating valid fibre-target pairs

    t1, t2: target indices to check

    Returns:
    collision_matrix: (F1, F2) bool array indicating collisions between fibres of target t1 and t2
    """
    verts_i = target_vertices[t1]  # (max_F, 4, 2)
    verts_j = target_vertices[t2]  # (max_F, 4, 2)

    edges_i = jnp.roll(verts_i, -1, axis=1) - verts_i  # (max_F, 4, 2)
    edges_j = jnp.roll(verts_j, -1, axis=1) - verts_j  # (max_F, 4, 2)

    mask_i = target_fibre_mask[t1]  # (max_F,)
    mask_j = target_fibre_mask[t2]  # (max_F,)

    # Ceck for button-button intersections
    p_bb = verts_i[:, None, :, None, :]  # (max_F, 1, 4, 1, 2)
    r_bb = edges_i[:, None, :, None, :]
    q_bb = verts_j[None, :, None, :, :]  # (1, max_F, 1, 4, 2)
    s_bb = edges_j[None, :, None, :, :]

    intersects_bb = seg_seg_intersect(p_bb, r_bb, q_bb, s_bb)  # (max_F, max_F, 4, 4)
    button_intersect = jnp.any(intersects_bb, axis=(2, 3))             # (max_F, max_F)

    # Fibre from t1 vs button from t2
    # fetch fibre data for target t1
    start_i = target_fibre_start[t1]  # (max_F, 2)
    dir_i   = target_fibre_dir[t1]    # (max_F, 2)

    # Fetch fibre segments from data for target t1 as p -> p+r
    p_fb1 = start_i[:, None, None, :]  # (max_F, 1, 1, 2)
    r_fb1 = dir_i[:, None, None, :]
    # Fetch button edges for target t2, as q -> q+s
    q_fb1 = verts_j[None, :, :, :]     # (1, max_F, 4, 2)
    s_fb1 = edges_j[None, :, :, :]

    intersects_fb1 = seg_seg_intersect(p_fb1, r_fb1, q_fb1, s_fb1)  # (max_F, max_F, 4)
    f1_on_b2 = jnp.any(intersects_fb1, axis=2)          # (max_F, max_F)

    # fibre from t2 vs button from t1
    # Fetch fibre data for target t2
    start_j = target_fibre_start[t2]  # (max_F, 2)
    dir_j   = target_fibre_dir[t2]    # (max_F, 2)

    p_fb2 = start_j[None, :, None, :]  # (1, max_F, 1, 2)
    r_fb2 = dir_j[None, :, None, :]
    q_fb2 = verts_i[:, None, :, :]     # (max_F, 1, 4, 2)
    s_fb2 = edges_i[:, None, :, :]

    intersects_fb2 = seg_seg_intersect(p_fb2, r_fb2, q_fb2, s_fb2)  # (max_F, max_F, 4)
    f2_on_b1 = jnp.any(intersects_fb2, axis=2)          # (max_F, max_F)

    # --- Combine all collision modes ---------------------------------
    block = button_intersect | f1_on_b2 | f2_on_b1    # (max_F, max_F)

    # Mask out padded fibres
    valid_i = mask_i[:, None]  # (max_F, 1)
    valid_j = mask_j[None, :]  # (1, max_F)
    valid_pair = valid_i & valid_j

    block = block & valid_pair

    return block
    
@jax.jit
def batch_collision_blocks(target_vertices, target_fibre_start, target_fibre_dir, 
    target_fibre_mask,t1_batch, t2_batch):
    def one_pair(t1, t2):
        return fibre_collision_block_padded(
            target_vertices, target_fibre_start, target_fibre_dir,
            target_fibre_mask, t1, t2
        )
    return jax.vmap(one_pair)(t1_batch, t2_batch)  # (B, max_F, max_F)


# ================================================
# COLLISION MATRIX
# ================================================

# Fetch fibre and tier IDs for all fibres
fibre_ids = jnp.arange(n_fibres)              # (F,)
r_id_for_fibre = fibre_ids // n_tiers         # (F,)
t_id_for_fibre = fibre_ids % n_tiers          # (F,)

px,py = rand_targets(r, n_targets)

allowed = TF_reach_matrix(px,py)
reachable_Ts = jnp.any(allowed, axis=0) # collabse fibre dimension, check if ANY fibre can reach a given target

always_collide_mat = always_collide(px, py, reachable_Ts, min_sep)

vertices = all_button_vertices(px, py, allowed)

# NEW 

xmin, ymin, xmax, ymax, footprint_valid = compute_target_bboxes(vertices)

path_xmin, path_ymin, path_xmax, path_ymax = compute_path_bboxes(
    px, py,
    retractor_x, retractor_y,
    r_id_for_fibre,
    allowed,
)

# Broadcast to (T, T)
# FIBRE PATH
pxmin = path_xmin[:, None]
pymin = path_ymin[:, None]
pxmax = path_xmax[:, None]
pymax = path_ymax[:, None]

# BUTTON FOOTPRINT
fxmin = xmin[None, :]
fymin = ymin[None, :]
fxmax = xmax[None, :]
fymax = ymax[None, :]

# Path(A) vs Foot(B)
pathA_hits_footB = (
    (pxmin <= fxmax) &
    (pxmax >= fxmin) &
    (pymin <= fymax) &
    (pymax >= fymin)
)

# Symmetric condition: (path(B) vs foot(A)) == transpose
path_bbox_overlap = pathA_hits_footB | pathA_hits_footB.T

# ignore self-pairs
path_bbox_overlap = path_bbox_overlap & (~jnp.eye(path_bbox_overlap.shape[0], dtype=bool))


bbox_overlap = compute_bbox_overlap(xmin, ymin, xmax, ymax, footprint_valid)
needs_fibre_level = (~always_collide_mat) & (bbox_overlap | path_bbox_overlap) # do NOT always collide, and have overlapping footprints OR path/footprint overlaps


print("Pairs needing fibre-level checks:", int(needs_fibre_level.sum()/2))

edges = jnp.roll(vertices, shift=-1, axis=2) - vertices  # (F, T, 4, 2)

# NEW

# Compute start and direction (p, p+r_vector) for all fibre alignments
fibre_start, fibre_dir = compute_fibre_segments(
    px, py,
    retractor_x, retractor_y,
    r_id_for_fibre,
    allowed
)

# allowed: JAX array (F, T)
allowed_np = np.array(allowed)  # (F, T)

# Python list: for each target t, the array of fibre indices that can reach it
reachable_fibres_per_target = [
    np.where(allowed_np[:, t])[0].astype(np.int32)
    for t in range(allowed_np.shape[1])
]

# Average number of fibres per target
avg_nfib = np.mean([len(fibs) for fibs in reachable_fibres_per_target])
print("Average fibres per target:", avg_nfib)

# max. fibres per target in field
max_F = max(len(fibs) for fibs in reachable_fibres_per_target)
print("Max fibres per target:", max_F)

n_targets = allowed_np.shape[1]

# Padded fibre ID table: (T, max_F), and mask
target_fibre_ids  = -np.ones((n_targets, max_F), dtype=np.int32)
target_fibre_mask = np.zeros((n_targets, max_F), dtype=bool)

for t, fibs in enumerate(reachable_fibres_per_target):
    n = min(len(fibs), max_F)
    target_fibre_ids[t, :n]  = fibs[:n]
    target_fibre_mask[t, :n] = True

# Build per-target padded geometry (always same dimension!!) from existing vertices, fibre_start, fibre_dir
# Fixed length to vectorize in JAX later
vertices_np = np.array(vertices) # (F, T, 4, 2)
fibre_start_np = np.array(fibre_start) # (F, T, 2)
fibre_dir_np = np.array(fibre_dir) # (F, T, 2)

target_vertices = np.zeros((n_targets, max_F, 4, 2), dtype=vertices_np.dtype) # EMPTY ARRAYS
target_fibre_start = np.zeros((n_targets, max_F, 2), dtype=fibre_start_np.dtype)
target_fibre_dir = np.zeros((n_targets, max_F, 2), dtype=fibre_dir_np.dtype)

for t in range(n_targets):
    # Fill in fibre ID data for target T, if any
    ids = target_fibre_ids[t]
    valid = ids >= 0
    if np.any(valid):
        target_vertices[t, valid, :, :] = vertices_np[ids[valid], t, :, :]
        target_fibre_start[t, valid, :] = fibre_start_np[ids[valid], t, :]
        target_fibre_dir[t, valid, :] = fibre_dir_np[ids[valid], t, :]

# PADDED ARRAYS
target_vertices_j = jnp.array(target_vertices)
target_fibre_start_j = jnp.array(target_fibre_start)
target_fibre_dir_j = jnp.array(target_fibre_dir)
target_fibre_mask_j = jnp.array(target_fibre_mask)

# Only keep each unordered pair once (upper triangle) - TERRET (2014) approach
needs_fibre_level_np = np.array(needs_fibre_level)
pairs_i, pairs_j = np.where(np.triu(needs_fibre_level_np, k=1)) # k=1 specifies that we want to keep everything from the upper triangle only (above main diagonal)
pairs = np.stack([pairs_i, pairs_j], axis=1).astype(np.int32) # list of all UNIQUE pairs to send to fibre-level checks

print("Unique pairs needing fibre-level checks:", len(pairs))

collision_matrix = {}

B = 256  # batch size: # targets to process per call

for i in tqdm(range(0, len(pairs), B), desc="Building fibre-level collision blocks (batched)"):
    batch = pairs[i:i+B]
    t1_batch = jnp.array(batch[:, 0]) # target indices for first target in each pair
    t2_batch = jnp.array(batch[:, 1]) # target indices for second target in each pair

    blocks_full = batch_collision_blocks(
        target_vertices_j,
        target_fibre_start_j,
        target_fibre_dir_j,
        target_fibre_mask_j,
        t1_batch,
        t2_batch,
    )  # (B, max_F, max_F) : batched JAX array of full collision blocks

    blocks_full_np = np.array(blocks_full)

    for (t1, t2), block_full in zip(batch, blocks_full_np):
        mask_i = target_fibre_mask[t1]
        mask_j = target_fibre_mask[t2]
        # Compress back down to (Fi, Fj) only for real fibres
        block_compressed = block_full[np.ix_(mask_i, mask_j)]
        collision_matrix[(int(t1), int(t2))] = block_compressed

collision_kind = jnp.where(
    always_collide_mat,
    1,
    jnp.where(needs_fibre_level, 0, -1)
)

print('Always:', len(jnp.where(collision_kind == 1)[0])/2)
print('Never:', len(jnp.where(collision_kind == -1)[0])/2)
print('Sometimes:', len(jnp.where(collision_kind == 0)[0])/2)

# Calculate the energy per fibre allocation
F, T = allowed.shape
rng = np.random.default_rng(42)
priorities = np.ones(T)
high_frac = 0.55 # fraction of high-priority targets
high_idx = rng.choice(np.arange(T), size=int(high_frac*T), replace=False) # get indicese of random high-priority targets, replace=False => select without replacement
priorities[high_idx] = 10.0  # high-priority targets

@jax.jit
def compute_energies_ft(px, py, retractor_x, retractor_y, n_tiers, phi_max, allowed, priorities):
    """
    Returns energy per fibre-target combination (F,T). Unreachable pairs are +inf.
    energy = (1 + s) / priority (Hughes, 2022)
    s = straightness of each fibre, s << 1
    """
    F, T = allowed.shape # n_fibres, n_targets
    fibre_ids = jnp.arange(F)      # (F,)
    r_ids = fibre_ids // n_tiers          # (F,)
    t_ids = fibre_ids % n_tiers           # (F,)

    rx = retractor_x[r_ids]               # (F,)
    ry = retractor_y[r_ids]               # (F,)
    tier_span = fibre_span_allowed[t_ids] # (F,) / FOR EACH FIBRE

    pos_r = jnp.stack([rx, ry], axis=1)  # (F,2) (x,y) pos. for each retractor
    pos_r_norm_inward = - pos_r / jnp.linalg.norm(pos_r, axis=1, keepdims=True) # (F,) : unit vector in retractor-origin direction

    vx = px[None, :] - rx[:, None]     # (F, T) Broadcast dimensions!! So cool!
    vy = py[None, :] - ry[:, None]     # (F, T)
    v_len = jnp.sqrt(vx**2 + vy**2) + 1e-12   # (F, T)

    ux = vx / v_len   # (F, T)
    uy = vy / v_len   # (F, T)

    # Dot product between norm (inward) from retractor & unit vector from retractor to target
    dot = (ux * pos_r_norm_inward[:, 0:1] + uy * pos_r_norm_inward[:, 1:2])     # (F, T)

    # Angle between norm @ rectractor & vector to target
    dot = jnp.clip(dot, -1.0, 1.0)  # numerical safety
    ang = jnp.arccos(dot)       # (F, T)

    # straightness penalty (small)
    s = 0.1 * (ang / phi_max) # (F,T) # GUESS AT THE PENALTY?

    # unreachable => inf. so we never choose it!
    s = jnp.where(allowed, s, jnp.inf)

    energy_ft = (1.0 + s) / priorities[None, :]  # broadcast priorities over fibres
    # minimize energy_ft by having SMALL s, i.e. phi_max >> ang
    return energy_ft

energies_ft_j = compute_energies_ft(px, py, retractor_x, retractor_y, n_tiers, phi_max, allowed, priorities)

# ============================================================
# MISZALSKI/WEAVE-STYLE SA (plugs into your existing variables)
# ============================================================

import numpy as np

# ----------------------------
# Step A: Build pivot->reachable-targets padded lists (from allowed_np)
# ----------------------------
def build_reach_targets_per_pivot(allowed_np: np.ndarray):
    F, T = allowed_np.shape
    reach_lists = [np.where(allowed_np[f])[0].astype(np.int32) for f in range(F)]
    Kp = max((len(x) for x in reach_lists), default=0)

    reach_targets = -np.ones((F, Kp), dtype=np.int32)
    reach_mask = np.zeros((F, Kp), dtype=bool)

    for f, lst in enumerate(reach_lists):
        n = len(lst)
        if n:
            reach_targets[f, :n] = lst
            reach_mask[f, :n] = True

    print("[A] reach_targets shape:", reach_targets.shape, "Kp(max targets per fibre)=", Kp)
    print("[A] mean reachable targets per fibre:", np.mean(reach_mask.sum(axis=1)))
    return reach_targets, reach_mask


# ----------------------------
# Step B: Build a fast global-fibre -> local-index lookup per target
#         to interpret your compressed collision blocks correctly.
# ----------------------------
def build_local_index_map(target_fibre_ids: np.ndarray, n_fibres: int):
    """
    local_index[t, f] = k where f is the k-th fibre in the compressed list for target t,
    or -1 if fibre f is not reachable for target t.
    """
    T, max_F = target_fibre_ids.shape
    local_index = -np.ones((T, n_fibres), dtype=np.int16)

    for t in range(T):
        ids = target_fibre_ids[t]
        valid = ids >= 0
        # compressed ordering is exactly the order in reachable_fibres_per_target (i.e. ids[valid])
        # we set local index k in [0..Fi-1]
        fibs = ids[valid]
        for k, f in enumerate(fibs):
            local_index[t, int(f)] = k

    print("[B] local_index shape:", local_index.shape)
    # quick sanity: for a random target, check first reachable fibre maps to k=0
    t0 = np.random.randint(0, T)
    first = target_fibre_ids[t0][target_fibre_ids[t0] >= 0]
    if len(first):
        f0 = int(first[0])
        assert local_index[t0, f0] == 0
    return local_index


# ----------------------------
# Step C: Collision oracle using YOUR collision_kind + collision_matrix
# ----------------------------
always_collide_np = np.array(always_collide_mat, dtype=bool)
needs_fibre_level_np = np.array(needs_fibre_level, dtype=bool)

def pair_key(t1: int, t2: int):
    return (t1, t2) if t1 < t2 else (t2, t1)

def collides(f1: int, t1: int, f2: int, t2: int, local_index: np.ndarray) -> bool:
    """
    True if (f1->t1) and (f2->t2) conflict.
    Uses:
      - always_collide (button overlap unavoidable)
      - needs_fibre_level -> consult collision_matrix compressed block
      - otherwise -> no collision
    """
    if t1 == t2:
        return True  # two fibres can't share a target

    a, b = pair_key(t1, t2)

    if always_collide_np[a, b]:
        return True

    if not needs_fibre_level_np[a, b]:
        return False

    block = collision_matrix.get((a, b), None)
    if block is None:
        # Should be rare: needs_fibre_level True but no block stored (e.g. build skipped)
        # Be conservative or permissive; I choose conservative False here to avoid killing yield.
        return False

    i1 = int(local_index[a, f1]) if t1 == a else int(local_index[b, f1])
    i2 = int(local_index[b, f2]) if t2 == b else int(local_index[a, f2])

    if i1 < 0 or i2 < 0:
        # should not happen if reachability and target_fibre_ids consistent
        return False

    # block is stored in (a,b) orientation with compressed indices:
    #   rows correspond to fibres that reach target a
    #   cols correspond to fibres that reach target b
    if t1 == a and t2 == b:
        return bool(block[i1, i2])
    elif t1 == b and t2 == a:
        return bool(block[i2, i1])
    else:
        # logically impossible due to a,b definition
        return False


# ----------------------------
# Step D: State + invariants
# ----------------------------
def validate_bijection(pivot_to_target: np.ndarray, target_to_pivot: np.ndarray):
    F = pivot_to_target.shape[0]
    seen = set()
    for f in range(F):
        t = int(pivot_to_target[f])
        if t >= 0:
            assert int(target_to_pivot[t]) == f
            assert t not in seen
            seen.add(t)

def count_alloc(pivot_to_target: np.ndarray) -> int:
    return int(np.sum(pivot_to_target >= 0))


# ----------------------------
# Step E: Energy handling (WEAVE Configure-style table you already compute)
# ----------------------------
energies_ft = np.array(energies_ft_j)  # (F,T) numpy; inf for unreachable

def total_energy(pivot_to_target: np.ndarray) -> float:
    e = 0.0
    for f, t in enumerate(pivot_to_target):
        t = int(t)
        if t >= 0:
            e += float(energies_ft[f, t])
    return e

def delta_energy(pivot_to_target: np.ndarray, moves: list[tuple[int,int]]) -> float:
    dE = 0.0
    for f, t_new in moves:
        t_old = int(pivot_to_target[f])
        if t_old >= 0:
            dE -= float(energies_ft[f, t_old])
        if t_new >= 0:
            dE += float(energies_ft[f, t_new])
    return dE


# ----------------------------
# Step F: Collision check for proposed atomic moves (correctness-first)
# ----------------------------
def proposal_collision_free(pivot_to_target: np.ndarray, moves: list[tuple[int,int]], local_index: np.ndarray) -> bool:
    """
    Check all moved fibres against all allocated fibres under the proposed state.
    O(#moves * #allocated), but fine for functionality-first.
    """
    # proposed target for moved fibres
    proposed = {int(f): int(t) for f, t in moves}

    def eff_target(f: int) -> int:
        return proposed.get(f, int(pivot_to_target[f]))

    F = pivot_to_target.shape[0]
    moved = list(proposed.keys())

    for f in moved:
        t = eff_target(f)
        if t < 0:
            continue

        # ensure target uniqueness (proposal should handle, but enforce here)
        for g in moved:
            if g != f and eff_target(g) == t:
                return False

        # check vs all other allocated fibres
        for g in range(F):
            if g == f:
                continue
            u = eff_target(g)
            if u < 0:
                continue
            if collides(f, t, g, u, local_index):
                return False

    return True


# ============================================================
# Miszalski-style "four paths" proposal kernel
# ============================================================

def random_reachable_target(rng, reach_targets, reach_mask, f):
    idx = np.where(reach_mask[f])[0]
    if len(idx) == 0:
        return -1
    k = int(rng.choice(idx))
    return int(reach_targets[f, k])

def propose_four_paths(
    rng,
    fA: int,
    pivot_to_target: np.ndarray,
    target_to_pivot: np.ndarray,
    reach_targets: np.ndarray,
    reach_mask: np.ndarray,
    local_index: np.ndarray,
    max_reloc_tries: int = 32,
):
    """
    Returns (ok, moves_list).
    moves_list is a list of (pivot -> new_target) applied atomically.
    Faithful structure to Miszalski: PivotA picks TargetB randomly; 4 cases depending
    on whether PivotA and TargetB are allocated.
    """
    tA = int(pivot_to_target[fA])
    tB = random_reachable_target(rng, reach_targets, reach_mask, fA)
    if tB < 0:
        return False, []

    fB = int(target_to_pivot[tB])  # -1 if unallocated

    # Case I: A free, B free -> assign A->B
    if tA < 0 and fB < 0:
        moves = [(fA, tB)]
        if proposal_collision_free(pivot_to_target, moves, local_index):
            return True, moves
        return False, []

    # Case II: A allocated, B free -> move A->B (old target becomes free)
    if tA >= 0 and fB < 0:
        moves = [(fA, tB)]
        if proposal_collision_free(pivot_to_target, moves, local_index):
            return True, moves
        return False, []

    # helper: relocate a pivot to a currently-free target
    def try_relocate(pivot: int, reserved_targets: set[int], pre_moves: list[tuple[int,int]]):
        idx = np.where(reach_mask[pivot])[0]
        if len(idx) == 0:
            return None
        for _ in range(max_reloc_tries):
            k = int(rng.choice(idx))
            tC = int(reach_targets[pivot, k])
            if tC < 0 or tC in reserved_targets:
                continue
            if int(target_to_pivot[tC]) != -1:
                continue  # must be free to conserve yield
            trial = pre_moves + [(pivot, tC)]
            if proposal_collision_free(pivot_to_target, trial, local_index):
                return tC
        return None

    # Case III: A free, B occupied -> A takes B; relocate B to free target C
    if tA < 0 and fB >= 0:
        pre = [(fA, tB)]
        tC = try_relocate(fB, {tB}, pre)
        if tC is None:
            return False, []
        moves = pre + [(fB, tC)]
        return True, moves

    # Case IV: A allocated, B occupied -> try swap; else A takes B and relocate B
    if tA >= 0 and fB >= 0:
        # try direct swap if fB can reach tA
        # (reach_targets is padded; easiest check is membership in its valid set)
        if np.any((reach_targets[fB] == tA) & reach_mask[fB]):
            moves = [(fA, tB), (fB, tA)]
            if proposal_collision_free(pivot_to_target, moves, local_index):
                return True, moves

        pre = [(fA, tB)]
        tC = try_relocate(fB, {tB}, pre)
        if tC is None:
            return False, []
        moves = pre + [(fB, tC)]
        return True, moves

    return False, []


def apply_moves(pivot_to_target: np.ndarray, target_to_pivot: np.ndarray, moves: list[tuple[int,int]]):
    """
    Apply moves atomically and maintain bijection.
    """
    piv = pivot_to_target.copy()
    tar = target_to_pivot.copy()

    # unassign moved pivots
    for f, _ in moves:
        old = int(piv[f])
        if old >= 0 and int(tar[old]) == f:
            tar[old] = -1
        piv[f] = -1

    # assign new
    for f, t in moves:
        if t >= 0:
            # target should be free; if not, clear (shouldn't happen if proposal correct)
            if int(tar[t]) != -1:
                other = int(tar[t])
                piv[other] = -1
            tar[t] = f
            piv[f] = t

    return piv, tar


def metropolis_accept(rng, dE: float, T: float) -> bool:
    if dE <= 0:
        return True
    x = -dE / max(T, 1e-12)
    if x < -700:
        return False
    return float(rng.random()) < float(np.exp(x))


# ============================================================
# STEPWISE runnable SA driver
# ============================================================

def greedy_initial_state(reach_targets, reach_mask, local_index, seed=0):
    """
    Simple feasible start state: iterate fibres, pick best-energy reachable target that is free+collision-free.
    """
    rng = np.random.default_rng(seed)
    F, T = energies_ft.shape
    piv = -np.ones(F, dtype=np.int32)
    tar = -np.ones(T, dtype=np.int32)

    order = rng.permutation(F)
    for f in order:
        ts = reach_targets[f][reach_mask[f]]
        if len(ts) == 0:
            continue
        # sort by energy
        ts = ts[np.argsort(energies_ft[f, ts])]
        for t in ts:
            t = int(t)
            if tar[t] != -1:
                continue
            moves = [(int(f), t)]
            if proposal_collision_free(piv, moves, local_index):
                piv, tar = apply_moves(piv, tar, moves)
                break

    validate_bijection(piv, tar)
    E = total_energy(piv)
    print("[INIT] allocated:", count_alloc(piv), "/", F, "energy:", E)
    return piv, tar, E


def anneal_miszalski_weave(
    piv0, tar0, E0,
    reach_targets, reach_mask, local_index,
    Ti=1.0, Tf=1e-3, cooling_rate=0.02,
    max_iters_per_T=5,
    n_sweeps_cap=None,
    seed=123,
    verbose=True,
):
    """
    Miszalski-style structure:
      for each T:
        repeat MaxIters:
          traverse all pivots in random order:
            pick random reachable TargetB
            apply one of four paths
            accept with Metropolis

    Uses your WEAVE-style energy table energies_ft.
    """
    rng = np.random.default_rng(seed)
    piv = piv0.copy()
    tar = tar0.copy()
    E = float(E0)
    F = piv.shape[0]

    Tcur = float(Ti)
    temp_step = 0

    while Tcur > Tf:
        for _ in range(max_iters_per_T):
            order = rng.permutation(F)
            for fA in order:
                ok, moves = propose_four_paths(
                    rng,
                    int(fA),
                    piv, tar,
                    reach_targets, reach_mask,
                    local_index,
                    max_reloc_tries=32
                )
                if not ok:
                    continue
                dE = delta_energy(piv, moves)
                if metropolis_accept(rng, dE, Tcur):
                    piv, tar = apply_moves(piv, tar, moves)
                    E += dE

        if verbose:
            print(f"[SA] T={Tcur:.5g}  allocated={count_alloc(piv)}/{F}  energy={E:.4f}")

        validate_bijection(piv, tar)

        temp_step += 1
        if n_sweeps_cap is not None and temp_step >= n_sweeps_cap:
            break

        Tcur *= (1.0 - cooling_rate)

    return piv, tar, E


# ============================================================
# RUN IN ORDER: A -> F
# ============================================================

# A) reachability lists per fibre
reach_targets, reach_mask = build_reach_targets_per_pivot(allowed_np)

# B) local index mapping (target, global fibre) -> compressed index
local_index = build_local_index_map(target_fibre_ids, n_fibres=allowed_np.shape[0])

# C) quick collision sanity check
# pick random stored collision block and confirm dimensions
if len(collision_matrix) > 0:
    (t1, t2), block = next(iter(collision_matrix.items()))
    print("[C] example collision block key:", (t1, t2), "shape:", block.shape)

# D) build an initial feasible configuration
piv0, tar0, E0 = greedy_initial_state(reach_targets, reach_mask, local_index, seed=0)

# E) run Miszalski/WEAVE-style SA
pivF, tarF, EF = anneal_miszalski_weave(
    piv0, tar0, E0,
    reach_targets, reach_mask, local_index,
    Ti=1.0, Tf=1e-3, cooling_rate=0.02,
    max_iters_per_T=5,
    seed=123,
    verbose=True,
)

print("[FINAL] allocated:", count_alloc(pivF), "/", pivF.shape[0], "energy:", EF)

# F) optional: posterior-like sampling at fixed Tf (Boltzmann at Tf)
def sample_fixed_T(piv_start, tar_start, E_start, Tfixed=1e-3, n_burn=20, n_samples=50, thin=5, seed=456):
    rng = np.random.default_rng(seed)
    piv = piv_start.copy()
    tar = tar_start.copy()
    E = float(E_start)
    F = piv.shape[0]

    def one_sweep():
        nonlocal piv, tar, E
        order = rng.permutation(F)
        for fA in order:
            ok, moves = propose_four_paths(
                rng, int(fA),
                piv, tar,
                reach_targets, reach_mask,
                local_index
            )
            if not ok:
                continue
            dE = delta_energy(piv, moves)
            if metropolis_accept(rng, dE, Tfixed):
                piv, tar = apply_moves(piv, tar, moves)
                E += dE

    for _ in range(n_burn):
        one_sweep()

    samples = []
    for _ in range(n_samples):
        for _ in range(thin):
            one_sweep()
        samples.append(piv.copy())

    return samples

# samples = sample_fixed_T(pivF, tarF, EF, Tfixed=1e-3, n_burn=20, n_samples=50, thin=5)
# print("[SAMPLES] collected:", len(samples))
