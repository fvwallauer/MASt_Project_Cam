import os
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

# =============================================================================
# OUTPUT DIRECTORY FOR PLOTS
# =============================================================================
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ===============================================
# PLATE PARAMETERS
# -----------------------------------------------
n_retractors = 168  # 168
r = 102.5  # mm, half of 205mm field
n_tiers = 3
n_fibres = n_retractors * n_tiers

# Retractor coordinates & constraints
angles = jnp.linspace(0, 2*jnp.pi, n_retractors, endpoint=False)  # placement of fibres
retractor_x = jnp.asarray(r * jnp.cos(angles))
retractor_y = jnp.asarray(r * jnp.sin(angles))
fibre_span_allowed = jnp.asarray([r*0.4, r*0.7, r*1.3])  # for tiers 0,1,2 (low-high)
phi_max = jnp.deg2rad(14)  # max bend angle

# Button vertices in local fibre-target coordinates
# box of 9mm length, 2mm width
button_vertices_local = jnp.array([
    [-1,  6],
    [ 1,  6],
    [ 1, -3],
    [-1, -3],
])

# Minimum possible target separation
min_sep = 2  # mm; min. distance of separation below which buttons overlap

# -----------------------------------------------
# FIELD PARAMETERS
# -----------------------------------------------
n_targets = 1400  # 1400

# ===============================================

def rand_targets(R, n):
    """
    Generate random targets uniform over a disc.
    """
    rng = np.random.default_rng(0)
    p = rng.random(n)
    q = rng.random(n)
    rr = R * np.sqrt(p)
    theta = 2*np.pi*q
    x = rr * np.cos(theta)
    y = rr * np.sin(theta)
    return jnp.asarray(x), jnp.asarray(y)

def fibre_id(retractor_id, tier_id):
    if tier_id < 0 or tier_id >= n_tiers:
        raise ValueError("Invalid tier_id")
    return retractor_id * n_tiers + tier_id

def fibre_rt(fibre_id_):
    r_id = fibre_id_ // n_tiers
    t_id = fibre_id_ % n_tiers
    return r_id, t_id

# -----------------------------------------------
# PHASE 1: Reachability & button footprints
# -----------------------------------------------
@jit
def TF_reach_matrix(px, py):
    fibre_ids = jnp.arange(n_fibres)      # (F,)
    r_ids = fibre_ids // n_tiers          # (F,)
    t_ids = fibre_ids % n_tiers           # (F,)

    rx = retractor_x[r_ids]               # (F,)
    ry = retractor_y[r_ids]               # (F,)
    tier_span = fibre_span_allowed[t_ids] # (F,)

    pos_r = jnp.stack([rx, ry], axis=1)  # (F,2)
    pos_r_norm_inward = -pos_r / jnp.linalg.norm(pos_r, axis=1, keepdims=True)

    vx = px[None, :] - rx[:, None]     # (F, T)
    vy = py[None, :] - ry[:, None]     # (F, T)
    v_len = jnp.sqrt(vx**2 + vy**2) + 1e-12

    ux = vx / v_len
    uy = vy / v_len

    dot = (ux * pos_r_norm_inward[:, 0:1] + uy * pos_r_norm_inward[:, 1:2])
    dot = jnp.clip(dot, -1.0, 1.0)
    ang = jnp.arccos(dot)

    span_allowed = tier_span[:, None]
    allowed = (v_len <= span_allowed) & (ang <= phi_max)
    return allowed

@jit
def all_button_vertices(px, py, allowed):
    F = allowed.shape[0]
    T = allowed.shape[1]

    fibre_ids = jnp.arange(F)
    r_ids = fibre_ids // n_tiers

    rx = retractor_x[r_ids, None]
    ry = retractor_y[r_ids, None]

    tx = px[None, :]
    ty = py[None, :]

    vx = tx - rx
    vy = ty - ry
    phi = jnp.arctan2(vy, vx)
    theta = phi + jnp.pi / 2

    c = jnp.cos(theta)
    s = jnp.sin(theta)

    bx = button_vertices_local[:, 0]
    by = button_vertices_local[:, 1]

    bx_broadcast = bx[None, None, :]
    by_broadcast = by[None, None, :]

    x_rot = c[:, :, None] * bx_broadcast - s[:, :, None] * by_broadcast
    y_rot = s[:, :, None] * bx_broadcast + c[:, :, None] * by_broadcast

    x_vertex = px[None, :, None] + x_rot
    y_vertex = py[None, :, None] + y_rot

    mask = allowed[:, :, None]
    x_vertex = jnp.where(mask, x_vertex, jnp.nan)
    y_vertex = jnp.where(mask, y_vertex, jnp.nan)

    vertices = jnp.stack([x_vertex, y_vertex], axis=-1)  # (F, T, 4, 2)
    return vertices

@jax.jit
def compute_target_bboxes(vertices):
    xmin = jnp.nanmin(vertices[..., 0], axis=(0, 2))
    xmax = jnp.nanmax(vertices[..., 0], axis=(0, 2))
    ymin = jnp.nanmin(vertices[..., 1], axis=(0, 2))
    ymax = jnp.nanmax(vertices[..., 1], axis=(0, 2))
    footprint_valid = jnp.isfinite(xmin)
    return xmin, ymin, xmax, ymax, footprint_valid

@jax.jit
def compute_bbox_overlap(xmin, ymin, xmax, ymax, footprint_valid):
    xmin_i = xmin[:, None];  xmin_j = xmin[None, :]
    xmax_i = xmax[:, None];  xmax_j = xmax[None, :]
    ymin_i = ymin[:, None];  ymin_j = ymin[None, :]
    ymax_i = ymax[:, None];  ymax_j = ymax[None, :]

    overlap = (
        (xmax_i >= xmin_j) &
        (xmax_j >= xmin_i) &
        (ymax_i >= ymin_j) &
        (ymax_j >= ymin_i)
    )

    valid_i = footprint_valid[:, None]
    valid_j = footprint_valid[None, :]
    overlap = overlap & valid_i & valid_j

    T = xmin.shape[0]
    overlap = overlap & (~jnp.eye(T, dtype=bool))

    overlap = jnp.triu(overlap, k=1)
    overlap = overlap | overlap.T
    return overlap

# ----------------------------------------------------------------
# PHASE 2 — pair-level checks independent of fibre assignment
# ----------------------------------------------------------------
@jax.jit
def always_collide(px, py, reachable_Ts, min_sep):
    xi = px[:, None]
    yi = py[:, None]
    dx = xi - xi.T
    dy = yi - yi.T
    d2 = dx**2 + dy**2

    T = px.shape[0]
    d2 = d2 + jnp.eye(T) * jnp.inf

    collide = d2 < (min_sep * min_sep)
    both_reachable = reachable_Ts[:, None] & reachable_Ts[None, :]
    always_c = collide & both_reachable

    always_c = jnp.triu(always_c, k=1)
    always_c = always_c | always_c.T
    always_c = jnp.where(jnp.eye(T, dtype=bool), False, always_c)
    return always_c

def seg_seg_intersect(p, r, q, s):
    cross = lambda a, b: a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]
    rxs = cross(r, s)
    q_p = q - p
    parallel = jnp.isclose(rxs, 0)
    rxs_safe = jnp.where(parallel, 1.0, rxs)

    t = cross(q_p, s) / rxs_safe
    u = cross(q_p, r) / rxs_safe

    intersect = (~parallel) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    return intersect

@jax.jit
def compute_path_bboxes(px, py, retractor_x, retractor_y, r_id_for_fibre, allowed):
    F, T = allowed.shape
    rx_f = retractor_x[r_id_for_fibre]
    ry_f = retractor_y[r_id_for_fibre]
    rx = rx_f[:, None]
    ry = ry_f[:, None]

    tx = jnp.broadcast_to(px[None, :], (F, T))
    ty = jnp.broadcast_to(py[None, :], (F, T))

    seg_xmin = jnp.minimum(tx, rx)
    seg_xmax = jnp.maximum(tx, rx)
    seg_ymin = jnp.minimum(ty, ry)
    seg_ymax = jnp.maximum(ty, ry)

    big = 1e30
    seg_xmin = jnp.where(allowed, seg_xmin, big)
    seg_ymin = jnp.where(allowed, seg_ymin, big)
    seg_xmax = jnp.where(allowed, seg_xmax, -big)
    seg_ymax = jnp.where(allowed, seg_ymax, -big)

    path_xmin = jnp.min(seg_xmin, axis=0)
    path_ymin = jnp.min(seg_ymin, axis=0)
    path_xmax = jnp.max(seg_xmax, axis=0)
    path_ymax = jnp.max(seg_ymax, axis=0)

    return path_xmin, path_ymin, path_xmax, path_ymax

@jax.jit
def compute_fibre_segments(px, py, retractor_x, retractor_y, r_id_for_fibre, allowed):
    F, T = allowed.shape
    target_pos = jnp.stack([px, py], axis=-1)
    start = jnp.broadcast_to(target_pos[None, :, :], (F, T, 2))

    rx = retractor_x[r_id_for_fibre]
    ry = retractor_y[r_id_for_fibre]
    end = jnp.stack([rx[:, None], ry[:, None]], axis=-1)  # (F,1,2)

    vector = end - start
    mask = allowed[..., None]
    start = jnp.where(mask, start, jnp.nan)
    vector = jnp.where(mask, vector, jnp.nan)
    return start, vector

@jax.jit
def fibre_collision_block_padded(target_vertices, target_fibre_start, target_fibre_dir, target_fibre_mask, t1, t2):
    verts_i = target_vertices[t1]
    verts_j = target_vertices[t2]

    edges_i = jnp.roll(verts_i, -1, axis=1) - verts_i
    edges_j = jnp.roll(verts_j, -1, axis=1) - verts_j

    mask_i = target_fibre_mask[t1]
    mask_j = target_fibre_mask[t2]

    p_bb = verts_i[:, None, :, None, :]
    r_bb = edges_i[:, None, :, None, :]
    q_bb = verts_j[None, :, None, :, :]
    s_bb = edges_j[None, :, None, :, :]

    intersects_bb = seg_seg_intersect(p_bb, r_bb, q_bb, s_bb)
    button_intersect = jnp.any(intersects_bb, axis=(2, 3))

    start_i = target_fibre_start[t1]
    dir_i   = target_fibre_dir[t1]

    p_fb1 = start_i[:, None, None, :]
    r_fb1 = dir_i[:, None, None, :]
    q_fb1 = verts_j[None, :, :, :]
    s_fb1 = edges_j[None, :, :, :]

    intersects_fb1 = seg_seg_intersect(p_fb1, r_fb1, q_fb1, s_fb1)
    f1_on_b2 = jnp.any(intersects_fb1, axis=2)

    start_j = target_fibre_start[t2]
    dir_j   = target_fibre_dir[t2]

    p_fb2 = start_j[None, :, None, :]
    r_fb2 = dir_j[None, :, None, :]
    q_fb2 = verts_i[:, None, :, :]
    s_fb2 = edges_i[:, None, :, :]

    intersects_fb2 = seg_seg_intersect(p_fb2, r_fb2, q_fb2, s_fb2)
    f2_on_b1 = jnp.any(intersects_fb2, axis=2)

    block = button_intersect | f1_on_b2 | f2_on_b1

    valid_i = mask_i[:, None]
    valid_j = mask_j[None, :]
    valid_pair = valid_i & valid_j
    block = block & valid_pair
    return block

@jax.jit
def batch_collision_blocks(target_vertices, target_fibre_start, target_fibre_dir, target_fibre_mask, t1_batch, t2_batch):
    def one_pair(t1, t2):
        return fibre_collision_block_padded(
            target_vertices, target_fibre_start, target_fibre_dir,
            target_fibre_mask, t1, t2
        )
    return jax.vmap(one_pair)(t1_batch, t2_batch)

# ================================================
# COLLISION MATRIX
# ================================================
fibre_ids = jnp.arange(n_fibres)
r_id_for_fibre = fibre_ids // n_tiers
t_id_for_fibre = fibre_ids % n_tiers

px, py = rand_targets(r, n_targets)

allowed = TF_reach_matrix(px, py)
reachable_Ts = jnp.any(allowed, axis=0)

always_collide_mat = always_collide(px, py, reachable_Ts, min_sep)

vertices = all_button_vertices(px, py, allowed)
xmin, ymin, xmax, ymax, footprint_valid = compute_target_bboxes(vertices)

path_xmin, path_ymin, path_xmax, path_ymax = compute_path_bboxes(
    px, py, retractor_x, retractor_y, r_id_for_fibre, allowed
)

pxmin = path_xmin[:, None]
pymin = path_ymin[:, None]
pxmax = path_xmax[:, None]
pymax = path_ymax[:, None]

fxmin = xmin[None, :]
fymin = ymin[None, :]
fxmax = xmax[None, :]
fymax = ymax[None, :]

pathA_hits_footB = (
    (pxmin <= fxmax) &
    (pxmax >= fxmin) &
    (pymin <= fymax) &
    (pymax >= fymin)
)
path_bbox_overlap = pathA_hits_footB | pathA_hits_footB.T
path_bbox_overlap = path_bbox_overlap & (~jnp.eye(path_bbox_overlap.shape[0], dtype=bool))

bbox_overlap = compute_bbox_overlap(xmin, ymin, xmax, ymax, footprint_valid)
needs_fibre_level = (~always_collide_mat) & (bbox_overlap | path_bbox_overlap)

print("Pairs needing fibre-level checks:", int(needs_fibre_level.sum()/2))

fibre_start, fibre_dir = compute_fibre_segments(px, py, retractor_x, retractor_y, r_id_for_fibre, allowed)

allowed_np = np.array(allowed)

reachable_fibres_per_target = [
    np.where(allowed_np[:, t])[0].astype(np.int32)
    for t in range(allowed_np.shape[1])
]
avg_nfib = np.mean([len(fibs) for fibs in reachable_fibres_per_target])
print("Average fibres per target:", avg_nfib)

max_F = max(len(fibs) for fibs in reachable_fibres_per_target)
print("Max fibres per target:", max_F)

n_targets = allowed_np.shape[1]

target_fibre_ids  = -np.ones((n_targets, max_F), dtype=np.int32)
target_fibre_mask = np.zeros((n_targets, max_F), dtype=bool)

for t, fibs in enumerate(reachable_fibres_per_target):
    n = min(len(fibs), max_F)
    target_fibre_ids[t, :n]  = fibs[:n]
    target_fibre_mask[t, :n] = True

vertices_np = np.array(vertices)
fibre_start_np = np.array(fibre_start)
fibre_dir_np = np.array(fibre_dir)

target_vertices = np.zeros((n_targets, max_F, 4, 2), dtype=vertices_np.dtype)
target_fibre_start = np.zeros((n_targets, max_F, 2), dtype=fibre_start_np.dtype)
target_fibre_dir = np.zeros((n_targets, max_F, 2), dtype=fibre_dir_np.dtype)

for t in range(n_targets):
    ids = target_fibre_ids[t]
    valid = ids >= 0
    if np.any(valid):
        target_vertices[t, valid, :, :] = vertices_np[ids[valid], t, :, :]
        target_fibre_start[t, valid, :] = fibre_start_np[ids[valid], t, :]
        target_fibre_dir[t, valid, :] = fibre_dir_np[ids[valid], t, :]

target_vertices_j = jnp.array(target_vertices)
target_fibre_start_j = jnp.array(target_fibre_start)
target_fibre_dir_j = jnp.array(target_fibre_dir)
target_fibre_mask_j = jnp.array(target_fibre_mask)

needs_fibre_level_np = np.array(needs_fibre_level)
pairs_i, pairs_j = np.where(np.triu(needs_fibre_level_np, k=1))
pairs = np.stack([pairs_i, pairs_j], axis=1).astype(np.int32)

print("Unique pairs needing fibre-level checks:", len(pairs))

B = 256
pairs_j_list = []
blocks_j_list = []

for i in tqdm(range(0, len(pairs), B), desc="Building fibre-level collision blocks"):
    batch = pairs[i:i+B]
    t1_batch = jnp.array(batch[:, 0], dtype=jnp.int32)
    t2_batch = jnp.array(batch[:, 1], dtype=jnp.int32)

    blocks_full = batch_collision_blocks(
        target_vertices_j,
        target_fibre_start_j,
        target_fibre_dir_j,
        target_fibre_mask_j,
        t1_batch,
        t2_batch,
    )

    pairs_j_list.append(jnp.array(batch, dtype=jnp.int32))
    blocks_j_list.append(blocks_full)

pairs_j = jnp.concatenate(pairs_j_list, axis=0)
blocks_j = jnp.concatenate(blocks_j_list, axis=0)

print("pairs_j shape:", pairs_j.shape)
print("blocks_j shape:", blocks_j.shape)

T = int(n_targets)
pair_index_np = -np.ones((T, T), dtype=np.int32)

for k, (t1, t2) in enumerate(pairs):
    pair_index_np[int(t1), int(t2)] = k
    pair_index_np[int(t2), int(t1)] = k

pair_index_j = jnp.array(pair_index_np)
print("pair_index_j shape:", pair_index_j.shape)

# =============================================================================
# SA SETUP
# =============================================================================
always_collide_np = np.array(always_collide_mat)
blocks_np = np.array(blocks_j)
px_np = np.array(px)
py_np = np.array(py)
retractor_x_np = np.array(retractor_x)
retractor_y_np = np.array(retractor_y)
r_id_np = np.array(r_id_for_fibre)  # fibre -> retractor id

# STATE ARRAYS
fibre_to_target = -np.ones(n_fibres, dtype=np.int32)
target_to_slot  = -np.ones(n_targets, dtype=np.int32)
target_to_fibre = -np.ones(n_targets, dtype=np.int32)

# Reverse mapping: fibre -> list of (target, slot)
fibre_to_possible = [[] for _ in range(n_fibres)]
for t in range(n_targets):
    for s in range(max_F):
        if target_fibre_mask[t, s]:
            f = int(target_fibre_ids[t, s])
            fibre_to_possible[f].append((t, s))

# Priorities (toy example)
_rng_pri = np.random.default_rng(99)
priorities = np.where(_rng_pri.random(n_targets) > 0.5, 10.0, 1.0)

# Straightness precompute
straightness = np.zeros((n_targets, max_F), dtype=np.float32)
_phi_max = np.deg2rad(14.0)
for t in range(n_targets):
    for s in range(max_F):
        if not target_fibre_mask[t, s]:
            continue
        f = int(target_fibre_ids[t, s])
        rid = int(r_id_np[f])
        vx = px_np[t] - retractor_x_np[rid]
        vy = py_np[t] - retractor_y_np[rid]
        rx, ry = -retractor_x_np[rid], -retractor_y_np[rid]
        cos_a = (vx*rx + vy*ry) / (np.hypot(vx, vy) * np.hypot(rx, ry) + 1e-12)
        straightness[t, s] = np.arccos(np.clip(cos_a, -1, 1)) / _phi_max

# Collision lookup
def collides(ti, si, tj, sj):
    if always_collide_np[ti, tj]:
        return True
    k = pair_index_np[ti, tj]
    if k == -1:
        return False
    return bool(blocks_np[k, si, sj])

def placement_valid(ti, si):
    for tj in range(n_targets):
        if target_to_fibre[tj] == -1 or tj == ti:
            continue
        sj = target_to_slot[tj]
        if collides(ti, si, tj, sj):
            return False
    return True

def placement_valid_excluding(ti, si, exclude):
    for tj in range(n_targets):
        if target_to_fibre[tj] == -1 or tj == ti or tj in exclude:
            continue
        sj = target_to_slot[tj]
        if collides(ti, si, tj, sj):
            return False
    return True

def find_slot(fibre_id, target_id):
    for s in range(max_F):
        if target_fibre_mask[target_id, s] and target_fibre_ids[target_id, s] == fibre_id:
            return s
    return -1

# Assign/unassign
def assign(f, t, s):
    fibre_to_target[f] = t
    target_to_slot[t] = s
    target_to_fibre[t] = f

def unassign_fibre(f):
    t = fibre_to_target[f]
    if t == -1:
        return -1, -1
    s = target_to_slot[t]
    fibre_to_target[f] = -1
    target_to_slot[t] = -1
    target_to_fibre[t] = -1
    return int(t), int(s)

def unassign_target(t):
    f = target_to_fibre[t]
    if f == -1:
        return -1, -1
    s = target_to_slot[t]
    fibre_to_target[f] = -1
    target_to_slot[t] = -1
    target_to_fibre[t] = -1
    return int(f), int(s)

# Energy
UNASSIGNED_PENALTY = 2.0

def energy_single(t, s):
    return (1.0 + straightness[t, s]) / priorities[t]

def energy_total():
    E = 0.0
    for f in range(n_fibres):
        t = fibre_to_target[f]
        if t == -1:
            E += UNASSIGNED_PENALTY
        else:
            E += energy_single(int(t), int(target_to_slot[t]))
    return float(E)

# Init
def init_assignment(seed=42):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_fibres)
    placed = 0
    for f in order:
        f = int(f)
        candidate_placements = list(fibre_to_possible[f])
        rng.shuffle(candidate_placements)
        for (t, s) in candidate_placements:
            t = int(t); s = int(s)
            if target_to_fibre[t] != -1:
                continue
            if placement_valid(t, s):
                assign(f, t, s)
                placed += 1
                break
    print(f"Initial assignment: {placed} out of {n_fibres} fibres placed")
    return placed

def reset_state():
    fibre_to_target[:] = -1
    target_to_slot[:]  = -1
    target_to_fibre[:] = -1

def accept(dE, T, rng):
    if dE <= 0:
        return True
    return rng.random() < np.exp(-1000.0 * dE / T)

# =============================================================================
# PLOTTING / DIAGNOSTICS
# =============================================================================
def snapshot_state():
    return {
        "fibre_to_target": fibre_to_target.copy(),
        "target_to_slot":  target_to_slot.copy(),
        "target_to_fibre": target_to_fibre.copy(),
    }

def plot_assignment(snapshot, filename, title="Assignment"):
    f2t = snapshot["fibre_to_target"]
    assigned_fibres = np.where(f2t >= 0)[0]
    assigned_targets = f2t[assigned_fibres].astype(int)
    rid = r_id_np[assigned_fibres].astype(int)

    tx = px_np[assigned_targets]
    ty = py_np[assigned_targets]
    rx = retractor_x_np[rid]
    ry = retractor_y_np[rid]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(px_np, py_np, s=5, alpha=0.15, label="All targets")
    ax.scatter(tx, ty, s=15, alpha=0.9, label="Assigned targets")
    ax.scatter(retractor_x_np, retractor_y_np, s=8, alpha=0.9, label="Retractors")

    for x0, y0, x1, y1 in zip(rx, ry, tx, ty):
        ax.plot([x0, x1], [y0, y1], linewidth=0.7, alpha=0.5)

    circ = plt.Circle((0, 0), r, fill=False, linewidth=1.5, alpha=0.8)
    ax.add_patch(circ)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.05*r, 1.05*r)
    ax.set_ylim(-1.05*r, 1.05*r)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"{title}\nassigned fibres: {len(assigned_fibres)}/{n_fibres}")
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)

def plot_anneal_history(history, filename_prefix):
    steps = np.arange(len(history["T"]))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, history["E"])
    ax.set_xlabel("temperature step")
    ax.set_ylabel("Energy")
    ax.set_title("Energy during annealing")
    fig.tight_layout()
    fig.savefig(f"{filename_prefix}_energy.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, history["assigned"])
    ax.set_xlabel("temperature step")
    ax.set_ylabel("Assigned fibres")
    ax.set_title("Assigned fibres during annealing")
    fig.tight_layout()
    fig.savefig(f"{filename_prefix}_assigned.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, history["accepted"])
    ax.set_xlabel("temperature step")
    ax.set_ylabel("Cumulative accepted swaps")
    ax.set_title("Accepted swaps during annealing")
    fig.tight_layout()
    fig.savefig(f"{filename_prefix}_accepted.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, history["swap1"], label="type 1 (assign)")
    ax.plot(steps, history["swap2"], label="type 2 (move)")
    ax.plot(steps, history["swap3"], label="type 3 (replace)")
    ax.plot(steps, history["swap4"], label="type 4 (swap)")
    ax.set_xlabel("temperature step")
    ax.set_ylabel("Cumulative count")
    ax.set_title("Move types accepted")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{filename_prefix}_swap_types.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, history["T"])
    ax.set_xlabel("temperature step")
    ax.set_ylabel("Temperature")
    ax.set_title("Cooling schedule")
    fig.tight_layout()
    fig.savefig(f"{filename_prefix}_temperature.png", dpi=200)
    plt.close(fig)

def plot_tier_counts(snapshot, filename, title="Tier composition"):
    """
    Bar chart: how many assigned fibres are in tier 0/1/2.
    Tier is fibre_id % n_tiers.
    """
    f2t = snapshot["fibre_to_target"]
    assigned_fibres = np.where(f2t >= 0)[0]
    if len(assigned_fibres) == 0:
        counts = np.zeros(n_tiers, dtype=int)
    else:
        tiers = assigned_fibres % n_tiers
        counts = np.array([(tiers == k).sum() for k in range(n_tiers)], dtype=int)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(np.arange(n_tiers), counts)
    ax.set_xticks(np.arange(n_tiers))
    ax.set_xticklabels([f"tier {k}" for k in range(n_tiers)])
    ax.set_ylabel("Assigned fibres")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)

def _assigned_straightness_values(snapshot):
    """
    Return straightness[t, slot] for each assigned fibre placement in snapshot.
    """
    f2t = snapshot["fibre_to_target"]
    t2s = snapshot["target_to_slot"]

    assigned_fibres = np.where(f2t >= 0)[0]
    if len(assigned_fibres) == 0:
        return np.array([], dtype=np.float32)

    targets = f2t[assigned_fibres].astype(int)
    slots = t2s[targets].astype(int)
    vals = straightness[targets, slots]
    vals = vals[np.isfinite(vals)]
    return vals.astype(np.float32)

def plot_straightness_hist(snapshot_init, snapshot_final, filename, bins=30):
    """
    Histogram comparing straightness distribution of assigned placements:
    init vs final.
    """
    v0 = _assigned_straightness_values(snapshot_init)
    v1 = _assigned_straightness_values(snapshot_final)

    fig, ax = plt.subplots(figsize=(8, 4))
    if len(v0) > 0:
        ax.hist(v0, bins=bins, alpha=0.5, label="initial")
    if len(v1) > 0:
        ax.hist(v1, bins=bins, alpha=0.5, label="final")

    ax.set_xlabel("straightness = (bend angle) / phi_max")
    ax.set_ylabel("count")
    ax.set_title("Straightness distribution of assigned fibres")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)

# =============================================================================
# ANNEALING (returns history)
# =============================================================================
def anneal(T_i=10.0, T_f=0.1, delta_T=0.01, max_iters=20, seed=42, return_history=True):
    rng = np.random.default_rng(seed)
    T = float(T_i)
    n_steps = 0
    n_accepted = 0
    swap_counts = {1: 0, 2: 0, 3: 0, 4: 0}

    history = {"T": [], "E": [], "assigned": [], "accepted": [],
               "swap1": [], "swap2": [], "swap3": [], "swap4": []}

    best_E = energy_total()
    print(f"Initial state: E={best_E:.2f}, T={T:.4f}, assigned={int(np.sum(fibre_to_target >= 0))}")

    while T > T_f:
        for _ in range(max_iters):
            for fibre_a in rng.permutation(n_fibres):
                fibre_a = int(fibre_a)
                possible = fibre_to_possible[fibre_a]
                if len(possible) == 0:
                    continue

                target_b, slot_ab = possible[rng.integers(len(possible))]
                target_b = int(target_b); slot_ab = int(slot_ab)

                target_a = int(fibre_to_target[fibre_a])
                fibre_b  = int(target_to_fibre[target_b])

                if target_a == target_b:
                    continue

                n_steps += 1
                accepted_flag = False

                if target_a == -1 and fibre_b == -1:
                    if placement_valid(target_b, slot_ab):
                        dE = energy_single(target_b, slot_ab) - UNASSIGNED_PENALTY
                        if accept(dE, T, rng):
                            assign(fibre_a, target_b, slot_ab)
                            accepted_flag = True
                            swap_counts[1] += 1

                elif target_a != -1 and fibre_b == -1:
                    slot_aa = int(target_to_slot[target_a])
                    if placement_valid_excluding(target_b, slot_ab, exclude={target_a}):
                        dE = energy_single(target_b, slot_ab) - energy_single(target_a, slot_aa)
                        if accept(dE, T, rng):
                            unassign_fibre(fibre_a)
                            assign(fibre_a, target_b, slot_ab)
                            accepted_flag = True
                            swap_counts[2] += 1

                elif target_a == -1 and fibre_b != -1:
                    slot_bb = int(target_to_slot[target_b])
                    if placement_valid_excluding(target_b, slot_ab, exclude={target_b}):
                        dE = energy_single(target_b, slot_ab) - energy_single(target_b, slot_bb)
                        if accept(dE, T, rng):
                            unassign_target(target_b)
                            assign(fibre_a, target_b, slot_ab)
                            accepted_flag = True
                            swap_counts[3] += 1

                else:
                    slot_aa = int(target_to_slot[target_a])
                    slot_bb = int(target_to_slot[target_b])
                    slot_ba = int(find_slot(fibre_b, target_a))
                    if slot_ba != -1:
                        check1 = placement_valid_excluding(target_b, slot_ab, exclude={target_a, target_b})
                        check2 = placement_valid_excluding(target_a, slot_ba, exclude={target_a, target_b})
                        check3 = not collides(target_b, slot_ab, target_a, slot_ba)

                        if check1 and check2 and check3:
                            dE = (
                                energy_single(target_b, slot_ab) + energy_single(target_a, slot_ba)
                                - energy_single(target_a, slot_aa) - energy_single(target_b, slot_bb)
                            )
                            if accept(dE, T, rng):
                                unassign_fibre(fibre_a)
                                unassign_target(target_b)
                                assign(fibre_a, target_b, slot_ab)
                                assign(fibre_b, target_a, slot_ba)
                                accepted_flag = True
                                swap_counts[4] += 1

                if accepted_flag:
                    n_accepted += 1

        # cool
        T *= (1.0 - delta_T)

        # record (once per temperature step)
        cur_E = energy_total()
        n_assigned = int(np.sum(fibre_to_target >= 0))

        print(f"  T={T:.4f}  E={cur_E:.2f}  assigned={n_assigned}  accepted={n_accepted}  steps={n_steps}")

        history["T"].append(float(T))
        history["E"].append(float(cur_E))
        history["assigned"].append(int(n_assigned))
        history["accepted"].append(int(n_accepted))
        history["swap1"].append(int(swap_counts[1]))
        history["swap2"].append(int(swap_counts[2]))
        history["swap3"].append(int(swap_counts[3]))
        history["swap4"].append(int(swap_counts[4]))

    print(f"\nAnnealing complete. Swap counts: {swap_counts}")
    print(f"Final: E={energy_total():.2f}, assigned={int(np.sum(fibre_to_target >= 0))}/{n_fibres}")

    return history if return_history else None

# =============================================================================
# RUN + SAVE FIGURES
# =============================================================================
reset_state()
init_assignment(seed=42)
snap_init = snapshot_state()
print(f"Initial energy: {energy_total():.2f}\n")

history = anneal(T_i=10.0, T_f=0.1, delta_T=0.01, max_iters=20, seed=123, return_history=True)
snap_final = snapshot_state()

# Main configuration plots
plot_assignment(snap_init, os.path.join(PLOT_DIR, "assignment_initial.png"), title="Initial greedy assignment")
plot_assignment(snap_final, os.path.join(PLOT_DIR, "assignment_final.png"), title="Final annealed assignment")

# Annealing diagnostics
plot_anneal_history(history, os.path.join(PLOT_DIR, "anneal"))

# Extra diagnostics requested
plot_tier_counts(snap_final, os.path.join(PLOT_DIR, "tier_counts_final.png"), title="Tier composition (final)")
plot_straightness_hist(snap_init, snap_final, os.path.join(PLOT_DIR, "straightness_hist_init_vs_final.png"), bins=30)

print(f"\nSaved plots to: ./{PLOT_DIR}/")
print("  assignment_initial.png")
print("  assignment_final.png")
print("  anneal_energy.png")
print("  anneal_assigned.png")
print("  anneal_accepted.png")
print("  anneal_swap_types.png")
print("  anneal_temperature.png")
print("  tier_counts_final.png")
print("  straightness_hist_init_vs_final.png")