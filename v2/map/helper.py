import numpy as np
import math
from numba import njit, prange

# =========================================================
# PRIMITIVE HELPERS  (inlined into hot paths – kept for reuse)
# =========================================================

@njit(fastmath=True, cache=True, inline='always')
def hash2(x, y, seed):
    n = x * np.int64(374761393) + y * np.int64(668265263) + seed * np.int64(1442695041)
    n = (n ^ (n >> np.int64(13))) * np.int64(1274126177)
    return (n ^ (n >> np.int64(16))) & np.int64(0xffffffff)


@njit(fastmath=True, cache=True, inline='always')
def lerp(a, b, t):
    return a + t * (b - a)


@njit(fastmath=True, cache=True, inline='always')
def fade(t):
    return t * t * (3.0 - 2.0 * t)


# =========================================================
# GRADIENT NOISE  — Perlin-style, eliminates value-noise grid
# artefacts and the "puffy terrain" bias.
#
# 8 unit-length gradient vectors for 2-D:
#   (±1,±1), (±1, 0), (0, ±1)
# Selected via the low bits of hash2.
# =========================================================

@njit(fastmath=True, cache=True, inline='always')
def _grad2(h, dx, dy):
    """Map hash h to one of 8 gradient directions and dot with (dx, dy)."""
    h = h & np.int64(7)
    if h == 0:
        return  dx + dy
    elif h == 1:
        return -dx + dy
    elif h == 2:
        return  dx - dy
    elif h == 3:
        return -dx - dy
    elif h == 4:
        return  dx
    elif h == 5:
        return -dx
    elif h == 6:
        return  dy
    else:
        return -dy


@njit(fastmath=True, cache=True, inline='always')
def gradient_noise(x, y, seed):
    """
    Gradient (Perlin-style) noise.  Returns values in roughly [-1, 1].
    Replaces value_noise: no grid bias, better directional continuity.
    """
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))

    sx = fade(x - x0)
    sy = fade(y - y0)

    n00 = _grad2(hash2(x0,     y0,     seed), x - x0,       y - y0)
    n10 = _grad2(hash2(x0 + 1, y0,     seed), x - x0 - 1.0, y - y0)
    n01 = _grad2(hash2(x0,     y0 + 1, seed), x - x0,       y - y0 - 1.0)
    n11 = _grad2(hash2(x0 + 1, y0 + 1, seed), x - x0 - 1.0, y - y0 - 1.0)

    # scale to [-1, 1] (max dot product of unit-ish grad with corner offset ≈ √2)
    raw = lerp(lerp(n00, n10, sx), lerp(n01, n11, sx), sy)
    return raw * np.float64(0.7071)   # 1/√2 — normalise range


# Keep value_noise for backward-compat (domain_warp still compiles cleanly with either)
@njit(fastmath=True, cache=True, inline='always')
def value_noise(x, y, seed):
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))

    sx = fade(x - x0)
    sy = fade(y - y0)

    n00 = hash2(x0,     y0,     seed) * np.float32(2.3283064370807974e-10)
    n10 = hash2(x0 + 1, y0,     seed) * np.float32(2.3283064370807974e-10)
    n01 = hash2(x0,     y0 + 1, seed) * np.float32(2.3283064370807974e-10)
    n11 = hash2(x0 + 1, y0 + 1, seed) * np.float32(2.3283064370807974e-10)

    return lerp(lerp(n00, n10, sx), lerp(n01, n11, sx), sy) * 2.0 - 1.0


# =========================================================
# FAST FBM  – parallelise over pixels, not over octaves.
# This avoids re-entering the parallel region for every octave
# and is significantly more cache-friendly.
# =========================================================

@njit(parallel=True, fastmath=True, cache=True)
def fbm(X, Y, octaves, persistence, lacunarity, scale, seed):
    h, w = X.shape
    out = np.empty((h, w), dtype=np.float32)

    # precompute per-octave amplitudes / frequencies
    amps   = np.empty(octaves, dtype=np.float64)
    freqs  = np.empty(octaves, dtype=np.float64)
    amp_o, freq_o = 1.0, 1.0
    norm = 0.0
    for o in range(octaves):
        amps[o]  = amp_o
        freqs[o] = freq_o
        norm    += amp_o
        amp_o   *= persistence
        freq_o  *= lacunarity
    inv_norm = np.float32(1.0 / norm)

    for i in prange(h):
        for j in range(w):
            xi = X[i, j]
            yi = Y[i, j]
            val = np.float32(0.0)
            for o in range(octaves):
                val += np.float32(amps[o]) * np.float32(
                    gradient_noise(xi * scale * freqs[o],
                                   yi * scale * freqs[o],
                                   seed + o * 1013))
            out[i, j] = val * inv_norm
    return out


# =========================================================
# RIDGED MULTIFRACTAL  — sharp mountain ridges.
#
# ridge(x,y) = (1 - |noise|)²
# Each octave is ridge-shaped; successive octaves are weighted
# by the previous ridge value so ridges self-sharpen.
# =========================================================

@njit(parallel=True, fastmath=True, cache=True)
def ridged_fbm(X, Y, octaves, persistence, lacunarity, scale, seed):
    """
    Ridged multifractal noise.  Returns values in [0, 1] where 1 is a ridge.
    Dramatically better than standard fBm for mountain chains.
    """
    h, w = X.shape
    out = np.empty((h, w), dtype=np.float32)

    amps   = np.empty(octaves, dtype=np.float64)
    freqs  = np.empty(octaves, dtype=np.float64)
    amp_o, freq_o = 1.0, 1.0
    norm = 0.0
    for o in range(octaves):
        amps[o]  = amp_o
        freqs[o] = freq_o
        norm    += amp_o
        amp_o   *= persistence
        freq_o  *= lacunarity
    inv_norm = np.float32(1.0 / norm)

    for i in prange(h):
        for j in range(w):
            xi = X[i, j]
            yi = Y[i, j]
            val    = np.float32(0.0)
            weight = np.float32(1.0)     # inter-octave sharpening weight
            for o in range(octaves):
                n = np.float32(gradient_noise(
                    xi * scale * freqs[o],
                    yi * scale * freqs[o],
                    seed + o * 1013))
                # ridge transform: fold noise, square for sharper peaks
                r = np.float32(1.0) - math.fabs(n)
                r = r * r
                # weight emphasises ridges from the previous octave
                r    *= weight
                val  += np.float32(amps[o]) * r
                weight = r   # propagate for next octave
            out[i, j] = val * inv_norm
    return out


# =========================================================
# DOMAIN WARP
# =========================================================

@njit(parallel=True, fastmath=True, cache=True)
def domain_warp(X, Y, strength, octaves, persistence, lacunarity, scale, seed):
    wx = fbm(X,         Y,         octaves, persistence, lacunarity, scale, seed + 17)
    wy = fbm(X + 100.0, Y + 100.0, octaves, persistence, lacunarity, scale, seed + 31)

    h, w = X.shape
    for i in prange(h):
        for j in range(w):
            X[i, j] += strength * wx[i, j]
            Y[i, j] += strength * wy[i, j]

    return X, Y


# =========================================================
# THERMAL EROSION  – red-black (checkerboard) sweep so that
# prange is race-free: red cells only touch black neighbours
# and vice-versa.
# =========================================================

@njit(fastmath=True, cache=True)
def thermal_erosion(hmap, iterations, talus, strength):
    """
    Sequential thermal erosion.

    The parallel checkerboard trick sounds attractive but is NOT race-free:
    two parity-0 cells in the same column (e.g. (1,1) and (3,1)) share a
    parity-1 neighbour ((2,1)) and would write to it simultaneously.
    Running sequentially guarantees correctness and determinism, and
    thermal erosion is rarely the performance bottleneck (few iterations).
    """
    h, w = hmap.shape
    out = hmap.astype(np.float32)

    for _ in range(iterations):
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                c = out[i, j]
                if c - out[i + 1, j] > talus:
                    delta = strength * (c - out[i + 1, j])
                    out[i,     j] -= delta
                    out[i + 1, j] += delta
                    c = out[i, j]
                if c - out[i - 1, j] > talus:
                    delta = strength * (c - out[i - 1, j])
                    out[i,     j] -= delta
                    out[i - 1, j] += delta
                    c = out[i, j]
                if c - out[i, j + 1] > talus:
                    delta = strength * (c - out[i, j + 1])
                    out[i, j    ] -= delta
                    out[i, j + 1] += delta
                    c = out[i, j]
                if c - out[i, j - 1] > talus:
                    delta = strength * (c - out[i, j - 1])
                    out[i, j    ] -= delta
                    out[i, j - 1] += delta

    return out


# =========================================================
# HYDRAULIC EROSION (SIMPLE BUT FAST)
# =========================================================

@njit(fastmath=True, cache=True)
def hydraulic_erosion(hmap, iterations, rain, evap, erode, deposit):
    """Legacy grid-flow hydraulic erosion (kept for reference)."""
    h, w = hmap.shape

    water    = np.zeros((h, w), dtype=np.float32)
    sediment = np.zeros((h, w), dtype=np.float32)
    out      = hmap.astype(np.float32)

    for _ in range(iterations):

        for i in range(h):
            for j in range(w):
                water[i, j] += rain

        for i in range(1, h - 1):
            for j in range(1, w - 1):

                h0 = out[i, j] + water[i, j]

                lowest = h0
                li, lj = i, j

                for di, dj in ((1,0), (-1,0), (0,1), (0,-1)):
                    ni = i + di
                    nj = j + dj
                    nh = out[ni, nj] + water[ni, nj]

                    if nh < lowest:
                        lowest = nh
                        li, lj = ni, nj

                if li != i or lj != j:
                    dh       = h0 - lowest
                    capacity = dh * 0.5 * erode

                    if sediment[i, j] < capacity:
                        delta          = capacity - sediment[i, j]
                        out[i, j]     -= delta
                        sediment[i, j] += delta
                    else:
                        delta          = (sediment[i, j] - capacity) * deposit
                        out[i, j]     += delta
                        sediment[i, j] -= delta

                water[i, j] *= (1.0 - evap)

    return out


# =========================================================
# BILINEAR HELPERS used by particle erosion
# Defined at module level so they are compiled once and
# inlined, rather than re-created inside prange every step.
# =========================================================

@njit(fastmath=True, cache=True, inline='always')
def _bilinear_height(out, bx, by, w, h):
    bx0 = int(bx);      by0 = int(by)
    bx1 = min(bx0 + 1, w - 1)
    by1 = min(by0 + 1, h - 1)
    bx0 = max(bx0, 0);  by0 = max(by0, 0)
    # Use the *clamped* integer corners for the fractional part so that the
    # weights are consistent with the actual corners being sampled.
    fx  = bx - np.float32(bx0)
    fy  = by - np.float32(by0)
    return (out[by0, bx0] * (np.float32(1.0) - fx) * (np.float32(1.0) - fy) +
            out[by0, bx1] * fx * (np.float32(1.0) - fy) +
            out[by1, bx0] * (np.float32(1.0) - fx) * fy +
            out[by1, bx1] * fx * fy)


@njit(fastmath=True, cache=True, inline='always')
def _bilinear_deposit(out, dx, dy, amount, w, h):
    dx0 = int(dx);      dy0 = int(dy)
    dx1 = min(dx0 + 1, w - 1)
    dy1 = min(dy0 + 1, h - 1)
    dx0 = max(dx0, 0);  dy0 = max(dy0, 0)
    # Use clamped corners for fractional part (matches _bilinear_height).
    fx  = dx - np.float32(dx0)
    fy  = dy - np.float32(dy0)
    out[dy0, dx0] += amount * (np.float32(1.0) - fx) * (np.float32(1.0) - fy)
    out[dy0, dx1] += amount * fx * (np.float32(1.0) - fy)
    out[dy1, dx0] += amount * (np.float32(1.0) - fx) * fy
    out[dy1, dx1] += amount * fx * fy


# =========================================================
# PARTICLE-BASED HYDRAULIC EROSION
# =========================================================
# Optimisations vs the original:
#   • float32 throughout (2× memory-bandwidth vs float64)
#   • prange over droplets – each is independent so embarrassingly
#     parallel; small write races on the hmap are statistically
#     negligible for erosion (same technique used in most GPU
#     erosion papers).
#   • All bilinear helpers are fully inlined (no call overhead).
#   • Gradient uses a single 4-sample cross instead of 8 calls.
#   • LCG seed offset per thread avoids correlation artefacts.
# =========================================================

@njit(parallel=True, fastmath=True, cache=True)
def particle_hydraulic_erosion(
    hmap,
    num_droplets   = 200_000,
    max_path       = 256,
    inertia        = 0.4,
    capacity_factor = 8.0,
    min_slope      = 0.002,
    erode_speed    = 0.3,
    deposit_speed  = 0.3,
    evaporate_speed = 0.02,
    gravity        = 10.0,
    initial_water  = 1.0,
    initial_speed  = 1.0,
    seed           = 42,
):
    """
    Particle (droplet) hydraulic erosion – parallelised over droplets.

    Parameters
    ----------
    num_droplets      : total droplets to simulate
    max_path          : maximum steps per droplet before it is killed
    inertia           : [0-1] momentum weight vs gradient
    capacity_factor   : scales sediment capacity
    min_slope         : minimum effective slope (prevents flat-area over-erosion)
    erode_speed       : fraction of sediment deficit eroded each step
    deposit_speed     : fraction of sediment surplus deposited each step
    evaporate_speed   : fraction of water lost each step
    gravity           : multiplies speed update
    initial_water     : starting water volume per droplet
    initial_speed     : starting speed per droplet
    seed              : RNG seed
    """
    h, w = hmap.shape
    # Work on float32 copy shared across threads.
    # Write races are rare and their effect averages out over many droplets.
    out = hmap.astype(np.float32)

    # constants cast once
    f_inertia   = np.float32(inertia)
    f_1mi       = np.float32(1.0 - inertia)
    f_grav      = np.float32(gravity)
    f_cap       = np.float32(capacity_factor)
    f_mslope    = np.float32(min_slope)
    f_erode     = np.float32(erode_speed)
    f_deposit   = np.float32(deposit_speed)
    f_evap      = np.float32(1.0 - evaporate_speed)
    f_iw        = np.float32(initial_water)
    f_is        = np.float32(initial_speed)

    LCG_A = np.uint64(6364136223846793005)
    LCG_C = np.uint64(1442695040888963407)

    for d in prange(num_droplets):
        # Per-droplet independent LCG state seeded from (seed, d)
        rng = np.uint64(seed) * LCG_A + LCG_C
        rng = rng ^ (np.uint64(d) * np.uint64(2654435761))
        rng = rng * LCG_A + LCG_C

        px = np.float32((rng >> np.uint64(33)) % np.uint64(w - 2)) + np.float32(0.5)
        rng = rng * LCG_A + LCG_C
        py = np.float32((rng >> np.uint64(33)) % np.uint64(h - 2)) + np.float32(0.5)

        vx       = np.float32(0.0)
        vy       = np.float32(0.0)
        water    = f_iw
        speed    = f_is
        sediment = np.float32(0.0)

        for _ in range(max_path):
            # ---- bilinear gradient (4-sample cross, eps=0.5) ----
            eps = np.float32(0.5)

            gx = (_bilinear_height(out, px + eps, py,       w, h) -
                  _bilinear_height(out, px - eps, py,       w, h))
            gy = (_bilinear_height(out, px,       py + eps, w, h) -
                  _bilinear_height(out, px,       py - eps, w, h))

            # update velocity
            vx = vx * f_inertia - gx * f_1mi
            vy = vy * f_inertia - gy * f_1mi

            vmag = math.sqrt(vx * vx + vy * vy)
            if vmag < np.float32(1e-8):
                break

            # vmag IS the current physical speed (pixels/step).
            # Normalise direction for the step, keep vmag as speed.
            speed    = vmag
            inv_vmag = np.float32(1.0) / vmag
            vx *= inv_vmag
            vy *= inv_vmag

            nx = px + vx
            ny = py + vy

            if nx < 0.0 or nx >= w - 1 or ny < 0.0 or ny >= h - 1:
                break

            old_h = _bilinear_height(out, px, py, w, h)
            new_h = _bilinear_height(out, nx, ny, w, h)
            dh    = new_h - old_h

            slope    = max(-dh, f_mslope)
            capacity = slope * speed * water * f_cap

            if sediment > capacity or dh > np.float32(0.0):
                if dh <= np.float32(0.0):
                    deposit_amt = (sediment - capacity) * f_deposit
                else:
                    deposit_amt = min(sediment, -dh)
                if deposit_amt > np.float32(0.0):
                    sediment -= deposit_amt
                    _bilinear_deposit(out, px, py,  deposit_amt, w, h)
            else:
                erode_amt = min((capacity - sediment) * f_erode,
                                -dh if dh < np.float32(0.0) else np.float32(0.05) * capacity)
                if erode_amt > np.float32(0.0):
                    sediment += erode_amt
                    _bilinear_deposit(out, px, py, -erode_amt, w, h)

            # Accelerate / decelerate with gravity based on actual height change.
            # Clamp to avoid imaginary speeds when depositing on uphill.
            vx *= math.sqrt(max(np.float32(1.0) + (-dh) * f_grav / max(speed * speed, np.float32(1e-8)), np.float32(0.1)))
            vy *= math.sqrt(max(np.float32(1.0) + (-dh) * f_grav / max(speed * speed, np.float32(1e-8)), np.float32(0.1)))

            water *= f_evap
            if water < np.float32(0.001):
                break

            px, py = nx, ny

        if sediment > np.float32(0.0):
            dx0 = max(0, min(int(px),     w - 1))
            dy0 = max(0, min(int(py),     h - 1))
            out[dy0, dx0] += sediment

    return out


# =========================================================
# DRAINAGE-FLOW ACCUMULATION + RIVER CARVING
#
# Algorithm:
#   1. D8 steepest-descent: each cell points to its lowest neighbour.
#   2. Sort cells high→low; propagate unit water downstream.
#   3. Carve the height map proportional to accumulated flow and slope.
#
# This creates coherent drainage basins, dendritic river networks,
# V-shaped valleys and sediment fans — none of which particle erosion
# alone produces.
# =========================================================

@njit(fastmath=True, cache=True)
def _d8_flow_dir(hmap):
    """
    Compute D8 steepest-descent flow direction.
    Returns (ri, rj): the row / col *offset* to the receiving cell.
    (0, 0) means the cell is a local sink (no lower neighbour).
    Diagonal moves are distance-corrected (÷√2) so steepness is comparable.
    """
    h, w = hmap.shape
    dir_i = np.zeros((h, w), dtype=np.int8)
    dir_j = np.zeros((h, w), dtype=np.int8)

    inv_sqrt2 = np.float32(0.7071067811865476)

    for i in range(h):
        for j in range(w):
            c          = hmap[i, j]
            best_drop  = np.float32(0.0)
            best_di    = np.int8(0)
            best_dj    = np.int8(0)

            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if di == 0 and dj == 0:
                        continue
                    ni = i + di
                    nj = j + dj
                    if ni < 0 or ni >= h or nj < 0 or nj >= w:
                        continue
                    drop = c - hmap[ni, nj]
                    if di != 0 and dj != 0:   # diagonal — correct for distance
                        drop *= inv_sqrt2
                    if drop > best_drop:
                        best_drop = drop
                        best_di   = np.int8(di)
                        best_dj   = np.int8(dj)

            dir_i[i, j] = best_di
            dir_j[i, j] = best_dj

    return dir_i, dir_j


@njit(fastmath=True, cache=True)
def _accumulate_flow(dir_i, dir_j, order, h, w):
    """
    Propagate unit rainfall downstream using a pre-sorted high→low order.
    `order` is a flat index array (np.int64) sorted by descending elevation.
    """
    accum = np.ones((h, w), dtype=np.float32)

    for idx in range(len(order)):
        flat = order[idx]
        i    = flat // w
        j    = flat %  w
        di   = int(dir_i[i, j])
        dj   = int(dir_j[i, j])
        if di == 0 and dj == 0:
            continue
        ni = i + di
        nj = j + dj
        if 0 <= ni < h and 0 <= nj < w:
            accum[ni, nj] += accum[i, j]

    return accum


def flow_accumulation(hmap):
    """
    Pure-Python wrapper: computes D8 flow directions then accumulates.
    Returns a float32 array of the same shape; values are pixel counts
    (1 = headwater, large = major river).
    """
    dir_i, dir_j = _d8_flow_dir(hmap)
    order = np.argsort(-hmap.flatten()).astype(np.int64)
    return _accumulate_flow(dir_i, dir_j, order, hmap.shape[0], hmap.shape[1])


# =========================================================
# ANISOTROPIC RIDGED MULTIFRACTAL
#
# Stretches coordinates along / across a dominant ridge axis so that
# ridges form coherent chains (Andes, Himalayas, Appalachians) instead
# of the isotropic blob-peaks produced by standard ridged_fbm.
#
# `angle`      – ridge strike in radians (0 = ridges run E-W).
# `anisotropy` – how much the ridge chains are elongated (1 = isotropic).
# =========================================================

@njit(parallel=True, fastmath=True, cache=True)
def anisotropic_ridged_fbm(X, Y, octaves, persistence, lacunarity, scale, seed,
                            angle=0.0, anisotropy=3.0):
    h, w = X.shape
    out  = np.empty((h, w), dtype=np.float32)

    cos_a = np.float64(math.cos(angle))
    sin_a = np.float64(math.sin(angle))

    amps  = np.empty(octaves, dtype=np.float64)
    freqs = np.empty(octaves, dtype=np.float64)
    amp_o, freq_o = 1.0, 1.0
    norm = 0.0
    for o in range(octaves):
        amps[o]  = amp_o
        freqs[o] = freq_o
        norm    += amp_o
        amp_o   *= persistence
        freq_o  *= lacunarity
    inv_norm = np.float32(1.0 / norm)
    f_aniso  = np.float64(anisotropy)

    for i in prange(h):
        for j in range(w):
            xi = np.float64(X[i, j])
            yi = np.float64(Y[i, j])
            # Rotate into ridge-aligned frame
            xr =  xi * cos_a + yi * sin_a   # along-ridge axis → compressed
            yr = -xi * sin_a + yi * cos_a   # cross-ridge axis → kept
            xs = xr / f_aniso               # elongate chains in strike direction
            ys = yr

            val    = np.float32(0.0)
            weight = np.float32(1.0)
            for o in range(octaves):
                n = np.float32(gradient_noise(
                    xs * scale * freqs[o],
                    ys * scale * freqs[o],
                    seed + o * 1013))
                r = np.float32(1.0) - math.fabs(n)
                r = r * r
                r   *= weight
                val += np.float32(amps[o]) * r
                weight = r
            out[i, j] = val * inv_norm
    return out


# =========================================================
# RAIN SHADOW  (orographic precipitation model)
#
# Moisture advects along the prevailing wind direction and is depleted
# whenever terrain rises (orographic lift), producing wet windward
# slopes and dry leeward deserts.
#
# Returns a precipitation map in [0, 1]:  1 = maximum rain / snow,
# 0 = completely in rain shadow.
# =========================================================

def rain_shadow(terrain, wind_angle_deg=270.0, num_passes=40, lift_factor=2.5):
    """
    Parameters
    ----------
    wind_angle_deg : direction the wind blows FROM, degrees clockwise from North.
                     270° = westerlies (blowing eastward).
    num_passes     : upwind-pixel look-back distance (= shadow reach in pixels).
    lift_factor    : rate at which rising terrain depletes moisture; larger →
                     sharper, drier shadows behind mountain ranges.
    """
    from scipy.ndimage import shift as _shift
    import math as _m

    rad      = _m.radians(wind_angle_deg)
    # image: col = +x (east), row = +y (south → downward).
    step_col =  _m.cos(rad)
    step_row = -_m.sin(rad)   # north is up ⟹ negate y

    h, w     = terrain.shape
    moisture = np.ones((h, w), dtype=np.float64)
    t        = terrain.astype(np.float64)

    for _ in range(num_passes):
        upwind = _shift(t, (step_row, step_col), mode='nearest')
        rise   = np.maximum(0.0, t - upwind)
        moisture *= np.exp(-lift_factor * rise)
        np.clip(moisture, 0.0, 1.0, out=moisture)

    return moisture.astype(np.float32)


# =========================================================
# VALLEY WIDENING  (lateral river erosion)
#
# After sharp incision from carve_rivers, blur the carving outward to
# create V-shaped valleys, broad floodplains, and canyon flanks.
# =========================================================

def widen_valleys(hmap, accum, threshold, valley_sigma=2.0, max_depth=0.025):
    """
    Parameters
    ----------
    valley_sigma : Gaussian σ in pixels controlling lateral valley width.
    max_depth    : extra lowering applied at the river centre-line.
    """
    from scipy.ndimage import gaussian_filter as _gf

    river  = np.where(accum > threshold,
                      np.log1p(np.maximum(0.0, accum - threshold)), 0.0).astype(np.float32)
    spread = _gf(river, sigma=valley_sigma)
    mx     = spread.max()
    if mx > 0:
        spread /= mx
    return (hmap - max_depth * spread).astype(np.float32)


# =========================================================
# ALLUVIAL DEPOSITION
#
# Add sediment in low-lying areas downstream of major rivers,
# producing alluvial fans, deltas, and valley fills.
# =========================================================

@njit(fastmath=True, cache=True)
def alluvial_deposition(hmap, accum, threshold,
                        deposit_amount=0.025, lowland_cutoff=0.35):
    """
    Parameters
    ----------
    deposit_amount : maximum height added at the most depositional cell.
    lowland_cutoff : normalised-elevation ceiling above which no deposition
                     occurs (0 = only in valleys, 1 = everywhere below ridges).
    """
    h, w  = hmap.shape
    out   = hmap.astype(np.float32)
    mn    = np.float32(hmap.min())
    rng   = np.float32(hmap.max()) - mn + np.float32(1e-8)

    max_a = np.float32(1.0)
    for i in range(h):
        for j in range(w):
            if accum[i, j] > max_a:
                max_a = np.float32(accum[i, j])

    inv_max = np.float32(1.0) / max_a
    f_thr   = np.float32(threshold)
    f_dep   = np.float32(deposit_amount)
    f_cut   = np.float32(lowland_cutoff)

    for i in range(h):
        for j in range(w):
            a = np.float32(accum[i, j])
            if a > f_thr:
                norm_elev = (out[i, j] - mn) / rng   # 0 = low, 1 = high
                if norm_elev < f_cut:
                    flow_str   = ((a - f_thr) * inv_max) ** np.float32(0.5)
                    dep_factor = (f_cut - norm_elev) / f_cut
                    out[i, j] += f_dep * flow_str * dep_factor

    return out


@njit(fastmath=True, cache=True)
def carve_rivers(hmap, accum,
                 threshold   = 500.0,
                 max_depth   = 0.08,
                 slope_power = 0.45):
    """
    Lower terrain proportional to sqrt(flow) for cells above `threshold`.

    Parameters
    ----------
    threshold   : minimum flow-accumulation to start carving (pixels of catchment).
    max_depth   : maximum height units carved at the largest river.
    slope_power : exponent on the normalised flow — 0.5 gives √-scaling,
                  matching the empirical width–discharge relationship.
    """
    h, w       = hmap.shape
    out        = hmap.astype(np.float32)
    max_accum  = np.float32(0.0)

    for i in range(h):
        for j in range(w):
            if accum[i, j] > max_accum:
                max_accum = accum[i, j]

    if max_accum <= threshold:
        return out

    inv_max = np.float32(1.0) / max_accum
    f_thr   = np.float32(threshold)
    f_depth = np.float32(max_depth)

    for i in range(h):
        for j in range(w):
            a = np.float32(accum[i, j])
            if a > f_thr:
                strength    = ((a - f_thr) * inv_max) ** np.float32(slope_power)
                out[i, j]  -= f_depth * strength

    return out