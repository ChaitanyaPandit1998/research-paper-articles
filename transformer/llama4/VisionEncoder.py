import torch
import torch.nn as nn
import math

# ---------------------------------------------------------------------------
# Llama4VisionRotaryEmbedding — 2D RoPE for vision patches
# ---------------------------------------------------------------------------
# This module computes the rotary position frequencies for a 32×32 patch grid.
# Unlike text RoPE (1D), vision patches have both an x (column) and y (row)
# coordinate, so we compute separate frequency tables for each axis and combine.
# ---------------------------------------------------------------------------

rope_theta = 10000  # standard RoPE base

class Llama4VisionRotaryEmbedding(nn.Module):

    @staticmethod
    def _compute_freqs_ci(config):

        # ── Step 1 ──────────────────────────────────────────────────────────
        # Compute the grid side length.
        # Input:  image_size=448, patch_size=14
        # Output: idx = 32  (integer scalar)
        # Meaning: the image is a 32×32 grid of patches
        idx = config.image_size // config.patch_size
        # idx = 32

        # ── Step 2 ──────────────────────────────────────────────────────────
        # Create flat indices for all 1024 patches, as a column vector.
        # Input:  idx = 32
        # Output: img_idx, shape [1024, 1]
        #
        # arange(32²) → [0, 1, 2, ..., 1023]   (1D, 1024 elements)
        # reshape →     [[0],
        #                [1],
        #                ...
        #                [1023]]                (2D column vector, [1024, 1])
        #
        # Column vector shape is intentional: allows broadcasting against
        # rope_freq [12] later to produce [1024, 12] without extra unsqueezing.
        img_idx = torch.arange(idx**2).reshape(idx**2, 1)
        # img_idx.shape = [1024, 1]

        # ── Step 3 ──────────────────────────────────────────────────────────
        # Append one extra row as a placeholder for the CLS token.
        # Input:  img_idx [1024, 1]
        # Output: img_idx [1025, 1]
        #
        # img_idx[:1] → [[0]]  (first row, shape [1, 1])
        # cat along dim=0 appends it at the bottom — value doesn't matter yet,
        # next line overwrites it with the CLS sentinel.
        img_idx = torch.cat([img_idx, img_idx[:1]], dim=0)
        # img_idx.shape = [1025, 1]
        # img_idx[-1] = [[0]]  ← placeholder, about to be overwritten

        # ── Step 4 ──────────────────────────────────────────────────────────
        # Mark the CLS row with sentinel value -2.
        # Input:  img_idx[-1, -1] = 0   (placeholder)
        # Output: img_idx[-1, -1] = -2
        #
        # -2 is negative → triggers (img_idx < 0) in masked_fill later
        # → CLS frequencies will be zeroed out (no spatial position for CLS)
        img_idx[-1, -1] = -2
        # img_idx[-1] = [[-2]]

        # ── Step 5 ──────────────────────────────────────────────────────────
        # Recover the x-coordinate (column) of each patch from its flat index.
        # Input:  img_idx [1025, 1],  idx = 32
        # Output: frequencies_x [1025, 1], values in range [0..31] (and -2 for CLS)
        #
        # Flat index i maps to column:  i % 32
        #   patch 0  → col 0
        #   patch 1  → col 1
        #   patch 31 → col 31
        #   patch 32 → col 0   (second row starts)
        #   patch 63 → col 31
        #
        # CLS: -2 % 32 = 30 in Python — junk value, zeroed out later by masked_fill
        frequencies_x = img_idx % idx
        # frequencies_x.shape = [1025, 1]

        # ── Step 6 ──────────────────────────────────────────────────────────
        # Recover the y-coordinate (row) of each patch from its flat index.
        # Input:  img_idx [1025, 1],  idx = 32
        # Output: frequencies_y [1025, 1], values in range [0..31]
        #
        # Flat index i maps to row:  i // 32
        #   patches  0– 31 → row 0  (top row)
        #   patches 32– 63 → row 1
        #   patches 992–1023 → row 31  (bottom row)
        frequencies_y = img_idx // idx
        # frequencies_y.shape = [1025, 1]

        # ── Step 7 ──────────────────────────────────────────────────────────
        # Compute how many frequency components each spatial axis gets.
        # Input:  hidden_size=768, num_attention_heads=16
        # Output: freq_dim = 24  (integer scalar)
        #
        # 768 // 16 = 48   → head dimension (each head operates on 48 values)
        # 48  // 2  = 24   → split evenly: 24 dims for x, 24 dims for y
        #
        # RoPE rotates pairs of dimensions. With 48 dims and 2 spatial axes:
        # half the pairs encode x-position, half encode y-position.
        freq_dim = config.hidden_size // config.num_attention_heads // 2
        # freq_dim = 24

        # ── Step 8 ──────────────────────────────────────────────────────────
        # Compute 12 base frequencies (the RoPE frequency ladder).
        # Input:  freq_dim = 24,  rope_theta = 10000
        # Output: rope_freq [12], values decreasing from 1.0 → ~0.000133
        #
        # arange(0, 24, 2)   → [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]  (12 values)
        # [:12]              → same (slice is a no-op here)
        # / 24               → [0.0, 0.083, 0.167, ..., 0.917]
        # 10000 ** [...]     → [1.0, 1.96, 3.83, ..., 7499]    (increasing denominators)
        # 1.0 / [...]        → [1.0, 0.51, 0.26, ..., 0.000133] (decreasing frequencies)
        #
        # High frequency (≈1.0) → rotates fast → distinguishes nearby positions precisely
        # Low frequency (≈0.0001) → rotates slowly → captures coarse/global position
        rope_freq = 1.0 / (rope_theta ** (torch.arange(0, freq_dim, 2)[:freq_dim//2] / freq_dim))
        # rope_freq.shape = [12]
        # rope_freq ≈ [1.0, 0.51, 0.26, 0.133, 0.068, 0.035, 0.018, 0.009, 0.005, 0.002, 0.001, 0.000133]

        # ── Step 9 ──────────────────────────────────────────────────────────
        # Compute x-position frequencies for all patches.
        # Input:  frequencies_x [1025, 1],  rope_freq [12]
        # Output: freqs_x [1025, 24]
        #
        # (frequencies_x + 1):
        #   Shift x from 0-based to 1-based (col 0 → 1, col 1 → 2, ..., col 31 → 32)
        #   WHY: without +1, the first column (x=0) gives 0 × rope_freq = [0,0,...,0]
        #        → sin(0)=0 and cos(0)=1 for all dims → identity rotation → no position info
        #        With +1, every column gets a unique, non-zero encoding.
        #
        # * rope_freq:
        #   [1025, 1] × [12] → [1025, 12]  (broadcast)
        #   Row i: (x_i + 1) × [f0, f1, ..., f11]
        #   = the rotation angle for each of the 12 frequency pairs, for patch i's x-position
        #
        # .repeat_interleave(2, dim=-1):
        #   [1025, 12] → [1025, 24]
        #   [a, b, c, ...] → [a, a, b, b, c, c, ...]
        #   Each frequency is duplicated because RoPE applies one frequency to a *pair*
        #   of dimensions (the real and imaginary parts of a complex rotation).
        freqs_x = ((frequencies_x + 1) * rope_freq).repeat_interleave(2, dim=-1)
        # freqs_x.shape = [1025, 24]
        # freqs_x[0] = [1×f0, 1×f0, 1×f1, 1×f1, ..., 1×f11, 1×f11]  ← col 0 patch
        # freqs_x[1] = [2×f0, 2×f0, 2×f1, 2×f1, ..., 2×f11, 2×f11]  ← col 1 patch

        # ── Step 10 ─────────────────────────────────────────────────────────
        # Same computation for y-position.
        # Input:  frequencies_y [1025, 1],  rope_freq [12]
        # Output: freqs_y [1025, 24]
        #
        # Identical logic to Step 9, using row (y) instead of column (x).
        freqs_y = ((frequencies_y + 1) * rope_freq).repeat_interleave(2, dim=-1)
        # freqs_y.shape = [1025, 24]
        # freqs_y[0]  = [1×f0, 1×f0, ..., 1×f11, 1×f11]  ← row 0 (patches 0–31 all share y=0)
        # freqs_y[32] = [2×f0, 2×f0, ..., 2×f11, 2×f11]  ← row 1 (patches 32–63)

        # ── Step 11 ─────────────────────────────────────────────────────────
        # Combine x and y frequencies, then remove duplicates.
        # Input:  freqs_x [1025, 24],  freqs_y [1025, 24]
        # Output: freqs [1025, 24]
        #
        # cat([freqs_x, freqs_y], dim=-1):
        #   [1025, 24] + [1025, 24] → [1025, 48]
        #   First 24 cols = x freqs (with duplicates): [x_f0, x_f0, x_f1, x_f1, ..., x_f11, x_f11]
        #   Last  24 cols = y freqs (with duplicates): [y_f0, y_f0, y_f1, y_f1, ..., y_f11, y_f11]
        #
        # .float(): ensure float32 (required by view_as_complex later)
        #
        # .contiguous(): ensure memory is laid out contiguously (required by view_as_complex)
        #
        # [..., ::2]: take every other element along the last dim
        #   [1025, 48] → [1025, 24]
        #   Picks indices 0, 2, 4, ..., 46
        #   → strips the duplicates introduced by repeat_interleave
        #   Result: [x_f0, x_f1, ..., x_f11, y_f0, y_f1, ..., y_f11]
        #            ←── 12 x-freqs ───→  ←── 12 y-freqs ───→
        freqs = torch.cat([freqs_x, freqs_y], dim=-1).float().contiguous()[..., ::2]
        # freqs.shape = [1025, 24]
        # Each row: 12 x-position frequencies + 12 y-position frequencies for that patch

        # ── Step 12 ─────────────────────────────────────────────────────────
        # Zero out frequencies for the CLS token.
        # Input:  freqs [1025, 24],  img_idx [1025, 1]
        # Output: freqs [1025, 24]  (same shape, CLS row set to 0)
        #
        # img_idx.reshape(-1, 1, 1) → [1025, 1, 1]
        # < 0                       → True only for CLS row (img_idx = -2), False for all patches
        # masked_fill(..., 0)       → sets freqs to 0.0 wherever mask is True
        #
        # Effect on CLS row: all 24 frequency values → 0.0
        # cos(0) = 1,  sin(0) = 0  →  complex number (1 + 0j)  →  identity rotation
        # CLS gets no positional bias — it has no spatial location in the image grid.
        freqs = freqs.masked_fill(img_idx.reshape(-1, 1, 1) < 0, 0)
        # freqs.shape = [1025, 24]
        # freqs[-1] = [0.0, 0.0, ..., 0.0]  ← CLS: all zeros
        # freqs[0]  = [x_f0, x_f1, ..., y_f11]  ← patches: unchanged

        # ── Step 13 ─────────────────────────────────────────────────────────
        # Convert frequency angles to complex unit-circle rotations (e^iθ).
        # Input:  freqs [1025, 24]   (real-valued rotation angles)
        # Output: freq_cis           (complex-valued, each entry = cos θ + i·sin θ)
        #
        # torch.cos(freqs), torch.sin(freqs):
        #   Element-wise cos and sin of every angle in freqs
        #   Each output: same shape as freqs
        #
        # torch.stack([cos, sin], dim=-1):
        #   Pairs each cosine with its sine along a new last dimension
        #   Output: [..., 2]
        #   Each pair (cos θ, sin θ) represents a point on the unit circle
        #
        # torch.view_as_complex(...):
        #   Interprets each (cos θ, sin θ) pair as a complex number: cos θ + i·sin θ
        #   This is exactly e^(iθ) — Euler's formula
        #   The last dimension (size 2) collapses into one complex value
        #
        # WHY complex? RoPE applies positional encoding by rotating query and key vectors.
        # Multiplying a complex vector by e^(iθ) rotates it by angle θ.
        # Two patches far apart in x or y will have very different θ values
        # → their Q·K dot products naturally decrease with spatial distance
        # → attention becomes position-aware without any extra learned parameters.
        freq_cis = torch.view_as_complex(
            torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)
        )
        # freq_cis: complex tensor
        # Each entry: e^(iθ) where θ is the rotation angle for that patch × frequency pair
        # freq_cis[-1] = (1 + 0j) for all entries  ← CLS: identity rotation (no position)

        return freq_cis


# ---------------------------------------------------------------------------
# Shape journey summary
# ---------------------------------------------------------------------------
#
#   Step  | Variable         | Shape        | What each element means
#   ────────────────────────────────────────────────────────────────────────
#   2     | img_idx          | [1024, 1]    | flat patch index (0..1023)
#   3-4   | img_idx          | [1025, 1]    | + CLS row marked -2
#   5     | frequencies_x    | [1025, 1]    | column index (0..31) per patch
#   6     | frequencies_y    | [1025, 1]    | row index (0..31) per patch
#   8     | rope_freq        | [12]         | 12 base frequencies (geometric ladder)
#   9     | freqs_x          | [1025, 24]   | x rotation angles × 12 freqs (duplicated)
#   10    | freqs_y          | [1025, 24]   | y rotation angles × 12 freqs (duplicated)
#   11    | freqs (after cat)| [1025, 48]   | x + y freqs concatenated (with duplicates)
#   11    | freqs (after ::2)| [1025, 24]   | de-duplicated: 12 x-freqs + 12 y-freqs
#   12    | freqs            | [1025, 24]   | CLS row zeroed out
#   13    | freq_cis         | [1025, 12]*  | complex rotations e^(iθ) per patch × freq
#
#   * view_as_complex collapses the last dim-2 into one complex value,
#     so [1025, 24, 2] → [1025, 24] complex (or shaped depending on freqs ndim)
#
# ---------------------------------------------------------------------------
