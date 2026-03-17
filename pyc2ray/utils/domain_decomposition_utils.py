import math
from typing import List, Tuple, Optional
import numpy as np

class VariableResolutionGrid:
    """Prototype variable-resolution grid model.

    Attributes
    ----------
    domain_min : np.ndarray
        Minimum domain corner (shape `(3,)`).
    domain_max : np.ndarray
        Maximum domain corner (shape `(3,)`).
    patches : List[Tuple[np.ndarray, np.ndarray, float]]
        Rectangular subdomains described as `(pmin, pmax, dx)`, where `dx`
        is the local cell size in that patch.
    """
    domain_min: np.ndarray
    domain_max: np.ndarray
    patches: List[Tuple[np.ndarray, np.ndarray, float]]  # (pmin, pmax, dx)

    def __init__(
        self,
        domain_min: np.ndarray,
        domain_max: np.ndarray,
        patches: List[Tuple[np.ndarray, np.ndarray, float]],
    ) -> None:
        self.domain_min = np.asarray(domain_min, dtype=float)
        self.domain_max = np.asarray(domain_max, dtype=float)
        self.patches = [
            (np.asarray(pmin, dtype=float), np.asarray(pmax, dtype=float), float(dx))
            for pmin, pmax, dx in patches
        ]

    def overlap_volume(self, a_min, a_max, b_min, b_max) -> float:
        """Return intersection volume between two axis-aligned boxes."""
        lo = np.maximum(a_min, b_min)
        hi = np.minimum(a_max, b_max)
        d = np.maximum(0.0, hi - lo)
        return float(d[0] * d[1] * d[2])

    def find_voxels_in_bbox(self, bbox_min: np.ndarray, bbox_max: np.ndarray) -> List[Tuple[int, np.ndarray, np.ndarray, int]]:
        """Estimate voxel count and indexes in a bbox for the list of patches."""
        indexes = []
        for i, (pmin, pmax, dx) in enumerate(self.patches):
            vol = self.overlap_volume(bbox_min, bbox_max, pmin, pmax)
            if vol > 0.0:
                # TODO CB: I'm not sure it make sense in cost evaluation: we should maybe only consider
                # the most refined patch overlapping the bbox. I'm not sure of the volume normalization either.
                indexes.append((i, np.floor(bbox_min/dx).astype(int), np.ceil(bbox_max/dx).astype(int), np.floor(vol / (dx**3)).astype(int)))

        return indexes

class Source:
    """Represents a single radiation source.

    Attributes
    ----------
    gid : int
        Global source identifier.
    pos : np.ndarray
        Source position in 3D Cartesian coordinates (shape `(3,)`).
    strength : float
        Source emissivity/intensity used by downstream physics models.
    radius : float
        Maximum tracing/influence radius for this source.
    """
    def __init__(
        self,
        gid: int,
        pos: np.ndarray,
        strength: float,
        radius: float,
    ) -> None:
        self.gid = gid
        self.pos = pos
        self.strength = strength
        self.radius = radius


class Group:
    """Container for a set of sources grouped for joint processing.

    Attributes
    ----------
    sources : List[Source]
        Sources that belong to this group.
    center : np.ndarray
        Center of the enclosing sphere of grouped source spheres (shape `(3,)`).
    radius : float
        Radius of the enclosing sphere.
    bbox_min : np.ndarray
        Minimum corner of the axis-aligned bounding box around the group.
    bbox_max : np.ndarray
        Maximum corner of the axis-aligned bounding box around the group.
    voxels : List[Tuple[int, np.ndarray, np.ndarray, int]]
        List of overlapping grid patches and their voxel min/max indexes.
    cost : float
        Estimated computational cost for this group.
    """
    def __init__(
        self,
        sources: Optional[List[Source]] = None,
        center: Optional[np.ndarray] = None,
        radius: float = 0.0,
        bbox_min: Optional[np.ndarray] = None,
        bbox_max: Optional[np.ndarray] = None,
        voxels: Optional[List[Tuple[int, np.ndarray, np.ndarray, int]]] = None,
        cost: float = 0.0,
    ) -> None:
        self.sources = list(sources) if sources is not None else []
        self.center = center.copy() if center is not None else np.zeros(3)
        self.radius = radius  # enclosing sphere radius for union of balls
        self.bbox_min = bbox_min.copy() if bbox_min is not None else np.zeros(3)
        self.bbox_max = bbox_max.copy() if bbox_max is not None else np.zeros(3)
        self.cost = cost
        self.voxels = voxels if voxels is not None else []

    def get_source_ids(self) -> List[Optional[int]]:
        """Return the list of source global IDs in this group."""
        return [s.gid for s in self.sources]

    def get_num_voxels(self) -> List[Optional[int]]:
        """Return the total number of voxels covered by this group."""
        num_voxels_per_patch = []
        for voxel in self.voxels:
            num_voxels_per_patch.append(int(np.prod(voxel[2] - voxel[1])))
        return num_voxels_per_patch

    def get_num_cells_per_side(self) -> int:
        """Return cubic side length (number of cells) for the top/first patch."""
        if not self.voxels:
            return 0

        vmin = np.asarray(self.voxels[0][1], dtype=int)
        vmax = np.asarray(self.voxels[0][2], dtype=int)
        nside = vmax - vmin
        if not (nside[0] == nside[1] == nside[2]):
            raise ValueError("Top patch is not cubic; expected equal cells per side.")
        return int(nside[0])

    def get_offset(self) -> np.ndarray:
        """Return the offset to apply to source positions for local processing."""
        return self.bbox_min[0]

def enclosing_sphere(
    centers: np.ndarray,
    radii: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, float]:
    """Approximate the minimum enclosing sphere of spheres/balls.

    The objective is:
        minimize_c max_i ||c - x_i|| + r_i

    Parameters
    ----------
    centers : np.ndarray
        Sphere centers, shape `(N, 3)`.
    radii : np.ndarray
        Sphere radii, shape `(N,)`.
    max_iter : int
        Maximum number of fixed-point iterations.
    tol : float
        Convergence tolerance on center displacement.

    Returns
    -------
    Tuple[np.ndarray, float]
        Estimated enclosing sphere center and radius.
    """
    if len(centers) == 0:
        return np.zeros(3), 0.0
    if len(centers) == 1:
        return centers[0].copy(), float(radii[0])

    c = centers.mean(axis=0)

    for k in range(max_iter):
        d = np.linalg.norm(centers - c[None, :], axis=1) + radii
        j = np.argmax(d)
        direction = centers[j] - c
        norm = np.linalg.norm(direction)
        if norm > 0.0:
            direction = direction / norm
        else:
            direction = np.zeros(3)

        # diminishing step
        eta = 1.0 / (k + 2.0)
        c_new = c + eta * direction * max(1e-12, np.linalg.norm(centers[j] - c))

        if np.linalg.norm(c_new - c) < tol:
            c = c_new
            break
        c = c_new

    R = np.max(np.linalg.norm(centers - c[None, :], axis=1) + radii)
    return c, float(R)


def morton_like_key(p: np.ndarray, domain_min: np.ndarray, domain_max: np.ndarray, bits: int = 10) -> int:
    """
    Lightweight Morton-like ordering. 
    Maps point to integer grid then interleaves bits.

    Parameters
    ----------
    p : np.ndarray
        Point coordinates (shape `(3,)`).
    domain_min : np.ndarray
        Minimum corner of the domain (shape `(3,)`).
    domain_max : np.ndarray
        Maximum corner of the domain (shape `(3,)`).
    bits : int
        Number of bits per dimension for the grid. Total key bits will be 3x this.

    Returns
    -------
    int     Morton-like key for the point.
    """
    # Normalize to [0, 1]
    x = np.clip((p - domain_min) / np.maximum(domain_max - domain_min, 1e-12), 0.0, 1.0 - 1e-12)

    # Scale to integer by shifting by bits. 
    grid = (x * (1 << bits)).astype(int)

    def split_by_3(v: int) -> int:
        out = 0
        for i in range(bits):
            out |= ((v >> i) & 1) << (3 * i)
        return out

    return split_by_3(grid[0]) | (split_by_3(grid[1]) << 1) | (split_by_3(grid[2]) << 2)

def evaluate_group(group_sources: List[Source], grid: VariableResolutionGrid) -> Group:
    """
    Build a `Group` and compute its geometric and cost properties.

    Parameters
    ----------
    group_sources : List[Source]
        Sources that belong to this group.
    grid : VariableResolutionGrid
        Grid used to estimate local voxel counts.

    Returns
    -------
    Group     Group object with computed center, radius, bounding box, local voxel count, and cost.
    """
    centers = np.array([s.pos for s in group_sources], dtype=float)
    radii = np.array([s.radius for s in group_sources], dtype=float)

    c, R = enclosing_sphere(centers, radii)
    bbox_min = np.maximum(c - R, grid.domain_min)
    bbox_max = np.minimum(c + R, grid.domain_max)
    voxels = grid.find_voxels_in_bbox(bbox_min, bbox_max) # A cubic box around the group center with side length 2R, clipped to the domain bounds.

    # simple cost model: number of sources times local voxel count
    cost = len(group_sources) * voxels[0][3]  # using the volume-based voxel count from the most refined patch

    return Group(
        sources=list(group_sources),
        center=c,
        radius=R,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        voxels=voxels,
        cost=cost,
    )

# TODO CB: this can be generalized with different grouping strategies, e.g. space-filling curve grouping, clustering-based grouping, etc...
# Thius funciton could also be interfaced to external domain decomposition libraries or tools such as Cornerstone.
def build_groups(
    sources: List[Source],
    grid: VariableResolutionGrid,
    r_group_max: float = -1.0,
    nsrc_max: int = 12,
    nvox_max: int = 80000,
    cost_max: float = 500000
) -> List[Group]:
    """Grouping of sources in Morton-like spatial order.

    Sources are sorted spatially, then accumulated into a running group until
    the radius constraint is violated.

    Parameters
    ----------
    sources : List[Source]
        List of sources to group.
    grid : VariableResolutionGrid
        Grid used to estimate local voxel counts for groups.
    r_group_max : float
        Maximum allowed group radius (enclosing sphere of source spheres).
    nsrc_max : int
        Maximum allowed number of sources in a group.
    nvox_max : int
        Maximum allowed number of local voxels covered by the group bounding box.
    cost_max : float
        Maximum allowed computational cost for the group.

    Returns
    -------
    List[Group]     List of groups that satisfy the constraints.
    """

    if not sources:
        return []

    # spatial ordering
    dmin, dmax = grid.domain_min, grid.domain_max
    ordered = sorted(sources, key=lambda s: morton_like_key(s.pos, dmin, dmax))

    groups: List[Group] = []
    current: List[Source] = []

    def valid(g: Group) -> bool:
        # TODO CB: I'm only reporting the number of voxels of the first overlapping patch.
        print(f"Evaluating group with {len(g.sources)} sources, radius {g.radius:.2f}, nvox {g.get_num_voxels()[0]}, cost {g.cost:.2f}")
        if r_group_max > 0.0:
            return (
                g.radius <= r_group_max
                and len(g.sources) <= nsrc_max
                # and g.nvox_local <= nvox_max
                # and g.cost <= cost_max
            )
        else:
            return (
                len(g.sources) <= nsrc_max
                # and g.nvox_local <= nvox_max
                # and g.cost <= cost_max
            )

    for s in ordered:
        if not current:
            current = [s]
            continue

        trial = current + [s]
        gtrial = evaluate_group(trial, grid)

        if valid(gtrial):
            current = trial
        else:
            groups.append(evaluate_group(current, grid))
            current = [s]

    if current:
        groups.append(evaluate_group(current, grid))

    return groups


def assign_groups_to_ranks(groups: List[Group], nranks: int):
    """
    Groups to ranks assignement according to cost.

    Parameters
    ----------
    groups : List[Group]
        List of groups to assign.
    nranks : int
        Number of ranks to distribute groups across.

    Returns
    Tuple[List[List[Group]], List[float]]
        A tuple of (rank_groups, rank_costs), where rank_groups is a list of lists of groups assigned to each rank,
        and rank_costs is the total cost for each rank. 
    """
    rank_groups = [[] for _ in range(nranks)]
    rank_costs = [0.0 for _ in range(nranks)]

    for g in sorted(groups, key=lambda x: x.cost, reverse=True):
        r = int(np.argmin(rank_costs))
        rank_groups[r].append(g)
        rank_costs[r] += g.cost

    return rank_groups, rank_costs
