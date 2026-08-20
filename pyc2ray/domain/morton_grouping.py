"""
This file contains the MortonGroupingParams and MortonSourceGrouping classes,
which provide an implementation of the Morton ordering-based grouping algorithm.
"""

import logging
from dataclasses import dataclass

import numpy as np

from pyc2ray.domain.cost_model import CostModel
from pyc2ray.domain.grid import Grid
from pyc2ray.domain.source_grouping import GroupingParams, SourceGrouping
from pyc2ray.domain.sources import Source, SourceGroup
from pyc2ray.domain.utils import (
    evaluate_sphere_intersection, 
    evaluate_sphere_intersection_new, 
    find_enclosing_sphere,
    find_enclosing_sphere_new,
)

logger = logging.getLogger(__name__)

@dataclass
class MortonGroupingParams(GroupingParams):
    """Parameters specific to the Morton grouping algorithm."""

    morton_bits: int = 10


# TODO: split geometric ordeding (Morton-like key) from the actual grouping logic,
# which is more related to the cost model and to the constraints on the groups.
class MortonSourceGrouping(SourceGrouping):
    """Morton ordering-based grouping algorithm."""

    def _morton_like_key(
        self, p: np.ndarray, domain_min: np.ndarray, domain_max: np.ndarray, bits: int
    ) -> int:
        """
        Lightweight Morton-like ordering.
        Maps point to integer grid then interleaves bits.

        Parameters
        ----------
        p : Point coordinates (shape `(3,)`).
        domain_min : Minimum corner of the domain (shape `(3,)`).
        domain_max : Maximum corner of the domain (shape `(3,)`).
        bits : Number of bits per dimension for the grid. Total key bits will be 3x this.

        Returns
        -------
        Morton-like key for the point.
        """
        # Normalize input coordinates to [0, 1]
        normalized_position = np.clip(
            (p - domain_min) / np.maximum(domain_max - domain_min, 1e-12),
            0.0,
            1.0 - 1e-12,
        )

        # Scale to integer by shifting by bits: the normalized_position input position is a 3 element array
        # of floating point numbers in [0, 1], so multiplying them by 2^bits gives an integer in [0, 2^bits)
        # when truncated. THe larger the bits, the finer the spatial resolution of the Morton ordering, but
        # also the larger the resulting keys (which can affect performance and memory usage).
        int_position = (normalized_position * (1 << bits)).astype(int)

        # Interleave bits to get the Morton key. The interleaving is done by taking the bits of each coordinate
        # and placing them in the final key in an interleaved manner.
        # TODO: the loop over bits for each coordinate for each source can become a hotspot when ordering large
        # source lists. Consider using a faster bit-interleaving approach (e.g., precomputed lookup tables per byte/word,
        # vectorized numpy where practical, or a specialized Morton encoding routine) to reduce per-source overhead during sorting.
        def split_by_3(v: int) -> int:
            out = 0
            for i in range(bits):
                out |= ((v >> i) & 1) << (3 * i)
            return out

        # The final Morton key is obtained by interleaving the bits of the x, y, and z coordinates.
        # For example, if bits=10, we take the 10 bits of the x coordinate and place them in positions 0, 3, 6, ...,
        # the 10 bits of the y coordinate and place them in positions 1, 4, 7, ..., and the 10 bits of the z coordinate
        # and place them in positions 2, 5, 8, ... of the final key.
        return (
            split_by_3(int_position[0])
            | (split_by_3(int_position[1]) << 1)
            | (split_by_3(int_position[2]) << 2)
        )

    def _build_group(
        self, group_sources: list[Source], grid: Grid, cost_model: CostModel
    ) -> SourceGroup:
        """
        Build a group of sources and compute its geometric and cost properties.

        Parameters
        ----------
        group_sources : Sources that belong to this group.
        grid : Grid used to estimate local cell counts.

        Returns
        -------
        Source group with computed center, radius, bounding box, local cell count, and cost.
        """
        centers = np.array([s.pos for s in group_sources], dtype=float)
        radii = np.array([s.radius for s in group_sources], dtype=float)

        # Find group enclosing sphere and bounding box. The enclosing sphere is used for the radius constraint,
        # while the bounding box is used to estimate the local cell count for cost evaluation.
        c, R = find_enclosing_sphere_new(centers, radii)
        bbox_min = c - R
        bbox_max = c + R
        # Basic cost evaluation: number of sources times local cell count
        # TODO: this is a very rough estimate. A more accurate cost model could be implemented
        # for example by evaluating the actual raytracing cost for a representative source in the group.
        # TODO: this estimate of n_cells_per_side is not correct in case of non periodic conditions
        n_cells_in_box = grid.find_num_cells_in_box(bbox_min, bbox_max)
        n_cells_per_side = max(1, int(np.ceil(n_cells_in_box ** (1.0 / 3.0))))
        # The cost model expects the radius of influence in grid units, while the source radius is
        # stored as a physical length. Convert it using the local cell size around the group center
        # (constant for regular grids, position dependent for non-uniform grids such as AMR).
        radius_in_grid_units = group_sources[0].radius / grid.get_average_cell_size(c)
        mem_cost, comp_cost = cost_model.compute_group_costs(
            radius_in_grid_units,
            n_cells_per_side,
            len(group_sources),
        )

        return SourceGroup(
            id=-1,  # ID will be assigned later
            sources=list(group_sources),
            center=c,
            radius=R,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            mem_cost=mem_cost,
            comp_cost=comp_cost,
        )

    # TODO: check source ordering influence on group creation: ideally, the order of sources in the input list should not influence the final groups,
    # which should only depend on their spatial distribution and on the cost model.
    def build_groups(
        self,
        sources: list[Source],
        grid: Grid,
        grouping_params: GroupingParams,
        cost_model: CostModel,
    ) -> list[SourceGroup]:
        """Build the groups of sources to be assigned to the ranks using Morton ordering.

        Parameters
        ----------
        sources : List of sources in the provided grid.
        grid : The grid of the simulation. (can be a sub-grid, in case of recursive grouping)
        grouping_params : The parameters for the Morton grouping algorithm. Must be an
        instance of MortonGroupingParams.
        cost_model : The cost model to use for the evaluation of the cost of processing a group of sources.

        Returns
        -------
        The list of source groups to be assigned to the ranks.
        """
        if not isinstance(grouping_params, MortonGroupingParams):
            raise TypeError("Morton grouping requires MortonGroupingParams.")

        if not sources:
            return []

        # Compute spatial ordering
        ordered_sources = sorted(
            sources,
            key=lambda s: self._morton_like_key(
                s.pos,
                grid.get_domain_min(),
                grid.get_domain_max(),
                grouping_params.morton_bits,
            ),
        )

        def valid(g: SourceGroup) -> bool:
            return (
                len(g.sources) <= grouping_params.max_num_sources_per_group
                and g.mem_cost <= cost_model.max_memory_cost_per_group
            )

        # Profiling counters to characterize grouping efficiency.
        non_intersection_splits = 0
        rejected_trial_groups = 0
        accepted_trial_groups = 0

        source_groups: list[SourceGroup] = []
        current_group: list[Source] = [ordered_sources[0]]
        gtrial = self._build_group(current_group, grid, cost_model)
        for s in ordered_sources[1:]:

            # Check if the new source intersects with the current group. If not, we can start a new group.
            if not evaluate_sphere_intersection_new(
                gtrial.center, gtrial.radius, s.pos, s.radius
            ):
                non_intersection_splits += 1
                source_groups.append(gtrial)
                current_group = [s]
                gtrial = self._build_group(current_group, grid, cost_model)
                continue

            # If the new source intersects with the current group
            # we try to add it to the group and check if it's still valid.
            trial = current_group + [s]
            gtrial = self._build_group(trial, grid, cost_model)

            if valid(gtrial):
                accepted_trial_groups += 1
                current_group = trial
            else:
                rejected_trial_groups += 1
                source_groups.append(self._build_group(current_group, grid, cost_model))
                current_group = [s]
                gtrial = self._build_group(current_group, grid, cost_model)

        if current_group:
            source_groups.append(self._build_group(current_group, grid, cost_model))

        # Update group IDs
        for i, g in enumerate(source_groups):
            g.id = i

        logger.info(
            (
                "Morton grouping stats | num_sources=%d num_groups=%d "
                "accepted_trial_merges=%d rejected_trial_merges=%d "
                "non_intersection_splits=%d"
            ),
            len(sources),
            len(source_groups),
            accepted_trial_groups,
            rejected_trial_groups,
            non_intersection_splits,
        )

        return source_groups

    # TODO: check source ordering influence on group creation: ideally, the order of sources in the input list should not influence the final groups,
    # which should only depend on their spatial distribution and on the cost model.
    def build_groups_parallel(
        self,
        comm,
        sources: list[Source],
        grid: Grid,
        grouping_params: GroupingParams,
        cost_model: CostModel,
    ) -> list[SourceGroup] | None:
        """Build the groups of sources to be assigned to the ranks using Morton ordering.

        Parameters
        ----------
        sources : List of sources in the provided grid.
        grid : The grid of the simulation. (can be a sub-grid, in case of recursive grouping)
        grouping_params : The parameters for the Morton grouping algorithm. Must be an
        instance of MortonGroupingParams.
        cost_model : The cost model to use for the evaluation of the cost of processing a group of sources.

        Returns
        -------
        The list of source groups to be assigned to the ranks on rank 0, and None on
        all other ranks. Must be called by all ranks with the same source list.
        """
        if not isinstance(grouping_params, MortonGroupingParams):
            raise TypeError("Morton grouping requires MortonGroupingParams.")

        rank = comm.Get_rank()
        num_chunks = comm.Get_size()

        # This is a collective routine: every rank must reach the scatter/gather calls
        # below, so returning early here is only safe because `sources` is the full,
        # replicated source list, which makes all ranks agree on this condition. Do not
        # turn this into a rank-local check on a partial source list: ranks taking
        # different branches would deadlock on the collectives.
        if not sources:
            return [] if rank == 0 else None

        if rank == 0:
            # Compute spatial ordering
            ordered_sources = sorted(
                sources,
                key=lambda s: self._morton_like_key(
                    s.pos,
                    grid.get_domain_min(),
                    grid.get_domain_max(),
                    grouping_params.morton_bits,
                ),
            )

            # Split into exactly one chunk per rank; some chunks may be empty.
            chunk_size, extra = divmod(len(ordered_sources), num_chunks)
            ordered_source_chunks = []
            start = 0
            for chunk_idx in range(num_chunks):
                end = start + chunk_size + (1 if chunk_idx < extra else 0)
                ordered_source_chunks.append(ordered_sources[start:end])
                start = end
        else:
            ordered_source_chunks = None

        # Scatter the chunks to all ranks
        local_ordered_sources = comm.scatter(ordered_source_chunks, root=0)

        def valid(g: SourceGroup) -> bool:
            return (
                len(g.sources) <= grouping_params.max_num_sources_per_group
                and g.mem_cost <= cost_model.max_memory_cost_per_group
            )

        # Profiling counters to characterize grouping efficiency.
        non_intersection_splits = 0
        rejected_trial_groups = 0
        accepted_trial_groups = 0

        local_source_groups: list[SourceGroup] = []
        if local_ordered_sources:
            current_group: list[Source] = [local_ordered_sources[0]]
            gtrial = self._build_group(current_group, grid, cost_model)
            for s in local_ordered_sources[1:]:

                # Check if the new source intersects with the current group. If not, we can start a new group.
                if not evaluate_sphere_intersection_new(
                    gtrial.center, gtrial.radius, s.pos, s.radius
                ):
                    non_intersection_splits += 1
                    local_source_groups.append(gtrial)
                    current_group = [s]
                    gtrial = self._build_group(current_group, grid, cost_model)
                    continue

                # If the new source intersects with the current group
                # we try to add it to the group and check if it's still valid.
                trial = current_group + [s]
                gtrial = self._build_group(trial, grid, cost_model)

                if valid(gtrial):
                    accepted_trial_groups += 1
                    current_group = trial
                else:
                    rejected_trial_groups += 1
                    local_source_groups.append(
                        self._build_group(current_group, grid, cost_model)
                    )
                    current_group = [s]
                    gtrial = self._build_group(current_group, grid, cost_model)

            local_source_groups.append(
                self._build_group(current_group, grid, cost_model)
            )

        # Gather all local groups (and original chunk indices) to the root rank
        num_local_groups = len(local_source_groups)
        all_source_groups = comm.gather(local_source_groups, root=0)
        all_num_local_groups = comm.gather(num_local_groups, root=0)
        all_accepted_trial_groups = comm.gather(accepted_trial_groups, root=0)
        all_rejected_trial_groups = comm.gather(rejected_trial_groups, root=0)
        all_non_intersection_splits = comm.gather(non_intersection_splits, root=0)

        if rank == 0:
            tmp_source_groups = [
                group
                for rank_source_groups in all_source_groups
                for group in rank_source_groups
            ]
            tmp_source_groups_chunk_indexes = [0]
            for n_groups in all_num_local_groups:
                tmp_source_groups_chunk_indexes.append(
                    tmp_source_groups_chunk_indexes[-1] + n_groups
                )
            accepted_trial_groups = sum(all_accepted_trial_groups)
            rejected_trial_groups = sum(all_rejected_trial_groups)
            non_intersection_splits = sum(all_non_intersection_splits)


        if comm.Get_rank() == 0:
            # Check if groups from different ranks can be merged together. This check is only
            # performed among groups that are close to the boundary between ranks, in order to
            # avoid spurious group splitting only due to the spatial partitioning of sources among
            # ranks. This is a simple heuristic that can be improved in the future by implementing
            # a more global merging strategy.
            source_groups = []
            accepted_merged_groups = 0
            rejected_merged_groups = 0
            for chunk_idx in range(comm.Get_size()):
                chunk_start = tmp_source_groups_chunk_indexes[chunk_idx]
                chunk_end = tmp_source_groups_chunk_indexes[chunk_idx + 1]
                chunk_groups = tmp_source_groups[chunk_start:chunk_end]
                if not chunk_groups:
                    continue

                if source_groups:
                    # Try to merge the first group of the current chunk with the last group of the previous chunk
                    prev_group = source_groups[-1]
                    curr_group = chunk_groups[0]
                    trial = self._build_group(
                        prev_group.sources + curr_group.sources, grid, cost_model
                    )
                    if valid(trial):
                        accepted_merged_groups += 1
                        source_groups[-1] = trial
                        chunk_groups = chunk_groups[1:]
                    else:
                        rejected_merged_groups += 1

                source_groups.extend(chunk_groups)

            # Update group IDs
            for i, g in enumerate(source_groups):
                g.id = i

            logger.info(
                (
                    "Morton grouping stats | num_sources=%d num_groups=%d "
                    "accepted_trial_merges=%d rejected_trial_merges=%d "
                    "accepted_merged_merges=%d rejected_merged_merges=%d "
                    "non_intersection_splits=%d"
                ),
                len(sources),
                len(source_groups),
                accepted_trial_groups,
                rejected_trial_groups,
                accepted_merged_groups,
                rejected_merged_groups,
                non_intersection_splits,
            )

            return source_groups

        return None
