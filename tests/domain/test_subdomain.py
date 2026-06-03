"""Unit tests focused on Subdomain._assign_groups_to_ranks."""

from __future__ import annotations

import numpy as np

from pyc2ray.domain.cost_model import CostModel
from pyc2ray.domain.morton_grouping import MortonGroupingParams
from pyc2ray.domain.regular_grid import RegularGrid
from pyc2ray.domain.sources import Source, SourceGroup
from pyc2ray.domain.subdomain import Subdomain


class DummyComm:
    """Small MPI-like communicator stub for assignment tests."""

    def __init__(self, rank: int, size: int) -> None:
        self._rank = rank
        self._size = size

    def Get_rank(self) -> int:
        return self._rank

    def Get_size(self) -> int:
        return self._size

    def scatter(self, data: object, root: int = 0) -> object:
        # Emulate MPI scatter for tests that run decomposition on a single rank.
        if isinstance(data, list) and len(data) == self._size:
            return data[self._rank]
        return data


class DummyCostModel(CostModel):
    """Simple deterministic cost model for Subdomain decomposition tests."""

    def compute_group_costs(
        self, R: float, n_cells_per_side: int, n_src: int
    ) -> tuple[float, float]:
        return float(n_src), float(max(1, n_cells_per_side) * n_src)


def _source(sid: int) -> Source:
    return Source(
        id=sid,
        pos=np.array([0.0, 0.0, 0.0], dtype=float),
        strength=1.0,
        radius=1.0,
    )


def _group(gid: int, comp_cost: float) -> SourceGroup:
    src = _source(gid)
    return SourceGroup(
        id=gid,
        sources=[src],
        center=np.array([0.0, 0.0, 0.0], dtype=float),
        radius=1.0,
        bbox_min=np.array([-1.0, -1.0, -1.0], dtype=float),
        bbox_max=np.array([1.0, 1.0, 1.0], dtype=float),
        mem_cost=comp_cost,
        comp_cost=comp_cost,
    )


def test_assign_groups_to_ranks_balances_by_comp_cost() -> None:
    subdomain = Subdomain(comm=DummyComm(rank=0, size=2))
    groups = [
        _group(0, 9.0),
        _group(1, 8.0),
        _group(2, 7.0),
        _group(3, 6.0),
        _group(4, 5.0),
        _group(5, 4.0),
        _group(6, 3.0),
    ]

    rank_groups, rank_costs = subdomain._assign_groups_to_ranks(groups)

    assert len(rank_groups) == 2
    assert rank_costs == [20.0, 22.0]
    assert [g.id for g in rank_groups[0]] == [0, 3, 4]
    assert [g.id for g in rank_groups[1]] == [1, 2, 5, 6]


def test_assign_groups_to_ranks_handles_empty_groups() -> None:
    subdomain = Subdomain(comm=DummyComm(rank=0, size=3))

    rank_groups, rank_costs = subdomain._assign_groups_to_ranks([])

    assert rank_groups == [[], [], []]
    assert rank_costs == [0.0, 0.0, 0.0]


def test_sources_set_after_decomposition() -> None:
    subdomain = Subdomain(comm=DummyComm(rank=0, size=1))
    grid = RegularGrid(cell_size=1.0, num_cells=16, is_periodic_mode_active=True)
    cost_model = DummyCostModel(max_memory_cost_per_group=1e12)
    grouping_params = MortonGroupingParams(max_num_sources_per_group=2, morton_bits=10)

    positions = [
        np.array([1.0, 1.0, 1.0]),
        np.array([1.4, 1.0, 1.0]),
        np.array([1.8, 1.0, 1.0]),
        np.array([2.2, 1.0, 1.0]),
        np.array([2.6, 1.0, 1.0]),
        np.array([5.0, 5.0, 5.0]),
        np.array([5.4, 5.0, 5.0]),
        np.array([5.8, 5.0, 5.0]),
        np.array([6.2, 5.0, 5.0]),
        np.array([6.6, 5.0, 5.0]),
        np.array([8.0, 8.0, 8.0]),
        np.array([8.4, 8.0, 8.0]),
        np.array([8.8, 8.0, 8.0]),
        np.array([9.2, 8.0, 8.0]),
        np.array([9.6, 8.0, 8.0]),
        np.array([12.0, 12.0, 12.0]),
        np.array([12.4, 12.0, 12.0]),
        np.array([12.8, 12.0, 12.0]),
        np.array([13.2, 12.0, 12.0]),
        np.array([13.6, 12.0, 12.0]),
    ]
    sources = [Source(i, pos, 1.0, 0.6) for i, pos in enumerate(positions)]

    subdomain.run_decomposition(
        global_grid=grid,
        sources=sources,
        cost_model=cost_model,
        grouping_algorithm="morton",
        grouping_params=grouping_params,
    )

    grouped_sources = [s for g in subdomain.get_source_groups() for s in g.sources]

    assert len(grouped_sources) == len(sources)
    grouped_ids = [s.id for s in grouped_sources]
    assert len(set(grouped_ids)) == len(sources)
    assert set(grouped_ids) == {s.id for s in sources}

    expected_pos_by_id = {s.id: s.pos for s in sources}
    got_pos_by_id = {s.id: s.pos for s in grouped_sources}
    for sid, expected_pos in expected_pos_by_id.items():
        np.testing.assert_allclose(got_pos_by_id[sid], expected_pos)
