from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import cast

import numpy as np
from mpi4py import MPI
from pyc2ray.domain.cost_model import pyC2RayCostModel
from pyc2ray.domain.morton_grouping import MortonGroupingParams
from pyc2ray.domain.regular_grid import RegularGrid
from pyc2ray.domain.sources import Source
from pyc2ray.domain.subdomain import Subdomain
from pyc2ray.utils import bin_sources
from pyc2ray.visualization.domain_decomposition import export_domain_decomposition_npz

alps_memory_per_GPU = 96e9 # 96 GB
ranks_per_GPU = 1
max_memory_cost_per_group=alps_memory_per_GPU/ranks_per_GPU

DEFAULT_SOURCES_DIR = Path("/capstor/store/cscs/pasc/c45/CDM_100Mpc_2048/sources/")
DEFAULT_PATTERN = "CDM_100Mpc_2048.[0-9][0-9][0-9][0-9][0-9].halo.txt"
DOMAIN_MIN = 0.0
DOMAIN_MAX = 100.0
GLOBAL_GRID_CELLS_PER_SIDE = 250
CACHE_DIR = Path(__file__).resolve().parent / "cache_npy_sources"
MAX_MEMORY_COST_PER_GROUP = max_memory_cost_per_group
GROUPING_MAX_SOURCES = 500
GROUPING_MORTON_BITS = 10
SOURCE_BATCH_SIZE = 1
PHOTO_ION_TABLE_SIZE = 2000

# Source radius in physical units (cMpc), following the same Worseck2014
# MFP parametrization used in examples/fstar_simulation/parameters.yml
# evaluated at zred_0 = 21.48153.
FSTAR_REF_Z = 21.48153
FSTAR_A_MFP = 210.0
FSTAR_ETA_MFP = -9.0
FSTAR_Z1_MFP = 6.0
FSTAR_ETA1_MFP = 9.0
SOURCE_RADIUS_PHYSICAL = FSTAR_A_MFP * ((1.0 + FSTAR_REF_Z) / 5.0) ** FSTAR_ETA_MFP
SOURCE_RADIUS_PHYSICAL *= 1.0 + (
	((1.0 + FSTAR_REF_Z) / (1.0 + FSTAR_Z1_MFP)) ** FSTAR_ETA1_MFP
)


def find_source_files(sources_dir: Path, pattern: str, max_files: int | None) -> list[Path]:
	"""Return sorted list of source files matching the requested pattern."""
	files = sorted(sources_dir.glob(pattern))
	if not files:
		raise FileNotFoundError(
			f"No source files found in '{sources_dir}' with pattern '{pattern}'."
		)
	if max_files is not None:
		if max_files < 1:
			raise ValueError("MAX_FILES must be >= 1 or None.")
		files = files[:max_files]
	return files


def create_subdomain(comm: MPI.Comm) -> tuple[Subdomain, RegularGrid]:
	"""Create a Subdomain object bound to a global periodic RegularGrid."""
	subdomain = Subdomain(comm=comm)
	cell_size = (DOMAIN_MAX - DOMAIN_MIN) / GLOBAL_GRID_CELLS_PER_SIDE
	global_grid = RegularGrid(
		cell_size=cell_size,
		num_cells=GLOBAL_GRID_CELLS_PER_SIDE,
		is_periodic_mode_active=True,
	)
	subdomain.global_grid = global_grid
	return subdomain, global_grid


def wrap_periodic(coords: np.ndarray, domain_min: float, domain_max: float) -> np.ndarray:
	"""Wrap coordinates to the periodic domain [domain_min, domain_max]."""
	period = domain_max - domain_min
	wrapped = ((coords - domain_min) % period) + domain_min
	return wrapped


def get_cache_paths(source_file: Path, cache_dir: Path) -> tuple[Path, Path]:
	"""Return data and metadata cache paths for one source file."""
	key = hashlib.sha1(str(source_file).encode("utf-8")).hexdigest()[:16]
	cache_data = cache_dir / f"{source_file.name}.{key}.npy"
	cache_meta = cache_dir / f"{source_file.name}.{key}.json"
	return cache_data, cache_meta


def get_binned_cache_paths(source_file: Path, cache_dir: Path) -> tuple[Path, Path, Path]:
	"""Return binned positions, weights and metadata cache paths for one source file."""
	key = hashlib.sha1(str(source_file).encode("utf-8")).hexdigest()[:16]
	cache_pos = cache_dir / f"{source_file.name}.{key}.binned_pos.npy"
	cache_wgt = cache_dir / f"{source_file.name}.{key}.binned_wgt.npy"
	cache_meta = cache_dir / f"{source_file.name}.{key}.binned.json"
	return cache_pos, cache_wgt, cache_meta


def cache_is_valid(
	source_file: Path,
	cache_meta_path: Path,
	domain_min: float,
	domain_max: float,
) -> bool:
	"""Check if the cache metadata still matches the source file and settings."""
	if not cache_meta_path.exists():
		return False

	try:
		with open(cache_meta_path, "r", encoding="utf-8") as fmeta:
			meta = json.load(fmeta)
	except (OSError, json.JSONDecodeError):
		return False

	stat = source_file.stat()
	return (
		meta.get("source_mtime_ns") == stat.st_mtime_ns
		and meta.get("source_size") == stat.st_size
		and float(meta.get("domain_min", -1.0)) == domain_min
		and float(meta.get("domain_max", -1.0)) == domain_max
	)


def binned_cache_is_valid(
	source_file: Path,
	cache_meta_path: Path,
	domain_min: float,
	domain_max: float,
	meshsize: int,
) -> bool:
	"""Check if binned cache metadata matches source file and binning settings."""
	if not cache_meta_path.exists():
		return False

	try:
		with open(cache_meta_path, "r", encoding="utf-8") as fmeta:
			meta = json.load(fmeta)
	except (OSError, json.JSONDecodeError):
		return False

	stat = source_file.stat()
	return (
		meta.get("source_mtime_ns") == stat.st_mtime_ns
		and meta.get("source_size") == stat.st_size
		and float(meta.get("domain_min", -1.0)) == domain_min
		and float(meta.get("domain_max", -1.0)) == domain_max
		and int(meta.get("meshsize", -1)) == meshsize
	)


def load_last_three_columns(
	source_file: Path,
	domain_min: float,
	domain_max: float,
) -> tuple[np.ndarray, float, int, float, np.ndarray]:
	"""Load x,y,z, detect out-of-domain particles, and wrap them periodically."""
	t0 = time.perf_counter()
	last_three = np.loadtxt(source_file, usecols=(-3, -2, -1), ndmin=2)

	outside_mask = (last_three < domain_min) | (last_three > domain_max)
	particle_outside_mask = np.any(outside_mask, axis=1)
	particle_outside_indices = np.flatnonzero(particle_outside_mask)
	n_outside = int(particle_outside_indices.size)

	max_deviation = 0.0
	if n_outside > 0:
		below_dev = domain_min - last_three[outside_mask & (last_three < domain_min)]
		above_dev = last_three[outside_mask & (last_three > domain_max)] - domain_max
		if below_dev.size > 0:
			max_deviation = max(max_deviation, float(np.max(below_dev)))
		if above_dev.size > 0:
			max_deviation = max(max_deviation, float(np.max(above_dev)))
		last_three[particle_outside_mask] = wrap_periodic(
			last_three[particle_outside_mask], domain_min, domain_max
		)

	dt = time.perf_counter() - t0
	return last_three, dt, n_outside, max_deviation, particle_outside_indices


def load_last_three_columns_with_cache(
	source_file: Path,
	cache_dir: Path,
	domain_min: float,
	domain_max: float,
) -> tuple[np.ndarray, float, int, float, np.ndarray, bool]:
	"""Load processed x,y,z either from cache (.npy) or from source text file."""
	cache_data_path, cache_meta_path = get_cache_paths(source_file, cache_dir)

	if cache_data_path.exists() and cache_is_valid(
		source_file, cache_meta_path, domain_min, domain_max
	):
		t0 = time.perf_counter()
		data_xyz = np.load(cache_data_path)
		dt = time.perf_counter() - t0

		outside_count = 0
		max_deviation = 0.0
		outside_indices = np.array([], dtype=np.int64)
		try:
			with open(cache_meta_path, "r", encoding="utf-8") as fmeta:
				meta = json.load(fmeta)
			outside_count = int(meta.get("outside_count", 0))
			max_deviation = float(meta.get("max_deviation", 0.0))
			sample = meta.get("outside_sample_indices", [])
			outside_indices = np.array(sample, dtype=np.int64)
		except (OSError, json.JSONDecodeError, ValueError, TypeError):
			pass

		return data_xyz, dt, outside_count, max_deviation, outside_indices, True

	(
		data_xyz,
		dt,
		n_outside,
		max_deviation,
		outside_indices,
	) = load_last_three_columns(source_file, domain_min, domain_max)

	np.save(cache_data_path, data_xyz)
	stat = source_file.stat()
	meta = {
		"source_path": str(source_file),
		"source_mtime_ns": stat.st_mtime_ns,
		"source_size": stat.st_size,
		"domain_min": domain_min,
		"domain_max": domain_max,
		"outside_count": int(n_outside),
		"max_deviation": float(max_deviation),
		"outside_sample_indices": outside_indices[:5].tolist(),
	}
	with open(cache_meta_path, "w", encoding="utf-8") as fmeta:
		json.dump(meta, fmeta)

	return data_xyz, dt, n_outside, max_deviation, outside_indices, False


def reduce_sources_with_binning(
	srcpos_xyz: np.ndarray,
	domain_min: float,
	domain_max: float,
	meshsize: int,
) -> tuple[np.ndarray, np.ndarray, float]:
	"""Reduce sources by binning to a regular mesh, as in ionizing_flux."""
	t0 = time.perf_counter()
	boxsize = domain_max - domain_min
	shifted_pos = srcpos_xyz - domain_min
	weights = np.ones(shifted_pos.shape[0], dtype=np.float64)
	binned_pos, binned_weights = bin_sources(
		srcpos_mpc=shifted_pos,
		mstar_msun=weights,
		boxsize=boxsize,
		meshsize=meshsize,
	)
	dt = time.perf_counter() - t0
	return binned_pos, binned_weights, dt


def reduce_sources_with_binning_and_cache(
	source_file: Path,
	srcpos_xyz: np.ndarray,
	cache_dir: Path,
	domain_min: float,
	domain_max: float,
	meshsize: int,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
	"""Return binned sources from cache when valid, otherwise compute and cache them."""
	cache_pos_path, cache_wgt_path, cache_meta_path = get_binned_cache_paths(
		source_file, cache_dir
	)

	if (
		cache_pos_path.exists()
		and cache_wgt_path.exists()
		and binned_cache_is_valid(
			source_file, cache_meta_path, domain_min, domain_max, meshsize
		)
	):
		t0 = time.perf_counter()
		binned_pos = np.load(cache_pos_path)
		binned_weights = np.load(cache_wgt_path)
		dt = time.perf_counter() - t0
		return binned_pos, binned_weights, dt, True

	binned_pos, binned_weights, dt = reduce_sources_with_binning(
		srcpos_xyz,
		domain_min,
		domain_max,
		meshsize,
	)

	np.save(cache_pos_path, binned_pos)
	np.save(cache_wgt_path, binned_weights)
	stat = source_file.stat()
	binned_meta = {
		"source_path": str(source_file),
		"source_mtime_ns": stat.st_mtime_ns,
		"source_size": stat.st_size,
		"domain_min": domain_min,
		"domain_max": domain_max,
		"meshsize": meshsize,
		"num_binned_sources": int(binned_pos.shape[0]),
	}
	with open(cache_meta_path, "w", encoding="utf-8") as fmeta:
		json.dump(binned_meta, fmeta)

	return binned_pos, binned_weights, dt, False


def build_sources_from_binned(
	loaded_binned_sources: list[tuple[np.ndarray, np.ndarray]],
	cell_size: float,
	radius: float,
	domain_min: float,
) -> list[Source]:
	"""Build Source objects from binned source positions and weights."""
	sources: list[Source] = []
	source_id = 0

	for binned_pos, binned_weights in loaded_binned_sources:
		for idx, strength in zip(binned_pos, binned_weights):
			pos = domain_min + (idx.astype(np.float64) + 0.5) * cell_size
			sources.append(
				Source(
					id=source_id,
					pos=pos,
					strength=float(strength),
					radius=radius,
				)
			)
			source_id += 1

	return sources


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Load matching halo files and keep only the last three columns."
	)
	parser.add_argument(
		"--max-files",
		type=int,
		default=None,
		help="Maximum number of matching files to load (default: load all).",
	)
	args = parser.parse_args()

	source_files = find_source_files(DEFAULT_SOURCES_DIR, DEFAULT_PATTERN, args.max_files)
	CACHE_DIR.mkdir(parents=True, exist_ok=True)

	total_rows = 0
	total_bytes = 0
	total_time = 0.0
	total_binning_time = 0.0
	total_outside_particles = 0
	global_max_deviation = 0.0
	cache_hits = 0
	cache_misses = 0
	binned_cache_hits = 0
	binned_cache_misses = 0
	total_reduced_sources = 0
	loaded_last_three: list[np.ndarray] = []
	loaded_binned_sources: list[tuple[np.ndarray, np.ndarray]] = []

	print(f"Found {len(source_files)} file(s).")
	if args.max_files is not None:
		print(f"File limit enabled: max_files={args.max_files}")
	for src_file in source_files:
		(
			data_xyz,
			dt,
			n_outside,
			max_deviation,
			outside_indices,
			from_cache,
		) = load_last_three_columns_with_cache(
			src_file,
			CACHE_DIR,
			DOMAIN_MIN,
			DOMAIN_MAX,
		)
		loaded_last_three.append(data_xyz)
		n_rows = data_xyz.shape[0]
		n_bytes = data_xyz.nbytes
		(
			binned_pos,
			binned_weights,
			dt_bin,
			binned_from_cache,
		) = reduce_sources_with_binning_and_cache(
			src_file,
			data_xyz,
			CACHE_DIR,
			DOMAIN_MIN,
			DOMAIN_MAX,
			GLOBAL_GRID_CELLS_PER_SIDE,
		)
		n_reduced = binned_pos.shape[0]
		loaded_binned_sources.append((binned_pos, binned_weights))

		total_rows += n_rows
		total_bytes += n_bytes
		total_time += dt
		total_binning_time += dt_bin
		total_reduced_sources += n_reduced
		total_outside_particles += n_outside
		global_max_deviation = max(global_max_deviation, max_deviation)
		if from_cache:
			cache_hits += 1
		else:
			cache_misses += 1
		if binned_from_cache:
			binned_cache_hits += 1
		else:
			binned_cache_misses += 1

		print(
			f"- {src_file.name}: rows={n_rows}, "
			f"load_time={dt:.3f}s, mem={n_bytes / 1024**2:.2f} MiB, "
			f"source={'cache' if from_cache else 'text'}"
		)
		if n_rows > 0:
			reduction = 100.0 * (1.0 - n_reduced / n_rows)
		else:
			reduction = 0.0
		print(
			f"  binned sources: {n_reduced} (reduction={reduction:.2f}%), "
			f"bin_time={dt_bin:.3f}s, source={'cache' if binned_from_cache else 'computed'}"
		)
		if n_outside > 0:
			sample_idx = outside_indices[:5].tolist()
			print(
				f"  corrected periodic wrap for {n_outside} particle(s), "
				f"max_deviation={max_deviation:.4e}, sample_indices={sample_idx}"
			)

	print("\nLoading summary")
	print(f"- total files: {len(source_files)}")
	print(f"- total rows: {total_rows}")
	print(f"- total loaded memory: {total_bytes / 1024**2:.2f} MiB")
	print(f"- total loading time: {total_time:.3f}s")
	print(f"- total binning time: {total_binning_time:.3f}s")
	print(f"- total reduced sources: {total_reduced_sources}")
	if total_rows > 0:
		print(
			f"- global source reduction: {100.0 * (1.0 - total_reduced_sources / total_rows):.2f}%"
		)
	print(f"- cache hits: {cache_hits}")
	print(f"- cache misses: {cache_misses}")
	print(f"- binned cache hits: {binned_cache_hits}")
	print(f"- binned cache misses: {binned_cache_misses}")
	print(f"- cache directory: {CACHE_DIR}")
	print(f"- particles corrected with periodic wrapping: {total_outside_particles}")
	print(f"- maximum absolute deviation outside domain: {global_max_deviation:.4e}")

	if total_time > 0.0:
		print(f"- throughput: {total_rows / total_time:.1f} rows/s")

	comm: MPI.Comm = MPI.COMM_WORLD
	subdomain, global_grid = create_subdomain(comm)
	print("\nSubdomain setup")
	print(f"- rank: {subdomain.rank}")
	print(f"- communicator size: {subdomain.comm.Get_size()}")
	print(f"- global grid cells per side: {global_grid.num_cells}")
	print(f"- global grid cell size: {global_grid.cell_size:.8f}")
	print(f"- source radius (physical units): {SOURCE_RADIUS_PHYSICAL:.8f}")
	print(
		f"- source radius (cell units): {SOURCE_RADIUS_PHYSICAL / global_grid.cell_size:.8f}"
	)

	sources = build_sources_from_binned(
		loaded_binned_sources,
		global_grid.cell_size,
		SOURCE_RADIUS_PHYSICAL,
		DOMAIN_MIN,
	)
	print(f"- total binned sources prepared for decomposition: {len(sources)}")

	if len(sources) > 0:
		cost_model = pyC2RayCostModel(
			max_memory_cost_per_group=MAX_MEMORY_COST_PER_GROUP,
			source_batch_size=SOURCE_BATCH_SIZE,
			is_periodic_mode_active=True,
			photo_ion_table_size=PHOTO_ION_TABLE_SIZE,
		)
		grouping_params = MortonGroupingParams(
			max_num_sources_per_group=GROUPING_MAX_SOURCES,
			morton_bits=GROUPING_MORTON_BITS,
		)

		t0_decomp = time.perf_counter()
		subdomain.run_decomposition(
			global_grid=global_grid,
			sources=sources,
			cost_model=cost_model,
			grouping_algorithm="morton",
			grouping_params=grouping_params,
		)
		dt_decomp = time.perf_counter() - t0_decomp

		print("\nDecomposition summary")
		print(f"- decomposition time: {dt_decomp:.3f}s")
		print(f"- source groups assigned to rank: {subdomain.get_num_source_groups()}")
		print(f"- rank decomposition cost: {subdomain.cost:.3e}")

		npz_path = CACHE_DIR / f"domain_decomposition_rank{subdomain.rank}.npz"
		local_regular_grids = [cast(RegularGrid, g) for g in subdomain.get_local_grids()]
		exported_npz = export_domain_decomposition_npz(
			global_grid=global_grid,
			source_groups=subdomain.get_source_groups(),
			local_grids=local_regular_grids,
			output_path=npz_path,
		)

		print(f"- decomposition NPZ export: {exported_npz}")
	else:
		print("\nDecomposition summary")
		print("- no sources available, decomposition skipped")

	# Keep arrays alive so this script can be extended with grouping benchmarks.
	_ = loaded_last_three
	_ = loaded_binned_sources


if __name__ == "__main__":
	main()
