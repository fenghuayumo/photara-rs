# Clustered NEE-AT mesh-light feedback

An independent implementation inspired by RTXPT's architecture; no RTXPT code
is copied or linked. This is not a feature-for-feature RTXPT port.

## Light LOD and scale

RealTime surface NEE supports up to **2,097,152 sampled emitter occurrences**,
including analytic/environment lights and emissive triangles. Shared mesh
instances retain separate (instance, triangle) identities. Larger scenes fall
back to the original Light Tree without truncating lights.

Spatial light-tree traversal produces the exact terminal light list. A
power-of-two light LOD stride bounds it to 4,096 sampling clusters. The stride
is 1 for small sets and at most 512 at the supported limit. This is a
scene-size-selected sampling LOD, not view-dependent remeshing. Boundaries
follow spatial traversal order and can cross adjacent subtrees. No proxy
geometry, averaged material or approximate emission replaces the real light.

Within each cluster, a static energy-weighted CDF includes 5% uniform
exploration. Cluster priors also include 5% exploration. Analytic-only scenes
retain uniform cold priors. Stored interval widths define exact sampled PDFs.

Feedback is per cluster, globally and in up to 8x8 screen tiles. Adaptive tables
are bounded by clusters times tiles, rather than triangles times tiles. Leaf
records and a sorted instance/emitter lookup use 32 bytes per occurrence. With
4,096 clusters and 65 tables the additional device allocation is 70,370,060
bytes (67.11 MiB), excluding existing geometry and Light Tree allocations.
Initial construction uses additional host staging memory.

## Estimator and learning

The default hybrid proposal is 25% ordinary Light Tree and 75% clustered feedback. With screen
feedback, the latter is one third global and two thirds local; otherwise it is
entirely global. Tables are frozen throughout each finite frame. NEE and forward
BSDF-hit MIS evaluate the same full probability, including cluster probability,
leaf interval width and actual instance identity.

Pure NEE-AT sets the Light Tree fraction to zero. Surface light selection uses
only the global/local feedback distribution, and both the selected-light PDF
and forward BSDF-hit MIS skip Light Tree traversal/PDF evaluation. Positive
cluster priors and leaf exploration retain support without a tree mixture.
Light Tree construction is still reused to enumerate spatially ordered lights
and build clusters. Volume sampling and unsupported configurations retain their
existing fallbacks; pure surface sampling does not remove those limitations.

Blender: in RealTime, enable NEE-AT and choose **NEE-AT Only** or **NEE-AT +
Light Tree** under Light Sampling. Disable NEE-AT for the original Light Tree.
Existing files default to the hybrid policy. Python exposes
`scene.configure_neeat(True, tree_fraction=0.0)` for pure and `0.25` for hybrid;
the finite fraction may be anywhere in [0,1]. Viewer CLI `--light-sampler
NEE_AT_PURE` selects pure; `NEE_AT` retains the hybrid policy and `LIGHT_TREE`
disables feedback sampling. These controls affect RealTime, leaving Reference
unchanged.

Visible, PDF-corrected shadow contributions feed the selected cluster after
transparency attenuation. Fully occluded connections add no feedback; AO and
volume shadow paths are excluded. At a frame boundary, finite positive feedback
is normalized, mixed as 80% previous proposal / 20% target, combined with 5% prior
exploration and normalized again. Empty local tiles inherit the new global target.

Single-device CUDA/OptiX updates, CDF construction and feedback clearing stay on
GPU. Work is bounded by 4,096 clusters times at most 65 tables; no million-entry
scan or feedback/pixel download is added per frame. CPU launch/completion
synchronization remains. Leaf construction and sorting run on CPU at light-table
rebuilds. `PHOTARA_NEEAT_CPU=1` uses the same update policy on the host.

Reference and volume-segment sampling are unchanged. MNEE, portals, light/shadow
linking, disabled Light Tree, multi-device rendering and over-limit scenes use
the original sampler. Screen feedback needs untiled, unscaled, full-origin
buffers; otherwise global feedback remains available. Light-table rebuilds
invalidate leaf metadata and learning. Mode/seed resets and explicit NEE-AT
configuration reset learning; camera navigation may retain valid proposals.

## Diagnostics and validation

Idle-only `scene.neeat_stats()` reports all participating real occurrences as
`active_lights`. `max_pdf` and `local_max_pdf` are complete per-light feedback
probabilities before mixing with the tree. Diagnostic snapshots may download
distribution data outside normal rendering. GPU/CPU update counters remain.

Blender `session_stats()` adds `neeat_emitters`, `neeat_clusters`,
`neeat_lod_stride`, `neeat_local_tables`, `neeat_device_bytes` and
`light_tree_emitters`. Allocation bytes can remain nonzero while disabled,
because metadata is retained until rebuild or session destruction.

Rebuild host binaries, Python extension, CUDA and OptiX blobs together; older
kernels are ABI-incompatible. Validation entry points:

- `src/test/neeat_cluster_test.cpp`: standalone C++17 production-builder and
  update-policy tests, exact CDF intervals, instance keys, nonfinite feedback,
  normalization and 2M memory bounds. Compile with the `src` include directory.
- `scripts/test_blender_neeat_clusters.py`: shared mesh instances, CPU/GPU energy
  against Reference, activation and Reference isolation.
- The same script with `--stress`: 2,097,152 actual emissive triangles lighting
  a diffuse receiver, GPU activation and allocation checks.
- `photara/python/tests/test_neeat.py`: analytic and mixed lighting, camera
  navigation, dynamic rebuilds, resize, resets and idle guards.
- `scripts/benchmark_blender_neeat.py`: unchanged input blend, on/off results
  with optional repeated seeds; history reconstruction and denoising disabled.
  Timings include training and are not steady-frame throughput measurements.

Scalable support does not guarantee a speedup or lower error in every scene.
Compare equal-time output and motion/disocclusion behavior before making such claims.
