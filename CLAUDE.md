# CLAUDE.md

Notes for future sessions in this repo.

## Week-2 experiment: is GroundingDINO's relational clause inert?

Two notebooks. **Work in notebooks, not scripts** — cells are kept small and
single-purpose so individual pieces can be re-run while debugging.

| notebook | role |
|----------|------|
| `scenario_redcar_behind_bus.ipynb` | the original single 15 s clip; source of the GT graph schema and the projection helpers |
| `01_record_sweep.ipynb` | records the sweep into `runs/sweep/` |
| `02_evaluate_grounding.ipynb` | scores a detector against the GT graphs |

Dependencies are limited to **carla, numpy, PIL, pandas, matplotlib**. Ask before
adding anything else. No tracking, no training.

### The design

The scene is parked, so a long clip is hundreds of near-identical frames. The sweep
trades clip length for variety: **15 configurations x 2 conditions x 20 frames (1 s)**.

Each configuration is one spawn point, one distractor offset, one camera pose
(azimuth / elevation / standoff, **roll always 0.0**), all drawn from a seeded RNG
(`SEED = 20260731`). The two conditions of a config share the *same* camera and the
*same* blueprints and differ only in which red car sits behind the bus:

| condition | red target | red distractor | correct answer to "red car behind the bus" |
|-----------|------------|----------------|--------------------------------------------|
| `behind`  | behind the bus | off to the side | **target** |
| `front`   | in front of the bus | behind the bus | **distractor** |

That is the whole point: one prompt, two different correct answers. A detector that
ignores "behind the bus" and just grounds "red car" gives the same answer in both.

Three slots in the bus's local frame — `behind` (−12 m), `front` (+12 m, nudged
sideways by `0.35 * lat_off` so the near car does not sit on top of the bus in image
space), and `side` (`fwd_off` forward, `lat_off` right). The condition is just a
slot assignment (`ROLE_SLOT` in cell 6).

Camera azimuth is kept within ±38° of the bus's nose axis, so the camera always sits
*in front* of the bus. That is what makes "behind the bus in the bus's frame" coincide
with viewer-centric `behind` in the GT graph. Verified: the depth ordering survives
even the maximum retry jitter (±52° total).

### `runs/sweep/` layout

```
runs/sweep/
  manifest.json                       <- index; the eval notebook reads only this
  <config_id>/                        <- cfg00 .. cfg14
    behind/
      rgb/000000.png .. 000019.png    <- 1920x1080 RGB
      gt_graphs/000000.json .. 000019.json
    front/
      rgb/ ...
      gt_graphs/ ...
  eval_grounding.csv                  <- written by notebook 02
  eval_flip_test.csv
```

Roughly 2 GB per full sweep at 1920x1080. Drop `WIDTH, HEIGHT` in cell 2 of notebook 01
if that is too much.

### GT graph schema — **unchanged, do not modify**

Identical to `scenario_redcar_behind_bus.ipynb`, which is why the older
`runs/redcar_behind_bus/` graphs stay readable:

```jsonc
{
  "frame": 0,
  "convention": "viewer_centric",
  "ids":   {"bus": 51, "target": 52, "distractor": 53},
  "nodes": [{"id": 52, "class": "car", "color": "red", "color_rgb": "180,20,20",
             "box2d": [x1, y1, x2, y2], "loc": [x, y, z], "yaw": 0.0}],
  "edges": [{"subj": 52, "relation": "behind", "obj": 51}]
}
```

Relations are **viewer-centric**, computed in camera coordinates with fixed thresholds:
`behind` / `in_front_of` at ±2.0 m of depth, `left_of` / `right_of` at ±1.5 m laterally.
`box2d` is absolute pixels, from the 8 projected world-space bounding-box vertices.

The projection helpers (`build_projection_matrix`, `get_image_point`, `camera_xyz`) are
copied **verbatim** across notebooks. The only change to `frame_graph` is that `ids` is
now a parameter rather than a global, because the sweep respawns actors per clip.

### `manifest.json` fields

Top level: `seed`, `fps`, `frames_per_clip`, `conditions`, `prompt`, `clips[]`, `skipped[]`.

Each entry in `clips[]`:

| field | meaning |
|-------|---------|
| `config_id` | `cfg00` … `cfg14` |
| `condition` | `behind` or `front` |
| `ids` | `{bus, target, distractor}` → CARLA actor ids **for this clip only** |
| `correct_answer_role` | `target` in `behind`, `distractor` in `front` |
| `correct_answer_id` | actor id the prompt "red car behind the bus" refers to |
| `confuser_role` / `confuser_id` | the *other* red car |
| `camera` | `{location:[x,y,z], rotation:{pitch,yaw,roll}, fov, image_size}` |
| `frames` | frame count (20) |
| `rgb_dir` / `gt_dir` | repo-relative paths |
| `validation` | `{status:"ok", attempts:N, checks:[...]}` — `attempts` counts camera framings tried |
| `config` | the raw config row that generated the clip |

**Actor ids are per-clip.** The sweep respawns everything for every clip, so the ids in
the `behind` and `front` clips of the same config differ. Always compare by *role*, never
by raw id — notebook 02 does all of its scoring in role space for this reason.

Each entry in `skipped[]`: `config_id`, `reason`, `attempts`, and `last_failures`
(the per-condition validation messages from the final attempt).

### Validation (before anything is written)

A configuration is recorded only if **both** conditions pass on a single probe tick with
the same camera. This is what guarantees the camera is identical across the two
conditions. Checks:

1. **In frame** — all three vehicles project in front of the camera and their full 2D
   boxes lie inside `[0, WIDTH] x [0, HEIGHT]`.
2. **Separated** — no pair of 2D boxes overlaps by more than **0.5 IoU**.
3. **Relation holds** — the condition-defining edges exist in the GT graph:
   - `behind`: `target behind bus`, and `distractor` must **not** also be behind the bus
     (otherwise the prompt has two correct answers)
   - `front`: `target in_front_of bus` **and** `distractor behind bus`

On failure the camera is perturbed (azimuth ±14°, elevation −4/+9°, standoff +12 % per
attempt; roll stays 0.0) and retried up to **5** times, after which the config is skipped
and the reasons logged to `manifest.json`. Perturbations are seeded per config
(`Random(f'{SEED}-{config_id}')`) so a re-run reproduces the same sweep.

Every spawn is torn down in a `finally`, so a failing config cannot leak actors.

### The three scoring definitions (notebook 02)

Middle frame only (`frames // 2`). Detections are matched to GT `box2d` by IoU with a
**0.5 threshold** (`>=`, so exactly 0.50 counts as a match); `match_role` returns the role
of the highest-IoU GT box, or `None`.

| metric | definition |
|--------|------------|
| **selection accuracy** | role of the **top-scoring** detection == `correct_answer_role` |
| **distractor confusion** | role of the **top-scoring** detection == `confuser_role` |
| **anchor recall** | **any** detection matches the bus |

`chosen_car_role` is tracked separately: the top-scoring detection that matches *a car*
(bus matches skipped). It exists so that a run where the bus outranks every car does not
silently count as "no flip".

Three prompt variants, increasing relational content: `car`, `red car`,
`red car behind the bus`.

#### The flip test — the headline number

Per config and prompt, does `chosen_car_role` differ between the `behind` and `front`
conditions?

- **flip rate 1.0** — the detector tracks the relation.
- **flip rate 0.0** — the relational clause is **inert**; the detector picks the same car
  either way and "behind the bus" is doing no work. This is the result the experiment is
  built to detect.
- **correct-flip rate** — flipped *and* landed correctly in both conditions (`target` in
  `behind`, `distractor` in `front`). A detector that flips for the wrong reason (e.g. it
  always picks the nearest car) scores on flip rate but not here.

### Gotchas

- `run_inference(image, text_prompt)` in notebook 02 is a **placeholder** — it must
  return boxes as `[x1, y1, x2, y2]` in **absolute pixels**. HF's
  `GroundingDinoProcessor` returns normalised `cxcywh`; convert inside the stub.
- Notebook 02 has an optional fake-detector cell (`USE_FAKE_DETECTOR`) for smoke-testing
  the scoring plumbing without a model. It simulates an inert detector.
- Re-running the sweep cell **deletes and rewrites** `runs/sweep/`.
- Notebook 01 must be run to the cleanup cell, or the CARLA server is left in
  synchronous mode and the next connection will appear to hang.
- `pandas` and `matplotlib` are not currently installed in the `carla` conda env
  (`C:\Users\azhar\miniconda3\envs\carla`); it has carla + numpy only.
