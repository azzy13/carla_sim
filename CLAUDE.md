# CLAUDE.md

Notes for future sessions in this repo.

## Week-2 experiment: is GroundingDINO's relational clause inert?

Two notebooks. **Work in notebooks, not scripts** — cells are kept small and
single-purpose so individual pieces can be re-run while debugging.

| notebook | role |
|----------|------|
| `scenario_redcar_behind_bus.ipynb` | the original single 15 s clip; source of the GT graph schema and the projection helpers |
| `01_record_sweep.ipynb` | fly the 15 cameras, then record the sweep into `runs/sweep/` |
| `02_evaluate_grounding.ipynb` | scores a detector against the GT graphs |

Dependencies are limited to **carla, numpy, PIL, pandas, matplotlib**. Ask before
adding anything else. No tracking, no training.

### The design

**15 configurations x 2 conditions x 1 frame.** The scene is parked, so a frame is all
there is to record.

A configuration is one spawn point (the bus), one longitudinal offset for the distractor,
and **one hand-flown camera**. Spawn point and offset come from a seeded RNG
(`SEED = 20260731`); the camera does not — you fly the spectator to it (see below). The
two conditions of a config share the *same* camera and the *same* blueprints, and
**only the target moves**:

| condition | red target | red distractor | correct answer to "red car behind the bus" |
|-----------|------------|----------------|--------------------------------------------|
| `behind`  | behind the bus, same lane | opposing lane, oncoming | **target** |
| `front`   | in front of the bus, same lane | opposing lane, oncoming | **nothing** |

**`front` is a negative control, not a second correct answer.** The distractor is parked
in the opposing lane facing the bus, identically in both conditions, so it is never the
red car behind the bus. A detector that returns a confident red car in `front` has
grounded "red car" and thrown the clause away — which is exactly the failure the
experiment is built to catch. `correct_answer_role` is `null` for `front`.

This replaced a design where the distractor moved into the behind slot in `front`, giving
both conditions one correct answer. Three things went wrong with it and all three are
gone: the correct answer in `front` had to hide behind a 12 m bus (the occlusion problem
below, now halved); the side slot it occupied in `behind` was camera-independent, so a
blocked spawn there could not be rescued by flying; and it needed validation to prove the
prompt had exactly one answer, which the layout now guarantees for free.

Each clip records `target_relation` (`behind` / `in_front_of`), so the mirror prompt
"red car **in front of** the bus" scores off the same images — correct in `front`,
nothing in `behind`.

Three slots. The condition is just a slot assignment (`ROLE_SLOT`).

- `behind` — the **target**, on a waypoint of the bus's own lane, `BUS_GAP` (12 m) up- or
  down-road, whichever end is **further from the camera**.
- `front` — the target on the other end of that same lane pair, i.e. the one nearer.
- `oncoming` — the **distractor**, in the opposing lane, `BUS_NOSE + fwd_off` metres
  ahead of the bus's centre, nose-to-nose with it. Same in both conditions. The
  `BUS_NOSE = 6.5` term is half the fusorosa's length; without it a low `fwd_off` parks
  the distractor alongside the bus's midsection instead of clear of its nose.

**`behind` / `front` ride the lane; which end is which is decided by the camera.**
Both are `carla_map.get_waypoint(bus).next(gap)` / `.previous(gap)`, so they inherit the
lane centre, the lane heading, and the road's curvature — same lane as the bus, correct
parked yaw even on a bend. Fly round to the far side of the bus and the two swap.

**The end is chosen by signed depth along the line of sight, not by camera distance.**
`depth(w) = (w − bus) · v̂`, where `v̂` is the unit camera→bus vector; positive is
`behind`. Comparing raw distances instead — "is this end further from the camera than the
bus is?" — is wrong, and wrong in a way that bites from any off-axis pose: both ends of
the lane are further away than the bus, because `√(D² + BUS_GAP²) > D` at every angle.
That emptied the `front` bucket and raised a bogus *"you are level with the bus"*.
Signed depth cannot do this: the two ends lie in opposite directions from the bus, so
their depths always have opposite signs.

Two earlier versions, both wrong in the opposite direction:

- **Bus-axis offsets** (`anchor + forward*±12`). With an off-axis camera the car appeared
  *beside* the bus: the `behind` edge fired on depth, but a `left_of` / `right_of` edge
  fired too. Spurious lateral edge on **13/15**. Lane placement is the same geometry when
  the road is straight, so it inherits this failure mode — see the azimuth note below.
- **Camera view ray** (`anchor + view_dir*±12`). This fixed the picture from any angle,
  but only by driving the car off the road: at azimuth `t` off the lane it sits
  `12*sin(t)` m sideways in world terms — 4 m at 20°, a whole lane over. That is the
  "target is a little to the side of the bus" look, and it also put the cars into kerbs
  and walls, which is where most `SpawnFailed` came from.

Lane placement makes the tradeoff explicit rather than hiding it in the world: the cars
are always in the lane, and **the shot has to look roughly down that lane**. From the
kerb the lane car really is beside the bus and the picture shows it. `snap()` prints
`lane_azimuth` — degrees between the lane and the line of sight, 0 = straight down the
lane — so it is obvious which way to fly.

The oncoming distractor comes from `opposing_lane(wp)`, which returns
`(waypoint, flip_yaw)`:

1. **Walk outward until the `lane_id` sign flips.** The sign is the direction of travel,
   so the first opposite-signed *driving* lane is the near oncoming lane. Medians and
   shoulders are walked through, not selected. `flip_yaw` is False — a waypoint in an
   opposing lane already faces back at the bus — and "ahead of the bus" is `previous()`
   along it, not `next()`.
2. Only if no lane on the road runs the other way (a real one-way street): the nearest
   adjacent driving lane, `flip_yaw` True.
3. Only with no neighbouring driving lane at all: step `LANE_W = 3.5` m to the bus's left
   and hope. If that is pavement, the spawn fails and says so.

**`get_left_lane()` steps exactly one lane, and a carriageway is usually two lanes wide.**
This is the trap. On a 4-lane road the lanes are `-2, -1 | 1, 2`; from lane -2 the left
neighbour is -1, still our own carriageway. Two earlier versions got this wrong in
opposite directions — one required the immediate neighbour to be opposite-signed and fell
through to the raw offset on every multi-lane road; the next accepted any immediate
neighbour and parked the distractor one lane over facing backwards. Hence the walk.

`lane_report(anchor)` prints the bus's lane id, the distractor's lane id, and whether it
is genuinely oncoming, under every `snap()`. Getting this wrong is silent — the car is on
tarmac either way — so the readout exists to make it checkable without squinting at the
picture.

`LANE_PEEK = 0.0`, replacing `RAY_PEEK = 1.2`. Same idea (step the two lane cars sideways
in opposite directions), but it is off, because a peek is precisely what walks the target
off the lane centre. The assert (`0 <= LANE_PEEK < LAT_THRESH`) still bounds it if it is
ever dialled back up.

**Slot placement has fallbacks now, so a blocked slot rarely costs a config.**
`scene_slots` returns a *list* per slot, best first, and `spawn_scene` walks it:

- lane slots — `BUS_GAP`, then `+3 m`, then `−3 m` (`GAP_FALLBACKS`), plus every branch
  the waypoint query returns at a junction.
- `oncoming` — `opp.previous(fwd_off)`, then `opp.next(fwd_off)`, then the waypoint
  abreast of the bus.

`stage()` no longer raises on `SpawnFailed`. It reports, still moves the spectator to the
perch, and returns — a `tour()` cannot be aborted by one bad starting pose.

#### Occlusion is the binding constraint, and nothing checks it for you

A car in the lane behind a 12 m bus is hidden by it. That is the `behind` condition's
target — the only correct answer in the experiment — so occluding it collapses selection
accuracy for reasons unrelated to the relational clause. No automated check catches it:
the pairwise-IoU number stays near 0.11 because the car's box is far smaller than the
bus's.

**Camera elevation is the only lever.** 12° → 6/15 fully hidden; 24° → 0/15, median 19 px
sliver; 28° → 30 px. (Those numbers were measured with the 1.2 m peek in place; with
`LANE_PEEK = 0` elevation is doing *all* of the work, so if anything shoot higher.) Since
every camera is hand-flown, keep shots at ~24° above the ground or higher. `snap()` and
the section 12a contact sheet both print `% clear of the bus` — but the picture is the
real check, which is why 12a exists.

Only the `behind` condition is exposed to this now. In `front` the target is on the near
side of the bus and the distractor is in the next lane, so nothing is hiding.

### Cameras are hand-flown — `cameras.json`

There is **no procedural camera placement** and **no retry jitter**. Both were removed:
every camera is flown by hand, so azimuth/elevation/standoff, `look_at_rotation`,
`camera_transform_for`, `jitter_for`, and `MAX_RETRIES` no longer exist.

Section 9 of notebook 01 is the workflow:

- `stage(cid)` — parks the bus + both red cars, perches the spectator nearby.
- fly the spectator in the CARLA window (WASD, Q/E, right-drag).
- `snap()` — reads the spectator pose, re-places the cars for it, renders **both
  conditions** side by side with GT boxes, the answer report and the occlusion readout.
  Re-runnable; writes nothing. Returns False only if a spawn was blocked.
- `save_camera()` — persists the pose to `cameras.json`. **Never refuses.**
- `tour()` — walks every config still missing a camera, prompting at each. `s` skips,
  `q` quits (saves are already on disk, so it resumes). This is the normal entry point.
- `drop_camera(cid)`, `cameras_status()`, `unstage()`.

`cameras.json` is `{config_id: {location: [x,y,z], rotation: {pitch,yaw,roll}}}`, at the
repo root so a sweep's `rmtree` cannot take it. The config cell reloads it on every
re-run, so the seeded table can be rebuilt without losing camera work.

**It is written atomically** — temp file, `fsync`, `os.replace`. `open(path, 'w')`
truncates before writing, so an interrupt in that window leaves a 0-byte file and every
pose is gone; this happened once and cost a flown camera. `load_cameras()` is also
non-raising: a damaged file prints a warning and yields no cameras, instead of taking
down the config cell and with it `CONFIGS` / `CONFIG_BY_ID` and everything downstream.

**Roll is forced to 0.0** on save and on load, so a level horizon is guaranteed without
the user managing it while flying.

Moving the camera still swaps which end of the lane the target parks on, but it stays on
the lane, so a pose is much less likely to push it into a wall — and when it does, the
fallback placements usually absorb it. `snap()` reports whatever is left.

**`snap()` leaves the `behind` scene standing.** It used to leave an empty map: it probed
`behind` with `keep=True`, then probed `front`, and every `spawn_scene` opens with
`clear_vehicles()` — which destroyed the actors it had just kept. `render_pair` now probes
the kept condition **last**. This was the real cause of "the actors disappear so I can't
fly the camera again"; validation failure was not doing it.

### Synchronous mode is off — deliberately

`settings.synchronous_mode = False`. Sync mode freezes the CARLA window between cells,
which makes flying the spectator impossible, and it buys deterministic frame pacing for
a scene where nothing moves. The dual-mode `SYNC` branching that briefly existed was
removed as unused complexity.

Consequences baked into the code:

- The camera listener is `LatestImage`, which keeps **only the newest frame**. A
  `queue.Queue` would grow without bound at ~8 MB/frame while the user flies.
- `advance()` is `world.wait_for_tick()`, not `world.tick()`.
- `capture()` waits for an image stamped at/after the frame it asked for, and takes
  `w2c` from `image.transform` (the pose recorded *with* that frame) rather than a
  separate `camera.get_transform()` RPC.
- **The old "run notebook 01 to the cleanup cell or the server is stuck in sync mode"
  gotcha is gone.** Cleanup only destroys actors now.

### One frame per clip

`FRAMES_PER_CLIP = 1`. The 20 came from `scenario_redcar_behind_bus.ipynb`, which
recorded a real 15 s clip; the sweep kept the clip structure after dropping the reason
for it. Notebook 02 scores the **middle frame only**, so 19 of every 20 PNGs were
written, indexed in the manifest, and never read — 1.65 GB and ~20x the runtime for
nothing. Now ~85 MB.

`frames // 2` with `frames = 1` is `0`, so notebook 02 needs no edit. Do not mix a
20-frame and a 1-frame sweep in the same directory.

**`SETTLE_SECONDS = 0.8` is now load-bearing**: the single frame kept has to be taken
after the vehicles have come to rest, with no later frame to fall back on.

### `runs/sweep/` layout

```
runs/sweep/
  manifest.json                       <- index; the eval notebook reads only this
  <config_id>/                        <- cfg00 .. cfg14
    behind/
      rgb/000000.png                  <- 1920x1080 RGB, one frame
      gt_graphs/000000.json
    front/
      rgb/ ...
      gt_graphs/ ...
  eval_grounding.csv                  <- written by notebook 02
  eval_flip_test.csv
```

`runs/sweep_1/` is an older 20-frame sweep recorded with the **bus-axis** placement, so
its `behind` cars sit beside the bus. Do not treat it as current data.

Any sweep recorded before the oncoming-distractor redesign is also stale: its `front`
clips carry `correct_answer_role: "distractor"` rather than `null`, and its distractor is
in a side slot rather than the opposing lane. Notebook 02 will run on it and produce
numbers that mean something different.

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
`DEPTH_THRESH = 2.0` m for `behind` / `in_front_of`, `LAT_THRESH = 1.5` m for
`left_of` / `right_of`. These were bare literals inside `frame_graph`; they are named
constants now because the scene builder places cars against `LAT_THRESH`. Values
unchanged. `box2d` is absolute pixels, from the 8 projected world-space bounding-box
vertices.

The projection helpers (`build_projection_matrix`, `get_image_point`, `camera_xyz`) are
copied **verbatim** across notebooks. The only change to `frame_graph` is that `ids` is
a parameter rather than a global, because the sweep respawns actors per clip.

### `manifest.json` fields

Trimmed to what notebook 02 actually reads, plus provenance. Removed: `fps` (meaningless
at one frame), `camera_source` (always hand-flown), `correct_answer_id` / `confuser_id`
(02 scores in role space), `validation.attempts` (no retries exist).

Top level: `seed`, `frames_per_clip`, `conditions`, `prompt`, `clips[]`, `skipped[]`.

Each entry in `clips[]`:

| field | meaning |
|-------|---------|
| `config_id` | `cfg00` … `cfg14` |
| `condition` | `behind` or `front` |
| `ids` | `{bus, target, distractor}` → CARLA actor ids **for this clip only** |
| `target_relation` | `behind` or `in_front_of` — where the target is, so a prompt in either direction can be scored |
| `correct_answer_role` | `target` in `behind`, **`null`** in `front` |
| `confuser_role` | the tempting wrong answer: the oncoming car in `behind`, the in-front target in `front` |
| `camera` | `{location, rotation, fov, image_size}` — the pose you flew to |
| `frames` | frame count (1) |
| `warnings` | what `clip_warnings` noticed; recorded, never acted on |
| `rgb_dir` / `gt_dir` | repo-relative paths |
| `config` | the raw config row that generated the clip |

**`correct_answer_role` is `null` for every `front` clip.** Nothing is behind the bus
there. Any consumer that assumes a role string will silently mis-score the negative
control.

**Actor ids are per-clip.** The sweep respawns everything for every clip, so the ids in
the `behind` and `front` clips of the same config differ. Always compare by *role*, never
by raw id — notebook 02 does all of its scoring in role space for this reason.

Each entry in `skipped[]`: `config_id` and `reason` — either no camera, or a spawn that
could not be placed at all.

### There is no validation gate

Nothing is rejected for the user. The layout guarantees the answer — the distractor is in
the opposing lane in both conditions, so the prompt has exactly one referent in `behind`
and none in `front` — which is what the old "exactly one correct answer" check existed to
prove. It is gone with the design that needed it.

The gate was also actively harmful: a rejected pose tore the scene down, so the user could
not keep flying to fix it. (The worse half of that symptom was the `clear_vehicles()` bug
described under `cameras.json`, which emptied the map on *every* snap, pass or fail.)

`clip_warnings` still computes two things, prints them under each shot, and writes them
into the manifest — but nothing branches on them:

1. **In frame** — all three vehicles project in front of the camera and their full 2D
   boxes lie inside `[0, WIDTH] x [0, HEIGHT]`.
2. **Separated** — no pair of 2D boxes overlaps by more than **0.5 IoU**.

`reads_as()` is deleted along with check 3. Its lesson is still worth keeping, because it
explains why the raw `left_of` / `right_of` edges are diagnostics and not evidence: the GT
`behind` edge is pure depth, and a car 12 m up the road picks up
`12*sin(angle off image centre)` metres of camera-frame lateral offset for free — 1.25 m
at 6°, 1.67 m at 8°, 3.11 m at 15°. Past ~7° off centre, which is most hand-flown shots,
a spurious lateral edge fires. `aligned()` survives because it is measured on `box2d` and
is immune to this; `answer_report` uses it to say whether a car "reads as behind the bus"
or is "off to the side".

**Review is a human step**, section 12a: a contact sheet of all 30 shots with GT boxes and
the `% clear of the bus` number, flagging anything with warnings. A click-to-label tool
was considered and rejected — `box2d` is projected from CARLA's own 3D bounding-box
vertices, so it is exact, and clicking would only re-derive it by hand. The thing that
actually needs eyes is whether the target is *visible*.

Every spawn is torn down in a `finally`, so a failing config cannot leak actors.

### The three scoring definitions (notebook 02)

Middle frame (`frames // 2`, which is frame 0 at one frame per clip). Detections are
matched to GT `box2d` by IoU with a **0.5 threshold** (`>=`, so exactly 0.50 counts);
`match_role` returns the role of the highest-IoU GT box, or `None`.

| metric | definition |
|--------|------------|
| **selection accuracy** | `behind`: role of the **top-scoring** detection == `target`. `front`: `chosen_car_role is None` — the only correct answer there is *no car* |
| **false positive** | `front` only: it returned a car anyway |
| **distractor confusion** | role of the **top-scoring** detection == `confuser_role` |
| **anchor recall** | **any** detection matches the bus |

The two conditions measure different things and the aggregate table's `behind` and `front`
columns are not comparable. In `front` a colour-only prompt should score near 0 by
construction; the question is whether the relational clause lifts it.

`front` is scored on `chosen_car_role`, not `best_role`, so a stray box on a building does
not count as a correct abstention.

`chosen_car_role` is the top-scoring detection that matches *a car* (bus matches skipped).
It exists so that a run where the bus outranks every car does not silently count as
"no flip".

Three prompt variants, increasing relational content: `car`, `red car`,
`red car behind the bus`.

#### The flip test — the headline number

Per config and prompt, does `chosen_car_role` differ between the `behind` and `front`
conditions?

- **flip rate 1.0** — the detector tracks the relation.
- **flip rate 0.0** — the relational clause is **inert**; the detector picks the same car
  either way and "behind the bus" is doing no work. This is the result the experiment is
  built to detect.
- **correct-flip rate** — flipped *and* landed correctly in both conditions: `target` in
  `behind`, and **no car at all** in `front`, since nothing there is behind the bus. A
  detector that flips for the wrong reason — swapping to the oncoming distractor rather
  than abstaining — scores on flip rate but not here.

### Gotchas

- `run_inference(image, text_prompt)` in notebook 02 is a **placeholder** — it must
  return boxes as `[x1, y1, x2, y2]` in **absolute pixels**. HF's
  `GroundingDinoProcessor` returns normalised `cxcywh`; convert inside the stub.
- Notebook 02 has an optional fake-detector cell (`USE_FAKE_DETECTOR`) for smoke-testing
  the scoring plumbing without a model. It simulates an inert detector.
- Re-running the sweep cell **deletes and rewrites** `runs/sweep/`. It does not touch
  `cameras.json`, which lives at the repo root for that reason.
- `tour()` calls `input()`, so it blocks the kernel until answered. The "press a key" is
  Enter *in the notebook*, not in the CARLA window — reading the game window's keyboard
  would need a global hook and a new dependency. It now only forces a retry on a blocked
  spawn; everything else is the user's call.
- Notebook 01 cell 17 binds `_park` (a `Location`) at module scope. The slot helper is
  called `parked_at` for that reason — do not rename it to `_park` or the camera cell
  will clobber it depending on run order.
- **The camera cell destroys existing `sensor.camera.rgb` actors before spawning.**
  Re-running it used to leak a camera per run, each still streaming ~8 MB frames into a
  listener nobody reads. `aim_camera` also checks `camera.is_alive` and raises, because
  `reload_world()` destroys the sensor without rebinding the name and calling into a
  destroyed sensor can take the kernel down instead of raising.
- **Jupyter will overwrite edits made on disk.** The browser tab holds its own copy of
  the notebook; if it autosaves after an external edit, that edit is silently gone. After
  any out-of-band change to a `.ipynb`, use *File → Reload Notebook from Disk* before
  touching it. This has already cost one round of changes.
- `pandas` and `matplotlib` are not currently installed in the `carla` conda env
  (`C:\Users\azhar\miniconda3\envs\carla`); it has carla + numpy only.
