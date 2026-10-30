# Porting ARTI-D1 to Isaac Lab 3.x (Isaac Sim 6.2 / Newton)

Branch: `isaaclab3`. Source: Isaac Sim 5.1 + Isaac Lab 2.3.x (PhysX only, AppLauncher scripts).
Target: Isaac Lab main @ `0ca25c56` (3.x dev, `/opt/isaac/IsaacLab-main`), Isaac Sim 6.2, Newton 1.8 dev,
Warp 1.17, MuJoCo-Warp 3.15, Python 3.12, torch 2.11+cu130, numpy 2.x, skrl 2.1.1.
Python env: `source /opt/isaac/IsaacLab-main/.venv/bin/activate` (isaaclab 42.2.0, isaaclab_physx 10.0.2,
isaaclab_newton 12.0.1, isaaclab_tasks 29.1.0, isaaclab_rl 5.3.0).

Reference port: `/opt/isaac/projects/arti_h1_local` branch `isaaclab3` (commits 3df3999..c932d9d, Steps 1-3 done;
its Newton work, i.e. the in-memory USD collider patch in `robots/arti_h1.py` and `scripts/porting/compare_backends.py`,
is **uncommitted** in that working tree). Its PORTING_NOTES.md is the source for every row marked "as H1" below.

Ground truth checked in the Isaac Lab main source for this port: `isaaclab/utils/math.py` (XYZW),
`assets/articulation/articulation_cfg.py` + `ordering.py` (joint/body ordering), `isaaclab_newton/.../actuator_control.py`
+ `articulation_data.py` (ordering on Newton), `docs/source/how-to/transfer_policies_between_physx_and_newton.rst`,
`docs/source/how-to/prepare_asset_for_newton.rst`, `isaaclab_physx/docs/CHANGELOG.rst` 8.0.0, newton
`solvers/mujoco/solver_mujoco.py` (what MJWarp consumes), `isaaclab_assets/robots/dr_legs.py` + `contrib/dr_legs`
+ the DR-Legs USD (Kamino loop closure).

## 1. Shared vs D1-specific (diff against arti_h1_local `main`, names normalised h1 -> d1)

Identical modulo the robot name (0 differing lines): `pyproject.toml`, `source/arti_d1/{pyproject.toml,
config/extension.toml}`, `arti_d1/__init__.py`, `ui_extension_example.py`, `tasks/**/__init__.py` (incl. gym registration),
both skrl yamls, `tasks/utils/motion_loader.py`, `scripts/skrl/train.py`, `scripts/automation/{arti_d1.env,auto_record.sh,
README.md}`, `scripts/tuning/{check.sh,auto_video_loop.sh,make_stand_clip.py,tb_summary.py,video_blank_check.py}`, docker.

Same code, D1 constants only (<= 30 lines): `arti_d1_env.py` (D1 generalised 12 -> `num_joints`, joint-vel noise slice
`9+6n : 9+7n`, configurable stale-obs keywords / ankle names, **no** foot-width reward), `mti3_imu.py` (D1 has **no**
IMU tilt random walk), `scripts/skrl/play.py` (camera, command sequence), `replay_demo.py`, `ilc_nominal.py`,
`setup.py` (`*.stl`), `auto_train.sh`, `record_video.sh`, `arti-autotrain@.service`.

D1-specific (new work): `robots/arti_d1.py` (Dynamixel PD model), `arti_d1_env_cfg.py` (all task constants,
no pelvis-CoM DR, no IMU tilt walk), `check_onnx_in_env.py`, `check_controller_obs.py`, `eval_gait.py`,
`analyze_robot_log.py`, `pt_to_onnx.py`, `gen_motions.py`, `reference_knee_lift.py`, README, `control/run.yaml`,
plus everything about joint order, the Newton preset for a 5.8 kg robot and the Kamino notes.

Dry run: a 3-way merge (base = H1 `main`, theirs = H1 `isaaclab3`, ours = D1, names normalised) applies the H1 port
cleanly to 19 of 25 files; conflicts only in README (3), run.yaml (1), check_controller_obs (1), check_onnx_in_env (2),
eval_gait (2), robots/arti_d1.py (1), arti_d1_env.py (2). Step 2 uses that merge, then reviews every hunk
(a clean merge is not proof: e.g. the cfg must not pick up H1-only DR terms).

## 2. Headline findings

1. **Quaternion order WXYZ -> XYZW** (as H1; confirmed `isaaclab/utils/math.py` "xyzw format"). The env converts
   with `[:, [1, 2, 3, 0]]` at `arti_d1_env.py:457,593,944,1210` and back with `[:, [3, 0, 1, 2]]` at `:1257`;
   `mti3_imu.py:323` likewise. All swaps are **deleted** (project math is already XYZW).
2. **Joint order is backend-dependent; Isaac Lab 3.x has a built-in fix.** `ArticulationCfg.joint_ordering` /
   `body_ordering` accept an explicit complete name list; every `ArticulationData` property, writer, `joint_names`,
   `find_joints` and the actuator path (PhysX and Newton `actuator_control.py`) then use that public order.
   An order equal to the backend order is normalised to `None` (zero-conversion), so on PhysX it is a strict no-op.
   The `"physx"` alias now means **OVPhysX** order and is resolved via Newton's USD parser (isaaclab_physx 8.0.0
   CHANGELOG warns that checkpoints trained on Isaac Sim PhysX must use an explicit list) -> use the explicit list.
   The docs require setting **both** joint and body ordering.
3. **The current (PhysX) order is breadth-first, the USD authored order is depth-first** (USD audit):
   authored = `left_arm_m8, left_arm_m8crank_plate, left_hip_yaw, left_hip_roll, ...` vs PhysX/Dynamixel =
   `left_arm_m8, left_hip_yaw, right_arm_m8, right_hip_yaw, left_arm_m8crank_plate, ...`. MJWarp's view is a
   different traversal, so without an ordering map Newton would almost certainly scramble actions/observations.
4. **Every joint-indexed path in the project uses `robot.data.joint_names` (= public order)**: motion-loader DOF map
   (by name), action scales, PD/actuator model per joint type, randomisation, policy/AMP obs, and the nominal-gait
   table `assets/motions/arti_d1_nominal_gait_command.npy` (64 x 14, **stored in PhysX order**, not by name).
   With `joint_ordering = Dynamixel list` all of these are correct on every backend with no code change; the
   startup assert guards it. `pt_to_onnx.py:ISAAC_JOINT_NAMES` and `check_controller_obs.py:POLICY` hard-code the same
   list -> they import the single constant instead.
5. **USD needs the same in-memory Newton fixes as H1**: `configuration/arti_d1_base.usd` references
   `/visuals/trunk` that the physics layer lacks (Newton raises on composition errors), and the 15 colliders are
   `CollisionAPI`+`convexHull` on `node_STL_BINARY_` Xforms whose child `mesh` holds the geometry (Newton ignores
   non-gprim colliders). H1's uncommitted `spawn_*_usd` patch (strip ref; copy APIs to the child mesh, Newton only,
   in memory, file untouched) applies with `_DANGLING_VISUAL_REFERENCE_BODIES = ("trunk",)`.
6. **Velocity limits matter more on D1 than on H1.** `joint_velocity_limit` (= Dynamixel no-load speed 3.14 / 4.82
   rad/s) is a PhysX solver clamp; **MJWarp drops it** (Kamino honours it) - prepare_asset_for_newton.rst. The D1
   reference gait runs up to 3.03-3.3 rad/s, i.e. at the clamp. The back-EMF term (`kd = stall / no_load_speed`) is
   unchanged on both backends and already drives torque to ~0 near no-load speed, so the clamp is a secondary bound,
   but MJWarp can exceed it. See Q1.
7. **Newton actuator mapping** (newton `solver_mujoco.py`): implicit PD -> MuJoCo position actuator from
   `joint_target_ke/kd` (kd integrated implicitly with `integrator="implicitfast"`), `joint_effort_limit` ->
   `jnt_actfrcrange` (stall torque clip), armature -> `dof_armature` (the env's per-env randomised armature write
   reaches MJWarp through `write_joint_armature_to_sim_index`). `actuators.applied_effort` is the same
   `clip(kp*e + kd*ed)` estimate on both backends (used by the torque/power rewards).
8. **Masses**: total 5.79 kg (pelvis 3.69). Light links: arm crank 7.9 g (I_xx 2.6e-7 kg m^2), arm plate 61 g.
   The `trunk` frame body is 1e-9 kg with zero inertia on a fixed joint (`trunk_frame`); Isaac Lab Newton does not
   collapse fixed joints, Newton's inertia validation has no mass/inertia bound set by default (it only symmetrises),
   so it stays negligible - to be printed in Step 4. Armature on the arm joints (0.002 x U(0.3,1.5)) is ~4 orders of
   magnitude above the crank's own inertia, which is what keeps the 18.9 Nm/rad / 0.85 Nm s/rad arm PD stable at
   dt = 1/240 on MJWarp; it is kept, unchanged.
9. Everything else is as H1: `ProxyArray` + `.torch`; `_index` writers; scene-cfg-declared assets
   (`clone_environments`, `clone_in_fabric` removed); `Imu` -> `Pva` (`gravity_bias` removed, accel unused in obs);
   `sim.physx` -> `sim.physics` with `PresetCfg`; `isaaclab train` supports AMP once `skrl_cfg_entry_point` is
   registered; external tasks via `isaaclab.tasks` entry point; `omni.ext` import must go; skrl yamls already
   skrl-2.x (identical files to H1, verified there key by key).

## 3. API table (D1 line numbers; class: U unchanged, R renamed/moved, X removed, B behaviour changed, P PhysX-only, D deprecated)

### tasks/direct/arti_d1/arti_d1_env.py

| Line | Old API (2.3) | New API (3.x) | Class | Risk |
|---|---|---|---|---|
| 12, 281 | `Articulation(self.cfg.robot)` in `_setup_scene` | robot in scene cfg, `self.robot = self.scene["robot"]` | B | low |
| 287-289 | `scene.clone_environments(copy_from_source=False)`, `scene.filter_collisions` | cfg-driven cloning | X | med |
| 291, 294 | `scene.articulations[...]=`, `scene.sensors[...]=` | automatic from scene cfg | B | low |
| 297 | `DomeLightCfg.func(...)` | `AssetBaseCfg(spawn=DomeLightCfg)` in scene | B | low |
| 59, 68 | `write_joint_position_limit_to_sim` (only if `hip_roll_limit`/`real_joint_limits`, both off for D1) | `write_joint_position_limit_to_sim_index` | D | low |
| 73 | `data.soft_joint_pos_limits` | same, `.torch` | B (ProxyArray) | low |
| all `data.*` reads | `joint_pos/vel/acc`, `joint_stiffness` (389), `projected_gravity_b` (680), `root_ang_vel_b` (683), `root_lin/ang_vel_w` (1028) | same names, `.torch` | B | low |
| 233, 345, 1168-1169 | `data.default_joint_pos/vel` | same, `.torch` | B | low |
| 355 | `robot.set_joint_position_target(target)` | `robot.actuators.target_command.set_position_index(value=)` | D | low |
| 372 | `write_joint_armature_to_sim(armature)` | `write_joint_armature_to_sim_index(armature=)` (MJWarp: `dof_armature`) | D | low |
| 408, 438, 449, 456, 592, 854-855, 944, 1133, 1142 | `data.body_quat_w` (WXYZ) | XYZW | **B** | **high** |
| 457, 593, 944, 1210, 1257 | `[:, [1,2,3,0]]` / `[:, [3,0,1,2]]` swaps | delete | **B** | **high** |
| 14, 409, 437, 448, 872, 875, 1132, 1141 | `quat_apply` | same, XYZW inputs | B | med |
| 416 | `instantaneous_wrench_composer.add_forces_and_torques` | `add_forces_and_torques_index` | D | low (Newton: verify `is_global` torque frame) |
| 635-642, 761-762 | `data.applied_torque` | `robot.actuators.applied_effort` | D | low |
| 1159 | `robot._ALL_INDICES` | now `wp.array` -> `torch.arange(num_envs)` | B | med |
| 1166, 1255 | `data.default_root_state` | `default_root_pose` + `default_root_vel` | D | low |
| 1183-1185 | `write_root_link_pose_to_sim`, `write_root_com_velocity_to_sim`, `write_joint_state_to_sim` | `*_to_sim_index`, joint state -> position + velocity writers | D | low |
| 1293, 1319 | `@torch.jit.script` | still used by humanoid_amp | U | low |
| implicit everywhere | joint index = PhysX BFS order | public order pinned by `joint_ordering` + startup assert | **B (backend)** | **high on Newton** |

### tasks/direct/arti_d1/arti_d1_env_cfg.py

| Line | Old API | New API | Class | Risk |
|---|---|---|---|---|
| 6, 89-136 | `mdp.randomize_rigid_body_material/mass`, `randomize_actuator_gains`, `push_by_setting_velocity` | same names; backend dispatch inside. Newton material DR ignores `dynamic_friction_range`, `num_buckets`, `make_consistent` (one coefficient) | U (B on Newton) | med on Newton |
| 8, 394-401 | `ImplicitActuatorCfg(effort_limit=, velocity_limit_sim=)` | `actuator_effort_limit` + `joint_effort_limit`, `joint_velocity_limit` | D | low |
| 371 | `imu_orientation_wxyz` | keep field (WXYZ), convert once when building `PvaCfg.OffsetCfg.rot` (XYZW) | B | med |
| 382-389 | `SimulationCfg(physx=sim_utils.PhysxCfg(solver_type=1, bounce_threshold_velocity=0.2))` | `SimulationCfg(physics=ArtiD1PhysicsCfg())` (`PresetCfg`) | R | low |
| 391-402 | top-level `robot` | `scene.robot` (+ `joint_ordering`/`body_ordering`) | B | low |
| 404-416 | `TerrainImporterCfg(physics_material=RigidBodyMaterialCfg)` | `PhysxRigidBodyMaterialCfg` (used by both backends, humanoid pattern) | D | low |
| 418-423 | `InteractiveSceneCfg(replicate_physics=True, clone_in_fabric=False)` | drop both (`clone_in_fabric` removed -> TypeError) | X | low |

### robots/arti_d1.py

| Line | Old API | New API | Class | Risk |
|---|---|---|---|---|
| 78-86 | `RigidBodyPropertiesCfg` | `PhysxRigidBodyCfg` (values verbatim) | D | low |
| 87-90 | `CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0)` | `PhysxCollisionCfg` (PhysX-only; never reached the instanced colliders in 2.x either, see H1 fix log) | D/P | low |
| 91-97 | `ArticulationRootPropertiesCfg(...)` | `[PhysxArticulationCfg(...), NewtonArticulationCfg(self_collision_enabled=False)]` | D | low |
| 76-77 | `UsdFileCfg(usd_path=)` | + `func=spawn_arti_d1_usd` (in-memory Newton fixes, finding 5) | B | med |
| 105-113 | `ImplicitActuatorCfg(effort_limit=, velocity_limit_sim=)` | renamed fields, **same numbers** | D | low |
| new | - | `joint_ordering = ARTI_D1_POLICY_JOINT_ORDER` (Dynamixel IDs 1-14), `body_ordering` = PhysX body order (explicit) | new | **high** |

### tasks/utils/mti3_imu.py (as H1)

| Line | Old | New | Class | Risk |
|---|---|---|---|---|
| 36-57 | `ImuCfg(offset=OffsetCfg(rot=wxyz), gravity_bias=...)`, hand-built sensor | `PvaCfg` in scene cfg, `rot` XYZW | R + X | **high** |
| 322-334 | `imu.data.quat_w` WXYZ + swap; fallback `body_quat_w[:, [1,2,3,0]]` | `pva.data.quat_w.torch` (XYZW), `.torch` everywhere | B | high |
| 19-24, 67-128, 221 | inactive `imu_backend="command"` path (`omni.kit.commands`, `isaacsim.sensors.*`, `pxr.Gf`) | left as is (inactive, Kit-only) | P/D | low |

### Packaging / scripts (as H1)

| File | Old | New | Class | Risk |
|---|---|---|---|---|
| `setup.py` | no entry point, py 3.10/3.11 | `entry_points={"isaaclab.tasks": ["arti_d1 = arti_d1.tasks"]}`, py >= 3.12 | B | med |
| `arti_d1/__init__.py` | imports `ui_extension_example` (`omni.ext`) | remove (entry points load before Kit) | X | high |
| `tasks/direct/arti_d1/__init__.py` | Amp task: `skrl_amp_cfg_entry_point` only | + `skrl_cfg_entry_point` -> AMP yaml, `default_agent: skrl`; IDs unchanged | B | low |
| `scripts/skrl/train.py` | AppLauncher + `hydra_task_config` (removed) | thin wrapper on `isaaclab_rl.entrypoints.backends.train_skrl.run()` + AMP diagnostics hook (H1 file verbatim) | X | high |
| `scripts/skrl/play.py` | AppLauncher, RecordVideo, `write_root_velocity_to_sim` | `play_skrl.py` pattern (H1 file + D1 camera/commands) | X/D | med |
| `scripts/tuning/*.py` (eval_gait, check_onnx_in_env, check_controller_obs, ilc_nominal, replay_demo) | WXYZ swaps, `cfg.robot`, old writers | XYZW, `cfg.scene.robot`, `.torch`, `_index` | B | **high** (silent quat) |
| `pt_to_onnx.py`, `verify_onnx.py`, `analyze_robot_log.py`, `reference_*`, `gen_motions.py`, `make_stand_clip.py`, `tb_summary.py` | no Isaac Lab API | unchanged; `pt_to_onnx` takes the joint list from the robot module | U | none |
| `record_video.sh`, `check.sh`, `auto_record.sh`, `auto_train.sh`, `arti_d1.env`, `control/run.yaml` | `--headless`, Windows paths, docker `python.sh`, `env.sim.physx.*` | as H1: new CLI, `physics=$PHYSICS`, `env.sim.physics.*` | B | high (server) |

## 4. Joint and body order: design

- `robots/arti_d1.py` gets `ARTI_D1_POLICY_JOINT_ORDER` (Dynamixel IDs 1-14, the table in the task brief, = current
  PhysX order = `pt_to_onnx.ISAAC_JOINT_NAMES` = humanoid_controller `kPolicyJointNames`) and
  `ARTI_D1_BODY_ORDER` (current PhysX body order, recorded from the Step 3 PhysX run before it is written in).
- `ARTI_D1_CFG.joint_ordering = ARTI_D1_POLICY_JOINT_ORDER`, `body_ordering = ARTI_D1_BODY_ORDER`
  (Isaac Lab's own remap; PhysX: normalised to `None`, zero cost, bit-identical; Newton: permutation map in every
  data property / writer / actuator call).
- Startup assert in `ArtiD1AmpEnv.__init__`: prints backend, `robot.data.joint_names`, the backend order
  (`backend_joint_names`), and checks `joint_names == ARTI_D1_POLICY_JOINT_ORDER`, `find_joints(ORDER,
  preserve_order=True)[0] == range(14)`, and that per-joint stiffness / damping / effort limit / velocity limit /
  default pos read back in that order equal the dict values by name (catches a mapping that orders names but
  not data). Fails hard otherwise.
- The nominal table `.npy` stays as is (it is in this order); the motion loader maps by name; action-scale and
  actuator-model tensors are built by name from `joint_names`. No manual permutation code in the env.

## 5. Physics presets (plan)

`ArtiD1PhysicsCfg(PresetCfg)`, default `isaacsim_physx`:
- `isaacsim_physx` = `PhysxCfg(solver_type=1, bounce_threshold_velocity=0.2)` (exact 2.x), dt 1/240, decimation 4.
- `ovphysx`, `physx` (auto) as H1.
- `newton_mjwarp`: start from H1's (MJWarp `implicitfast`, pyramidal cone, `impratio=1`, 1 substep,
  `NewtonShapeCfg(margin=0, ke, kd)`); `njmax`/`nconmax` and contact ke/kd re-derived for 5.8 kg / D1 feet with
  a zero-action standstill A/B against PhysX (H1's `compare_backends.py`, adapted to 14 joints). dt stays 1/240
  (policy-facing timing must match); only substeps may change.
- `newton_kamino`: stub `NewtonCfg(solver_cfg=KaminoPADMMSolverCfg(...))`, untuned (arms are open chains).

## 6. Decisions (proposed, pending OK)

- D1. Reuse H1's port by 3-way merge, then review every hunk; D1 deltas (num_joints generalisation, no foot-width
  reward, no IMU tilt walk, no CoM DR) are kept.
- D2. Joint/body order through `ArticulationCfg.joint_ordering`/`body_ordering` explicit lists + startup assert
  (section 4), rather than a hand-written remap in the env. This is the "find_joints preserve_order or equivalent"
  option; it also covers PD gains, limits and randomisation, which a hand remap in the env would not.
- D3. Default preset stays `isaacsim_physx`. Newton-only values live only in the `newton_*` presets and in the
  Newton-gated USD patch; PhysX numbers in `robots/arti_d1.py` stay byte-identical.
- D4. Quaternions: delete the swaps; keep `imu_orientation_wxyz` field name/values, convert once.
- D5. Training: `isaaclab train --rl_library skrl` works after registering `skrl_cfg_entry_point`;
  `scripts/skrl/train.py` = H1 wrapper (keeps AMP diagnostics); `scripts/skrl/play.py` = H1 port with D1 constants.

## 7. Open questions

- Q1. MJWarp drops `joint_velocity_limit` (finding 6). Proposed: document only, keep PhysX unchanged, and report in
  the Newton smoke test how often |qd| exceeds the Dynamixel no-load speed. (Alternatives: enforce a speed clamp in
  the env's actuator model, which would change PhysX behaviour too; or use Kamino later.)
- Q2. As H1 (answered yes there): port `scripts/tuning/*.py` minimally, and port `auto_train.sh` to the new CLI
  while leaving `docker/` on the old image? Assumed yes unless told otherwise.

## 8. Risks for Steps 3-4

- Newton: ordering map must be verified (assert); colliders need the in-memory patch; `nconmax`/`njmax` for
  15 convex hulls; sole penetration with default contact ke on a 5.8 kg robot; material DR reduced to one
  coefficient; velocity clamp missing; `arm_m8crank_plate` default -0.2268 rad sits 0.09 mrad inside its
  -0.22689 lower limit (MuJoCo limit margin/solref may push it at reset); balance-support torque (`is_global=True`)
  frame on Newton.
- `trunk` 1e-9 kg zero-inertia welded body: check resolved mass/inertia on Newton.
- Kit prints "possibly invalid inertia tensor {1,1,1}" for frame bodies under Newton (seen on H1): Kit-side only.

## 9. Ordered work plan

1. Branch + this file (Step 1).
2. Step 2: 3-way merge of the H1 port, conflict resolution, D1 review; packaging + entry point; robot schema
   fragments + actuator renames + ordering + USD patch; `ArtiD1PhysicsCfg` + `ArtiD1SceneCfg`; env port + assert;
   IMU; registration; train/play; tuning + automation scripts; README versions. Verify `import arti_d1`, gym registry,
   `isaaclab list_envs --keyword Arti-D1 --show_presets`.
3. Step 3: PhysX smoke (16 envs, 3 iters), record PhysX body order into `ARTI_D1_BODY_ORDER`, AMP/reset
   consistency check (H1's `check_amp_consistency.py`).
4. Step 4: Newton smoke + standstill A/B + 4-env `--viz newton_gl` play; Newton tuning only in presets.
5. Step 5: final report.

## 10. Future arm loop closure (m8 crank <-> plate) - not implemented

Reference: Isaac Lab `isaaclab_assets/robots/dr_legs.py`, `isaaclab_tasks/contrib/dr_legs/*` and the DR-Legs USD
(`newton-assets/disneyresearch/dr_legs/usd/dr_legs.usda`).

Today each arm is an open chain `pelvis -(arm_m8)- crank -(arm_m8crank_plate)- plate`, both joints driven.
If the real crank-rocker linkage is modelled:

- **USD**: keep the tree joints as the articulation spanning tree. Add the missing links (e.g. the connecting rod)
  with their own revolute joints in the tree, and close the loop with one extra joint authored with
  `uniform bool physics:excludeFromArticulation = 1` (DR-Legs: `j5_*_i`, `j9_*_{i,o}`, 6 loop joints, all
  `PhysicsRevoluteJoint`, `collisionEnabled = 0`, unlimited). Excluded joints are not DOFs, do not appear in
  `joint_names` and are enforced by the solver as bilateral constraints (Kamino: maximal coordinates).
- **Assembled pose**: author every `localPos0/1`, `localRot0/1` so that the loop is closed **at all joint
  coordinates = 0** (`dr_legs.py`: "Closed-loop FK is only valid at the assembled reference"; its init state is
  `joint_pos={".*": 0.0}`). The D1 rest pose (plate at -13 deg) then has to be expressed as a non-zero default of the
  driven joint after the loop is assembled at zero, or the zero reference moved in the USD.
- **Actuators**: only the physically driven joint keeps the Dynamixel PD; the passive linkage joints get an explicit
  `ImplicitActuatorCfg(stiffness=0.0, damping=0.0, armature=0.0, friction=0.0, ...)` so no USD drive defaults
  remain (DR-Legs "passive_joints" group). The policy action / ONNX would then lose `arm_m8crank_plate` (or keep it
  as the driven joint with `arm_m8` passive, depending on the real mechanism) -> **ONNX layout change** and
  `ARTI_D1_POLICY_JOINT_ORDER` change; `joint_ordering` must list every non-excluded joint, passive ones included,
  while the action term selects only the driven ones.
- **Solver**: only `newton_kamino` enforces such loops (DR-Legs presets: Kamino default; PhysX and MJWarp variants
  use different gains/armature). PhysX articulations ignore `excludeFromArticulation` joints as articulation DOFs
  but simulate them as maximal-coordinate joints (separate constraint); MJWarp support for the loop joint is not
  verified here (DR-Legs only ships its walk task tuned for Kamino). So the `newton_kamino` stub becomes the primary
  preset for a closed-loop arm.
- Keep `self_collision_enabled` consistent with the rod/plate overlap (DR-Legs enables self collision with
  `collisionEnabled = 0` on the loop joints).

## 11. Step 2 log (port)

User answers: Q1 yes (document the MJWarp velocity-limit gap, measure it, PhysX unchanged), Q2 yes (tuning scripts
ported minimally, `auto_train.sh` on the new CLI, `docker/` untouched).

Method: 3-way merge of the H1 port (base H1 `main`, theirs H1 `isaaclab3`, ours D1 `main`, names normalised),
then every hunk reviewed. Conflict resolutions:
- `robots/arti_d1.py`: D1 joint tables kept; H1's in-memory USD patch taken with `_DANGLING_VISUAL_REFERENCE_BODIES =
  ("trunk",)` (D1's base layer has only that dangling arc; H1 also had the feet).
- `arti_d1_env.py` (2): D1 docstrings/comments, H1's `.torch` lines.
- `eval_gait.py`: D1's per-foot `sole_offsets` corner line with `.torch`; H1's `--mode standpush` block not taken
  (feature never existed on D1). `check_onnx_in_env.py`: H1's max-tilt stats not taken (same reason).
- `control/run.yaml`: D1 content (its args have no `env.sim.physx.*` override). README: 3.x install/CLI, D1 commands.
- Clean merges re-checked: the cfg got only the 3.x structure (`ArtiD1PhysicsCfg`, `ArtiD1SceneCfg`, Pva slot,
  light); no H1-only DR term (pelvis CoM, IMU tilt walk, foot width) leaked in. No `data.*` read left without
  `.torch`, no 2.x writer, no quaternion swap except the boundary conversions (robot-log WXYZ in
  `check_onnx_in_env --save_log` / `check_controller_obs`, inactive Kit `command` IMU backend).

D1-specific additions:
- `robots/arti_d1.py`: `ARTI_D1_POLICY_JOINT_ORDER` (Dynamixel IDs 1-14) and `ARTI_D1_CFG.joint_ordering` set to it.
  `ArtiD1SceneCfg.robot = ARTI_D1_CFG.replace(...)` keeps it.
- `ArtiD1AmpEnv._check_joint_order()` (called right after `DirectRLEnv.__init__`): prints the backend, the backend's
  own order and the public order, and raises unless `joint_names == ARTI_D1_POLICY_JOINT_ORDER`,
  `find_joints(order, preserve_order=True) == 0..13`, effort / velocity limits and default pose match the cfg by
  name, and stiffness / damping lie inside the start-up DR range of the by-name values (leg and arm ranges do not
  overlap, so a swapped arm/leg joint is caught despite the +-30 % gain DR).
- `body_ordering`: not set in Step 2 (needs the PhysX body order from a run; Step 3). The env resolves every body by
  name (`find_bodies`), so body order is not policy-facing; it is set anyway because the docs require both.
- `scripts/porting/compare_backends.py`: D1 joints, `action_space` zero action, `trunk`/arm-crank mass + inertia,
  share of joint samples above the no-load speed (Q1). `check_amp_consistency.py`: AMP groups from `num_joints`,
  no `pelvis_com` event.

Deviation from the Step 1 plan:
- `pt_to_onnx.py:ISAAC_JOINT_NAMES` and `check_controller_obs.py:POLICY` keep their own lists. `pt_to_onnx.py` only
  needs torch (importing `arti_d1.robots` would pull Isaac Lab / Newton into the export), and `POLICY` deliberately
  mirrors humanoid_controller's `kPolicyJointNames`, which makes it an independent cross-check. All three lists
  are identical (checked).

Verified: `uv pip install -e source/arti_d1`; `import arti_d1`; gym IDs `Template-Arti-D1-Amp-Direct-v0`
(skrl_cfg + skrl_amp_cfg -> AMP yaml, default_agent skrl) and `Template-Arti-D1-Direct-v0` (skrl_cfg -> PPO yaml);
`isaaclab list_envs --keyword Arti-D1 --show_presets` lists `isaacsim_physx [default]`, `newton_kamino`,
`newton_mjwarp`, `ovphysx`; every changed .py byte-compiles, every changed .sh passes `bash -n`.

Behaviour changes forced by the API (as H1): B2 `replay_demo.py --video` removed; B3 `play.py --video` without
`_BlankFrameGuard`, no `--use_pretrained_checkpoint`; B4 `ovphysx` cannot express `solver_type` /
`bounce_threshold_velocity`; B5 the Pva accelerometer has no +g (not in any observation).
