# arti_d1

AMP walking policy for the ARTI-D1 biped (Isaac Lab + skrl), set up like
[arti_h1](https://github.com/SeonilChoi/arti_h1): reference motions from motion_generator, residual policy on a
clock-driven nominal gait, sim-to-real randomisation. The cfg defaults are the training setup, so training needs no
extra arguments. Trained policies are exported to ONNX and deployed with
[humanoid_controller](https://github.com/SeonilChoi/humanoid_controller) (`ARTI-D1` executable,
`config/robot/ARTI-D1/agent.onnx`).

## Robot

- BDX-like layout, 14 joints: per leg hip yaw / roll / pitch, knee, ankle (Dynamixel XH540-W270), per arm `arm_m8` and
  `arm_m8crank_plate` (Dynamixel XM430-W350). ~5.8 kg, trunk 0.255 m.
- Policy / action / observation joint order = Isaac Lab articulation order = Dynamixel IDs 1-14:

  | ID | joint | ID | joint |
  |---:|---|---:|---|
  | 1 | left_arm_m8 | 8 | right_hip_roll |
  | 2 | left_hip_yaw | 9 | left_hip_pitch |
  | 3 | right_arm_m8 | 10 | right_hip_pitch |
  | 4 | right_hip_yaw | 11 | left_knee |
  | 5 | left_arm_m8crank_plate | 12 | right_knee |
  | 6 | left_hip_roll | 13 | left_ankle |
  | 7 | right_arm_m8crank_plate | 14 | right_ankle |

- Actuators (`robots/arti_d1.py`): Dynamixel position mode as a joint PD, from the datasheet and the factory Position
  P Gain 800: kp = P/128 * 4096/2pi / 885 * stall torque (legs 48.8, arms 18.9 Nm/rad), kd = stall torque / no-load
  speed (back-EMF; legs 3.38, arms 0.85 Nm s/rad), effort limit = 12 V stall torque (10.6 / 4.1 Nm). Not identified on
  the robot yet: gains are randomised +-30 % and the actuator model (delay / armature / Coulomb) uses generic values.
- Joint limits: URDF (`assets/models/arti_d1/arti_d1.urdf`, hip_roll -18 / +12 deg mirrored).
- IMU: Xsens MTi model, assumed at the trunk frame (`ARTI_D1_IMU_OFFSET_FROM_TRUNK`); update it with the real mount.

## Requirements

Isaac Sim 5.1.0, Isaac Lab 0.54.3, skrl 2.0.0, PyTorch 2.7.0 (cu128); onnxruntime / onnx for export.

## Install

The robot USD files are stored with Git LFS, so install it first.

```bash
git lfs install
git clone https://github.com/SeonilChoi/ARTI-D1_IsaacLab.git arti_d1 && cd arti_d1
python -m pip install -e source/arti_d1      # python with Isaac Lab installed
python -m pip install onnxruntime onnx        # only for ONNX export
```

## Train

```bash
python scripts/skrl/train.py --task Template-Arti-D1-Amp-Direct-v0 --algorithm AMP --headless
```

- Output: `logs/skrl/arti_d1_amp_walk/<date>_amp_torch/` (checkpoints, `params/env.yaml`, `params/agent.yaml`).
- Resume: add `--checkpoint <agent_N.pt> env.curriculum_step_offset=N`.
- Progress summary: `PY=<isaac lab python> bash scripts/tuning/check.sh`.

Setup:
- Reference motions `assets/motions/arti_d1`: 1000 motion_generator clips (1.25 s cycle, 0.5 s single support, 1.5 cm
  swing, CoM 0.255 m, ZMP 3 cm inside the sole; one distinct command each) plus a standing clip (`1000.json`, weight 176).
- Command (0, 0, 0) means stand still and balance: the gait clock stops at a double-support rest phase
  (`gait_rest_phases` 0.46 / 0.96).
- Residual policy on an ILC nominal gait table (`assets/motions/arti_d1_nominal_gait_command.npy`).

## Play / evaluate / export

```bash
python scripts/skrl/play.py --task Template-Arti-D1-Amp-Direct-v0 --algorithm AMP --num_envs 4 --checkpoint <pt>
PY=<isaac lab python> bash scripts/tuning/record_video.sh <pt> <name>    # 40 s video with pushes
python -u scripts/tuning/eval_gait.py --mode sequence --checkpoint <pt> --headless --num_envs 32
python scripts/pt_to_onnx.py --checkpoint <pt> --output agent.onnx
python -u scripts/tuning/verify_onnx.py <pt> agent.onnx
python -u scripts/tuning/check_onnx_in_env.py --onnx agent.onnx --headless
python -u scripts/tuning/check_controller_obs.py --headless          # humanoid_controller ArtiD1 observation
```

humanoid_controller `config/robot/ARTI-D1/robot.yaml` policy settings for this setup: `gait_phase_rate: 0.8`,
`initial_gait_phase: 0.96`, `gait_rest_phases: [0.46, 0.96]`, `command_max: [0.0812, 0.0271, 0.1349]`.

## Regenerate reference motions

```bash
python scripts/tuning/gen_motions.py arti_d1 1000 20          # env_placo python, MOTION_GENERATOR=<motion_generator path>
WINDOW_S=1.25 python scripts/tuning/reference_stats.py        # default pose, action ranges, swing phases, commands
M=source/arti_d1/arti_d1/assets/motions
python scripts/tuning/make_stand_clip.py $M/arti_d1/0.json $M/arti_d1/1000.json --phase 0.96 --weight 176
python -u scripts/tuning/ilc_nominal.py --headless --iters 8   # writes $M/arti_d1_nominal_gait_command.npy
```

## Known gaps (sim-to-real)

- Joint speed is capped at the 12 V no-load speed (3.14 rad/s legs) and the back-EMF is modelled only as damping. The
  reference gait is the fastest that stays below it (leg joints <= 3.1 rad/s): the 0.72 s gait needs 8 rad/s, and
  gaits slower than ~1.3 s need more CoM sway than the hip_roll range (-18 / +12 deg) allows (with the ZMP inside
  the sole, hip_roll reaches +-0.5 rad at 2.4 s).
- Servo gains, delay, friction and the IMU mount are datasheet / assumed values: run a sysid and measure the IMU mount
  before trusting a policy on the robot.

## Real robot logs

`python scripts/tuning/analyze_robot_log.py <log.csv> --onnx <agent.onnx>` checks a humanoid_controller ARTI-D1 runtime
log against the training environment.
