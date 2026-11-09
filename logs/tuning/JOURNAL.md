# ARTI-D1 tuning journal (AWS server training; local = eval / reference work)

## 2026-10-28 exp1 / exp2
- 600k eval (run 2026-10-27_06-45-28, 15 mm reference, task/style .5/.5): 0/32 falls, stands still (roll 0.2 deg),
  turn 0.17/0.18, but linear tracking ~27 % (2 cm steps), foot peak 15 mm = reference swing, style .26, disc acc .998.
- exp1 (550fc0c): resume 06-45-28 with task/style .3/.7. Superseded by exp2 before evaluation.
- USER: "arti_d1은 발을 거의 들질 않네". Lift is capped by the servos: at the 1.25 s gait 30 mm needs knee 4.6 rad/s,
  40 mm 5.7 rad/s (no-load 3.14). Longer swings push hip_roll past -18/+12 deg. Grid -> chosen: swing 22 mm,
  single support 0.55 s, feet_spacing 0.17, step <= 4.5 cm (motion_generator config/arti_d1; old config + data kept as
  *_15mm_2026-10-28). 1000 clips: leg |qd| max 3.03 rad/s, hip_roll L -17.9/+10.5 R -11.7/+18.2 deg, period 1.359 s,
  commands x +-0.052 y +-0.024 yaw +-0.12 (max speed ~0.065 m/s, was ~0.08).
- cfg recomputed (0fb1c58): pose, action scales, commands, windows 1.36 s, phase rate 0.73593, swing (0.5072, 0.0055)
  dur 0.4048, rest (0.4588, 0.9588), swing height 0.022 / sigma 0.011, air max 0.65, lin sigma 0.0015, joint_acc -4.2e-7.
  ILC 6 it: joint err <= 0.7 deg, zero-action sole peak 12.7 mm (8 it -> 9.2 mm, rejected). Smoke 256 x 3k: no falls.
- exp2 (8e743b6): server restart, 4096 envs, task/style .3/.7, ckpt every 20000. Judge after support fade (~600k):
  foot peak vs 22 mm, step length / linear tracking, falls, style.
- 10-28 14:52 check: exp2 run 2026-10-28_00-29-11 (code 8e743b6, swing 0.022 / phase 0.73593 confirmed in env.yaml) at 196k: lin .38-.42, style .24 -> .26, disc 1.000, air ratio L/R .106/.088 (15 mm run at same stage .11-.12, later .085), joint_acc 194 -> 78, ep 898, roll 2.6 deg. No 300k ckpt yet -> no eval. Next push ~19:51 KST (~400k): eval foot peak vs 22 mm.
- 10-28 20:22 check: run 00-29-11 at 428k: lin .37-.39, style .26, disc .998, joint_acc 71, ep 874, roll .59 deg, air ratio falling .125 -> .074 (watch). EVAL 300k: eval/play commands were still the old 15 mm range (x .10 vs new max .053) -> rescaled (eval_gait COMMANDS/PLAY_SEQUENCE + play.py, same commit). In-range eval: 0/32 falls; forward cmd .05 got .019 (38 %), step 1.5 cm; turn .12 got .083; foot max height walking L 21 / R 37 mm vs standing L 7.6 / R 18.6 (meter offset) -> lift above stand ~14 / ~18 mm (target 22); lowest sole corner in swing +10 / +12 mm above stand; walking roll range ~10 deg (stand 1 deg); hip_roll p-p 10.8/8.4. Video 300k (old x .10 cmd): walks, no falls. No change: judge at 600k (support fade). If lift stays < 18 mm then, raise gait_swing_height reward scale.
- 10-29 02:2x check: run 00-29-11 at 648k: lin .41 -> .36, style .25, disc .996, joint_acc 70, ep 878, roll .61, air ratio .11 -> .07 (falling). EVAL 600k (in-range cmds, stand baseline now 0.1 mm): 0/32 falls, smooth (walk roll range 3.8 deg, stand .12), fwd .05 got .021 (42 %), step 1.4 cm, foot max height 15 mm (target 22), time >10 mm only 7-8 %, hip_roll p-p 4.5/4.2, knee 10/9 deg; DXL knee outside T-w 2.5 %. -> exp3: resume latest, env.gait_swing_height_reward_scale 1 -> 3 (run.yaml rev 6). Local smoke 256 x 3k from 600k: swing-height reward .57 -> .62, air ratio .081 -> .086, lin/ep/style flat (first try hit a transient CUDA unknown error, retry OK). Judge at +300k: foot peak >= 18 mm without tracking/style loss.
- 10-29 08:2x check: exp3 run 2026-10-28_18-11-02 (code 5100025, swing-height x3 confirmed, offset 680k) at 189k: air ratio .080 -> .121 (rising, 15 mm-ish run peaked .12), lin .36 -> .39, style .26 -> .24-.25, disc .996, joint_acc 71 -> 78, roll .68 -> .88, ep 894. No 300k ckpt yet -> no eval. Next push ~10:5x KST (~420k): eval foot peak vs 22 mm.
- 10-29 14:5x check: exp3 run 18-11-02 at 422k: air ratio .12 (was .08 at start), lin .40, style .26, disc .998, joint_acc 77, ep 868, roll .85. EVAL 300k: 0/32 falls; foot max height 21.4-23.4 mm on all walk segments (exp2 600k: 15 mm) = reference 22 mm reached; time >10 mm 12-14 % (was 7 %); fwd .05 got .0265 (53 %, was 42 %), step 1.7-1.8 cm; turn .12 got .100; walk roll range 4.2-4.6 deg (exp2 3.8), stand .1; hip_roll p-p 6-8 deg (within ROM); DXL knee outside T-w 4.8/5.2 % (exp2 2.5 %, watch), no-load speed exceed .4/.25 %. Video 300k: walks upright, visible foot lift, no falls. -> KEEP exp3, no change. Next judge 600k: knee T-w %, smoothness.
- 10-29 eve: exp3 run 18-11-02 at 654k: lin .40, air .12, style .26, disc .997, joint_acc 77, ep 886 (flat vs 14:5x); 600k not evaluated (superseded).
- USER: "hip roll 안쪽으로 10도씩 움직인걸 기본자세로 하고싶어, 이걸 기반으로 보행궤적 다시 생성하고 학습해줘". New reference (0749163): feet_spacing .17 -> .1354 (stance hip_roll 10.0 deg in), zmp_y -.03 -> -.035 (else stance hip_roll -19.2 > -18 limit; ZMP now ~0.5 cm outside the sole fillet), com .255 -> .2585 (else swing knee -20.2 at -20.3 limit). Clips: hip_roll L -17.7/+0.3 R -1/+17.8, knee -16.7/+13.8, |qd| 3.30 max (>3.14 in .15 % frames). Note: no ankle roll, so reference soles roll 10 deg (was 4.8). cfg: pose, root z .01225, action scales, cmd y .0215, knee_lift .023; eval/play y .020. ILC: err <= .6 deg, zero-action sole peak 20 mm. Smoke 256 x 144k: no falls, lin .41, air .12, style .27.
- exp4 (5e27659, run.yaml rev 7): server RESTART on the new reference, args = exp3 (task/style .3/.7, swing-height x3). Judge at 300k/600k: foot peak, hip_roll vs limits, knee DXL %, tracking.
- 10-30 02:2x check: exp4 run 2026-10-29_14-07-00 (code 5e27659, restart confirmed) at 127k: lin .42, air ratio .21/.22 (exp3 .12), style .27 -> .25, disc .999, joint_acc 85, ep 898 (support on), roll 4.0 deg (exp2 at same stage 2.6; narrower stance). No 300k ckpt -> no eval. Next: 300k eval (foot peak, hip_roll vs limits, knee DXL %).

## 2026-10-30 / 31 new URDF (local Ubuntu, Isaac Lab 3.x)
- URDF change (motion_generator e7c14e0): hip_roll joint frame built 0.15 rad inward (L rpy -0.15, R +0.15), pelvis
  +7.797 mm. Same reference gait config (gait.json unchanged): only the stance hip_roll moves -/+0.17466 -> -/+0.02466
  (10 deg inward stance = -/+1.4 deg joint), root z 0.01225 -> 0.00445; other joints, speeds, swing 22 mm, timing and
  action ranges identical. hip_roll now -9 / +8 deg around stance (limits -18 / +12).
- USD re-converted with convert_urdf.py (importer 3.0: arti_d1.usda + payloads/, links nested under Robot/Geometry);
  the in-memory 5.1-layout patches are gone, IMU prim path -> Robot/Geometry/pelvis. check_amp_consistency PhysX /
  Newton: exact, toes 2.0e-4 / 2.5e-4 m. ILC 8 it: joint err <= 0.4 deg, zero-action sole peak 19 / 18 mm.
- Cleanup: AWS / Docker automation removed (training is local now).
- Training on the new reference: not running (stopped on request; an earlier 2048-env run on an interim patched USD,
  2026-10-31_00-08-46, reached 40k and is obsolete). 4096 envs do not fit next to an ARTI-H1 PhysX run on the 16 GB GPU.

## 2026-10-30 (clock reset, KST ~17:3x) D1 Newton campaign (this session; H1 paused)
USER: train D1 on Newton, goal = smaller sim-to-real gap; tune Newton params using PhysX as the reference (test and
compare); every 2 h: check + video + code/param changes.
Method per check: eval the same checkpoint on newton_mjwarp AND isaacsim_physx (eval_gait --physics, sequence, 32 env,
--uniform_action_scale 0.0), video on Newton, compare_backends A/B for any physics/actuator change. One change at a time.
NOTE: the system clock jumped back; run dirs named 2026-10-30_17-* sort BEFORE older dirs -> pass the run dir explicitly.

| run | change | status |
|---|---|---|
| 2026-10-30_17-31-59 | baseline Newton, defaults (4096 env, task/style .5/.5) | stopped at 15k (superseded) |
| 2026-10-30_17-37-41 | + actuator voltage saturation (ea6c5cb), fresh | stopped at 595k |
| 2026-10-30_21-12-00 | + swing-height reward x3 (989d804), warm start 595k, offset 595000 | stopped at 325k (cum. 920k) |
| 2026-10-30_23-17-26 | + Newton contact kd 1100 -> 450 (b4019a3), warm start 21-12-00 agent_325000, offset 920000 | stopped 625k (collapsed after ~300k; best agent_290000) |
| 2026-10-31_03-16-39 | + action-rate diff clamped to +-2 (ae7524b), warm start 23-17-26 agent_290000, offset 1210000 | stopped 335k (eval falls 17/32, degraded) |
| 2026-10-31_05-29-11 | + Newton contact kd 450 -> 500 (6a3ebb4), clamp kept, warm start 23-17-26 agent_290000, offset 1210000 | stopped 285k (falls 16/32, saturated policy) |
| 2026-10-31_07-22-51 | + raw action bound penalty -1 (aecfb5e), kd 500, warm start 21-12-00 agent_305000 (900k, least saturated), offset 900000 | stopped ~280k (best agent_265000; collapsed at 274k) |
| 2026-10-31_09-14-34 | + swing lift target 22 -> 35 mm, knee lift 23 -> 36 mm (67f92c7, user request), warm start 07-22-51 agent_265000, offset 1165000 | stopped 68k (superseded: 35 mm exceeds the knee motor) |
| 2026-10-31_09-46-26 | NEW REFERENCE 28 mm / 1.625 s, arm 20 deg (6e3e774, motion_generator d051a14), warm start 07-22-51 agent_265000, offset 1165000 | stopped at 435k (cum 1.60M): right foot double hop -> knee-lift-once fix |
| 2026-10-31_12-38-06 | knee_lift target paid once per foot per gait cycle (b76d196), warm start 09-46-26 agent_435000, offset 1600000 | stopped at 80k (cum 1.68M): double hop unchanged |
| 2026-10-31_13-12-41 | extra liftoff penalty -10 per liftoff after a foot's first in its gait cycle (1cca970), warm start 12-38-06 agent_80000, offset 1680000 | stopped at 300k (cum 1.98M): double hop unchanged, extra_liftoff flat -.09 |
| 2026-10-31_15-12-51 | FRESH (no ckpt) with current code: 28 mm motion, knee-lift-once, extra liftoff -10 (1cca970); support curriculum from 0 | stopped at 345k: kd 500 fresh learning broken (tilted-foot shuffle, 32/32 falls) |
| 2026-10-31_17-12-09 | A/B fresh 25k, extra_liftoff 0 (kd 500) | done: same 11 deg foot tilt -> penalty not the cause |
| 2026-10-31_17-21-04 | A/B fresh 25k, kd 1100 | done: tilt 3.4, roll 2.8, reward +1.1 (healthy) |
| 2026-10-31_17-30-17 | Newton kd 500 -> 1100 (884ea2f), continue 17-21-04 agent_25000, offset 25000 (= fresh lineage) | stopped at 430k (cum 455k): legs splayed (hip_roll ~0.15 out) |
| 2026-10-31_20-07-37 | + stance width reward -10 x |width - 0.137| (d020a9a), warm start 17-30-17 agent_430000, offset 455000 | stopped at 2125k (cum 2.58M): plateau, width .180 (foot_flat vs 10 deg reference sole roll) |
| 2026-10-31_21-13-01 | FRESH (no ckpt) with stance width reward (d020a9a), kd 1100; parallel A/B vs warm start 20-07-37 | stopped at 230k: width .183 like the warm start (hip_roll ~ +-0.15 is foot-flat geometry, see Log) |
| 2026-11-01_11-03-31 | FRESH: foot tilt measured from the 10 deg reference sole roll (3432243), stance width -10 kept, kd 1100 | stopped at 495k: legs inward OK (width 130-135, hip_roll -.04/+.045); next: sole pad |
| 2026-11-01_14-16-37 | + flat sole pad collider at the 10 deg stance roll (52f2ff0, USD), warm start 11-03-31 agent_495000, offset 495000 | stopped at 440k (cum 935k): USER moved the A/B to PhysX (log train_newton_solepad.log) |
| 2026-11-01_17-21-00 | B: PURE RL, fresh: agent.agent.style_reward_scale=0.0 task_reward_scale=1.0, same env/USD (sole pad, neutral roll, width), nominal gait table kept | stopped at ~25k: A/B moved to PhysX |
| 2026-11-01_17-27-12 | A on PhysX 4096 (aborted: two PhysX 4096 runs OOM, 7.5 GB each) | stopped |
| 2026-11-01_17-28-57 | B on PhysX 4096 | crashed at start: CUDA OOM |
| 2026-11-01_17-31-40 | A: AMP on PhysX, fresh, 2048 env (style .5 / task .5) | stopped at ~10k: restarted with standing pushes |
| 2026-11-01_17-32-31 | B: pure RL on PhysX, fresh, 2048 env (style 0 / task 1, same nets/hparams = PPO update) | stopped at ~10k: restarted with standing pushes |
| 2026-11-01_17-49-34 | smoke 64 env PhysX (standing pushes, B overrides) | done: standing_fraction ~.3, no error |
| 2026-11-01_17-53-46 | A: AMP on PhysX, fresh, 2048 env + standing 20 % / pushes 0.1-0.4 m/s (b89eb4c) | running (log train_physx_amp_push.log) |
| 2026-11-01_17-54-36 | B: pure RL on PhysX, fresh, 2048 env, style 0 / task 1 + same standing pushes (b89eb4c) | running (log train_physx_pureRL_push.log) |

- ea6c5cb voltage saturation: PD drive kp*(target-q) capped at the stall torque; with kd = stall/no-load this caps joint
  speed at the no-load speed on every backend. A/B (standstill nominal gait): above-no-load samples Newton .774 % -> .024 %
  (PhysX .032 % -> 0 %), joint vel rms Newton .519 / PhysX .522 rad/s, pelvis tilt 3.4 / 3.2 deg, sole penetration
  Newton +0.94 mm / PhysX +1.32 mm. Balance support fades 150k-600k: support-off evals are meaningful from ~600k.
- 10-30 19:1x check 270k (run 17-37-41, 64 it/s): ep 899 (support on), style .22, disc .996, lin track .021/.027,
  air ratio .21 -> .17, base roll 1.0 deg, Reward/total .89 -> .03 (torque/regularization curricula ramping).
  EVAL seq 32, same ckpt on both backends (support off, so reference only before 600k):
  | | Newton | PhysX |
  | fell | 0/32 | 0/32 |
  | fwd .05 got | .023 | .013 |
  | foot max height walk / stand | 31-59 / 15 mm | 24-28 / 9 mm |
  | hip_yaw torque rms / vel rms | .30 Nm / 5 deg/s | .71 Nm / 11 deg/s |
  | all-joint torque / power | 1.08 Nm / 5.7 W | 1.6 Nm / 9.2 W |
  Newton fwd segment: one env rolled (roll range 74 deg); no Dynamixel over-speed on either (voltage saturation works).
  Probe of the torque gap (compare_backends, zero action, + new torque / stance foot spin metrics): torques
  Newton hip_yaw .18 / knee 2.29 / ankle .78 vs PhysX .66 / 3.11 / 1.12 Nm; stance foot yaw rate Newton 8.5 vs PhysX
  11.3 deg/s -> not missing torsional friction. MJWarp multiccd: no effect (plane-convex already multi-contact).
  Contact sweep: ke 1e6 -> hip_yaw .21; kd 300 -> .24; ke 1e6 + kd 300 explodes. -> the gap is not contact stiffness.
  Decision: no change. Next: fixed-base (hung) step/chirp on both backends to separate joint dynamics from contact.
  Video auto_270000.mp4 (Newton, VIDEO_SOURCE=newton_gl; Kit recording gives 0 frames under Newton) 2001 frames, 0 blank.
- 10-30 21:0x check 585k: ep 873 (support fade ending), style .25, disc .997, roll .8 deg, lin track .024 -> .015,
  air ratio L/R .06 / .12 (was .21), swing-height reward .31 -> .19 = the feet stop lifting (shuffle).
  EVAL seq 32 same ckpt: Newton fell 0, fwd .05 got .025, roll range 3.5-3.9, foot max 13-17 mm (ref 22), time >10 mm
  4-13 %, power 4.9 W. PhysX fell 0 but fwd got .008, roll range 16-28 deg incl. STANDING (pitch 12) -> the Newton
  policy does not transfer to PhysX yet; torques still ~1.6x higher on PhysX (hip_yaw .8 vs .23 Nm).
  Video auto_585000.mp4 (newton_gl) 2001 frames, 0 blank.
  Decision: swing-height reward x3 as the cfg default (the old server exp3/exp4 passed it as an arg; the PhysX exp2
  showed the same 15 mm shuffle and x3 fixed it: 21-23 mm). Warm start 595k -> run 21-12-00. Next: foot height /
  air ratio, then the PhysX transfer (torque gap probe: fixed-base step test).
- 10-30 23:0x check cum. 900k (21-12-00 agent_305000): ep 873, style .28 -> .19 (watch), disc .993, lin .017,
  swing-height reward .60 -> .83, air ratio L/R .07 -> .10 / .13 -> .18 (L still half of R).
  EVAL seq 32: Newton fell 0, fwd .05 got .029, foot max 22-27 mm (ref 22, was 13-17), walk roll 6.3-6.9, stand .7,
  power 6.2 W. PhysX fell 0, fwd .018 (was .008), foot 25-34 mm, walk roll 11.5-25, stand 1.2-3.6 (was 17) -> x3 helped
  both. Torques PhysX ~1.35x Newton (hip_yaw .7 vs .25).
  Gap probe (compare_backends): HANG (fixed base, no contact): torques similar (hip_roll .54/.59, knee .33/.52 PhysX/Newton)
  -> joint dynamics are not the gap. GROUND, support off, zero action: PhysX topples (tilt 44.5 deg), Newton stays up
  (6.1) -> Newton contact over-damped. kd sweep (support off): 1100/800/600 tilt 6.1-6.2, hip_yaw .19; 450 tilt 31.6,
  hip_yaw .40, knee 2.34, ankle 1.16 (PhysX 2.20 / .96); 300 diverges (tilt 85, bouncing). Support on: kd 450 tilt 3.5,
  sole 1.0 mm (PhysX 3.2 / 1.3). Note: support-on hip_yaw stays ~.2 on Newton for every kd (PhysX .66) -> open item.
  Decision: kd 450 (b4019a3). Risk: knife edge between 450 and 600; if training collapses, try 500-550.
  Video auto_900k.mp4 (newton_gl) 2001 frames, 0 blank.
- 10-31 01:0x check cum. 1.21M (23-17-26 agent_290000, kd 450): ep 808 -> 754 (harder contact, more falls in training),
  style .18, disc .995, lin .019, air ratio L/R .10 / .17 (unchanged asymmetry), swing-height reward .79.
  EVAL seq 32: Newton fell 0, fwd .024, back -.015, roll side/back 4.9/5.3 (fwd 33: one env), stand .23, foot 29-43 mm,
  power 5.9 W. PhysX fell 0, fwd .017, back -.004, roll side/back 4.2/3.7 (fwd 15.8, was 25), stand .17 (was 1.2-3.6),
  foot L 40-42 / R 25-27 mm, power 7.2 W -> the Newton policy now behaves about the same on PhysX (sim2sim gap down).
  Torques PhysX ~1.3x (hip_yaw .72 vs .28; support-on open item). Foot height now overshoots the 22 mm reference
  (x3 + softer contact) and L > R on PhysX. Decision: no change (kd 450 only 290k old). Next: if the overshoot / L-R
  air-ratio asymmetry persists at ~500k on this run, try swing-height x2.
- 10-31 03:1x check cum. 1.52M (23-17-26 agent_605000): COLLAPSE. ep 808 -> 512, action_rate -0.2 -> -5.9 (instantaneous
  reward min -9167: a few envs with huge raw-action jumps), value loss .11 -> .64, swing-height reward .79 -> .71, foot_flat
  -.18 -> -.59, base_flat -.11 -> -.34, style .18.
  EVAL seq 32: Newton fell 0 but roll range 144-155 deg in walking, stand 68-77 deg (was .23), feet 56-86 mm, sole tilt 45,
  feet yawed out 27 deg at stand, left hip_yaw chatter (vel rms 25 vs 2 deg/s, torque 1.8 vs .3). PhysX: same pattern
  (stand roll 24, feet L 26 / R 61 mm, hip_yaw 2.2/.7). agent_400000 (cum. 1.32M) on Newton already bad (fwd roll 60).
  A/B kd (compare_backends, zero action): support off 500 tilt 6.1, 550 5.9 (= the over-damped 600/1100 regime; PhysX 44.5,
  450 31.6) -> raising kd moves Newton away from PhysX, keep 450. Support on 450/500/550 tilt 3.4-3.5, hip_yaw .19-.21.
  Decision: action-rate difference clamped to the applied range +-2 (ae7524b; bounds the per-step penalty, removes the
  outlier value targets). Smoke 64 env ok. Warm start from agent_290000 (last good, 1.21M) -> run 03-16-39.
  Video auto_1525k.mp4 (newton_gl, collapsed policy) 2001 frames, 0 blank.
  Next: watch action_rate / value loss / ep length; eval at ~300k of the new run vs the 1.21M numbers.
- 10-31 05:2x check cum. 1.505M (03-16-39 agent_295000, action-rate clamp): training: outliers gone (instantaneous min
  -9167 -> -132, action_rate -.5, value loss .40 flat), ep ~590 flat, style .16, lin .020, air ratio L/R .10/.17.
  EVAL BUG FOUND: eval_gait added the +0.15 m trunk offset unrotated, so a D1 pelvis lying at 0.09 m (or flipped,
  feet up) stayed above the 0.2 m fall threshold -> every earlier D1 "fell 0" was wrong, and the 144-360 deg roll ranges
  were fallen robots. Fixed (ea70f06: env termination rule = rotated offset + max tilt). Re-eval seq 32 (fell Newton /
  PhysX): 900k (kd 1100) 5 / 5, 1.21M (kd 450 290k) 5 / 3, 1.505M (clamp run) 17 / 15; falls mostly in the first forward
  segment. Surviving envs at 1.505M: Newton feet 83-102 mm (overshoot), PhysX L 32 / R 62 mm.
  -> both continuations from the kd 450 1.21M ckpt degraded with or without the clamp, so the clamp was not the fix; kd 450
  is the common factor (knife edge: 500 open-loop = 1100 regime, 300 diverges; ep length fell from the start of the kd 450
  run while kd 1100 stayed ~890). Decision: kd 500 (6a3ebb4; A/B support off tilt 6.1 / on 3.44, hip_yaw .20: stable
  side, same as 1100 open loop), clamp kept. Warm start agent_290000 -> run 05-29-11. Smoke 64 env ok.
  Video auto_1505k.mp4 (newton_gl) 2001 frames, 0 blank. Old eval logs without _v2 have the broken fall count.
  Next: fixed-eval falls on both engines at ~300k vs the 1.21M baseline (5 / 3); if still degrading, kd 1100 back
  (900k ckpt) and look elsewhere (e.g. swing-height x3 overshoot).
- 10-31 07:1x check cum. 1.47M (05-29-11 agent_260000, kd 500): ep ~600 flat (no recovery), style .19, lin .018, no reward
  outliers. EVAL seq 32 (fixed): fell Newton 16 / PhysX 16 -> kd 500 did not help; kd was not the cause.
  ROOT CAUSE (EVAL_FALL_DEBUG, 07cae16): falls at 0.37-0.62 s after reset, pelvis 0.07 m, actions at +-1 on nearly every
  joint. Raw policy output (PhysX, obs |max| ~3): 300k (17-37-41) mean .35 / 4% beyond +-1 after the first step;
  900k (21-12-00 305k) 1.7 / 21-25%; 1.47M 20 (max 255) / 50%. The policy mean drifted outside the +-1 clamp (nothing
  penalizes it; std is pinned at the 0.082 floor) -> bang-bang policy. Explains the falling ep length since ~1M and the
  -9000 action_rate outliers. Even 300k jumps to |raw| 30 on the first step after reset.
  Decision: penalty on raw |a| beyond 1 (aecfb5e, scale -1, overshoot capped at 3, not ramped; logs Reward/action_bound,
  Policy/raw_action_saturated_fraction). Warm start from the least-drifted good ckpt 21-12-00 agent_305000 (900k) with
  kd 500 -> run 07-22-51. First 14k: saturated fraction .185 -> .139.
  Video auto_1470k.mp4 (newton_gl) 2001 frames, 0 blank.
  Next: saturated fraction -> ~0.05, raw mean < 1, eval falls (fixed tool) below the 900k baseline 5 / 5. Then revisit
  kd 450 vs 500 for the PhysX match (the kd 450 run's collapse was this drift, not necessarily the contact).
- 10-31 09:1x check cum. 1.175M (07-22-51, action bound): ep 825-890 up to 270k (best of the campaign), saturated fraction
  .185 -> ~.10 (not lower), value loss .03-.2; then a sudden collapse at 270-274k: ep 890 -> 606, action_bound -.06 -> -2.1.
  EVAL seq 32 (fell Newton / PhysX): agent_265000 (1.165M) 2 / 4 (best so far; PhysX falls late 1.5-31 s), raw mean ~1,
  16-20% beyond +-1; agent_275000 (1.175M) 17 / 15 (bang-bang falls at 0.35-0.85 s, raw mean 4-5). Feet at 1.165M:
  Newton 25-30 mm, PhysX 21-25 mm, knee lift ~21 mm; forward .036 Newton; torques PhysX ~1.45x (hip_yaw .7 vs .26).
  -> the bound penalty helps but the drift / sudden collapse is not solved (std pinned at the .082 floor).
  USER (after the videos): the legs should lift more. Decision: lift target 22 -> 35 mm, knee lift 23 -> 36 mm (67f92c7);
  warm start agent_265000 -> run 09-14-34. Video auto_1165k.mp4 (newton_gl).
  Next: foot height toward 35 mm on both engines, falls; if the sudden collapse repeats, stronger bound / tanh-squashed
  actions (structural).
- 10-31 09:3x USER: change the reference motion to lift the legs more and walk more naturally, like Disney BDX.
  Feasibility (motion_generator test sets, 16 clips each, reference joint speed vs X540 no-load 3.14 rad/s at 12 V):
  current 22 mm / 0.55 s: knee max 3.07. BDX-like 35 mm / 0.40 s: knee 6.35 (22% of frames over), 0.35 s: 7.33.
  35 mm / 0.55 s: 4.67 (15%); 30 mm / 0.55 s: 4.13; 25 mm / 0.55 s: 3.53 (2.4%); 28 mm / 0.65 s: 3.23 (0.2%);
  25 mm / 0.65 s: 2.95; 28 mm / 0.70 s: 3.06; 30 mm / 0.75 s: 3.05. -> higher lift needs a slower cycle on this motor.
  USER chose: stay inside the hardware limit. Chosen 28 mm / 0.65 s single support (1.625 s cycle), arm swing 15 -> 20 deg.
  Full set: knee max 3.4 rad/s in 0.5% of frames (old set 0.15% at 3.3), commands x +-0.0415 (was 0.0519, ~20% slower).
  ILC nominal table re-run (8 iters, knee err .5 deg). Env cfg moved to the new clock (6e3e774); action scales kept
  (warm start). The 35 mm reward target (67f92c7) is replaced by 28 mm. eval / play command sequences now forward .04,
  side .016, backward -.04, diagonal (.038, -.015), turn .095 (not directly comparable to earlier evals).
  Deploy note: humanoid_controller must use phase_rate 0.61538, rest phases (0.4572, 0.9572) and the new nominal table.
- 10-31 11:1x check cum. 1.375M (09-46-26 agent_210000, 28 mm / 1.625 s motion): ep 342 -> 600-650 (flat since 40k),
  style .14 -> .21, disc .994, lin .015, swing-height reward .46 -> .52 (flat), air ratio L/R .075/.135, saturated .10-.14,
  power 7.0 W, no reward outliers.
  EVAL FINDING: the sequence eval used RSI starts (reset_standstill_probability 0): envs began mid-stride in random clips
  whose motion disagreed with the forced command, and most falls came 0.4-0.9 s after reset with saturated actions
  (raw mean 13-21). New default --reset standstill (clip frame 0, as the robot starts; c.f. commit). Same ckpt:
  rsi 8 / 15 falls (Newton / PhysX), standstill 0 / 3 (PhysX falls at 1.2, 1.3, 20 s); raw mean .32, 3% beyond +-1.
  -> the "raw drift" seen in earlier evals was largely this start-state mismatch; earlier fall counts are rsi-start.
  Standstill eval 1.375M: Newton fwd .021 / .04, turn .091 / .095, side .008 / .016, stand roll .14, forward roll 5.8;
  PhysX fwd .013, turn .044, stand roll .17, roll 5.7. Feet only 18-22 mm (target 28), knee lift 12-18 mm (ref 27)
  on both engines -> not lifting to the new reference yet. Torques PhysX ~1.5x (hip_yaw .75 vs .25).
  Video auto_1375k.mp4 (newton_gl) 2001 frames, 0 blank. Decision: no change (new motion only 210k old).
  Next: foot height vs 28 mm; if still ~20 mm at ~400k, look at the swing-height reward / knee lift weighting.
- 10-31 12:3x USER: "왜 오른발을 두번 움직여? 수정해." EVAL_STEP_DEBUG (env 0, Newton, 09-46-26 agent_430000,
  logs/tuning/steps_1595k_newton.log): every cycle R lifts twice (phase .04-.14 and .30-.38), L once (.72-.82);
  swings only ~0.13 s vs 0.65 s reference. Cause: knee_lift_reward paid at EVERY touchdown (+2 each), so two hops in
  one clock swing pay double. Fix b76d196: paid only at the first touchdown since that foot's clock swing began.
  Smoke 64 env ok (run dir 12-37-20 = smoke). Warm start agent_435000 -> run 12-38-06 (offset 1.60M).
  Next: re-run EVAL_STEP_DEBUG, expect one lift per foot per cycle and longer swings.
- 10-31 13:1x check cum. 1.675M (12-38-06 agent_75000): ep 574, style .21, disc .995, swing-height .55, air ratio
  L .08 / R .14, roll 1.34. knee_lift reward only ~.002/step -> b76d196 removed almost nothing; the R double hop
  remains on BOTH engines (Newton R .04-.13 + .30-.39, L .72-.82; PhysX same, L sometimes skips). Eval 32 env:
  Newton 0 fell, fwd .028/.04, foot max L 25 / R 20 mm, knee lift 13-15 mm, roll range 6.0, torque all 1.36 Nm,
  5.8 W; PhysX 9 fell (1.0-2.8 s and 26-31 s), fwd .019, foot L 22-24 / R 31-36 mm, knee lift 21-31 mm, roll 6.8-8.9,
  torque 1.74 Nm, 7.1 W (hip_yaw .66 vs .25). Video auto_1675k.mp4 0 blank. Decision (USER asked to fix the double
  hop): 1cca970 extra liftoff penalty -10, smoke 64 ok (run 13-11-50 = smoke), warm start agent_80000 -> 13-12-41.
  Next: EVAL_STEP_DEBUG one lift per foot per cycle? Reward/extra_liftoff -> 0? PhysX falls.
- 10-31 15:0x check cum. 1.975M (13-12-41 agent_295000): ep 687 -> 745, style .22, disc .93, roll 1.1, air L .076 /
  R .132, Reward/extra_liftoff flat at -.09 for 295k (1000: -.090, 295k: -.089) -> penalty not learned away. Policy std
  pinned at the floor .082, lr ~2.7e-5. Step order (EVAL_STEP_DEBUG) unchanged on BOTH engines: R lifts at .04 and .30,
  L at .72. Eval 32 env: Newton 0 fell, fwd .026/.04, foot L 23 / R 20 mm, knee lift 12-15 mm, roll range 5.1-5.4,
  torque 1.22 Nm, 5.4 W; PhysX 4 fell (2.9-4.5 s, 27.9 s), fwd .018, foot L 21-23 / R 25-27 mm, knee lift 9-24 mm,
  roll 5.8-7.6 (diag 16.5), torque 1.70 Nm, 7.1 W (hip_yaw .6-.7 vs .22). Video auto_1975k.mp4 0 blank.
  Decision: the double hop is inherited through the warm-start lineage (old 22 mm / other clock) and a floor-std
  policy does not leave it -> FRESH run 15-12-51 with the same code (no code change). Old ckpts kept.
  Next: step order once it walks (~300k), PhysX falls; balance support fades 150k-600k, so pre-600k evals are reference only.
- 10-31 17:0x check fresh 15-12-51 at 340k (support .58): reward -4.6 -> -10.8 and falling, roll 3.9 -> 5.6, foot tilt
  9-12 deg (fresh 17-37-41 had 3.7), joint vel rms 54 -> 76 deg/s, style .13, extra_liftoff -.14 -> -.21. Eval 340k
  (support off, reference only): 32/32 fell on BOTH engines (0.6-4.5 s); fresh 17-37-41 at 270k had 0/32 on both.
  Bisect with 25k fresh A/Bs (only one setting changed each): extra_liftoff 0 -> tilt still 12 (not the cause);
  kd 1100 -> tilt 3.4, roll 2.8, reward +1.1 = healthy. Decision: kd 500 -> 1100 (884ea2f). kd 500 had no open-loop
  gain toward PhysX over 1100 (10-31 A/B), so the PhysX contact gap (hip_yaw torque ~3x) stays open; find a non-contact
  lever later. No video this check (340k policy falls; nothing to judge). Continue A/B 17-21-04 agent_25000 -> 17-30-17.
  Next: ~300k step order (one lift per foot per cycle?), falls on both engines after support fades (600k).
- 10-31 19:0x check 17-30-17 (kd 1100) at 285k (support .64): DOUBLE HOP GONE: EVAL_STEP_DEBUG one lift per foot per
  cycle (L .55-.68, R .03-.16, swings ~0.2 s, was 2x0.1 s), Reward/extra_liftoff ~-.003 all run. Training: ep 865, style
  .27, disc .999, roll 2.3, BUT raw-action drift from ~150k: Policy/raw_action_saturated_fraction .015 -> .24,
  Reward/action_bound -.004 -> -11 (dominates total +1.0 -> -13.4), foot tilt 2.6 -> 6.2. Eval 32 (support off,
  reference only): Newton 17-24 fell, PhysX 23 fell (fresh 17-37-41 at 270k: 0/0); Newton fwd .023, knee lift 24-25 mm
  (ref 27), torque 1.33 Nm / 7.7 W vs PhysX 1.86 / 10.4 (hip_yaw .34 vs .8). New eval [RAW-SAT] (per joint, envs not
  fallen): hip_roll L .13 / R .08, knee .02-.03, rest ~0 -> the policy wants more hip_roll authority than the action
  scale (L -.16/+.21 rad). Video auto_k285k.mp4 0 blank. Decision: no change yet (pre-600k). Next: if saturation /
  falls still rise after support ends (~600k), widen hip_roll action scale (or tanh-squash) - fresh, one change.
- 10-31 20:0x USER: "why is hip_roll not rotated 0.15? real robot is angled inward like BDX". Reference + USD + URDF all
  carry the 0.15 rad inward hip_roll FRAME (USD rot0 quat .9972/-+.0749); stance joint value -+0.025. But the 17-30-17
  policy (agent_425000, Newton eval, 1/32 fell) holds hip_roll mean +.12-.13 / -.13 walking and +.17 / -.15..-.18
  standing = ~0.15 rad OUTWARD, soles 192-211 mm apart vs reference 137 (60 clips, p5 .123 p95 .152) -> legs look
  vertical. No reward held the width. Fix d020a9a: stance width term (cycle EMA like stagger, L1, -10, not ramped); train
  env reads 0.184 (= eval). Warm start 430k -> 20-07-37 (offset 455k). Smoke run dir 20-06-33.
  Next: eval sole width / hip_roll mean; if no movement in ~100k (std at floor, cf. double hop), restart fresh.
- 20-07-37 agent_155000 (cum 610k, support 0): train width .184 -> .179, tilt 3.2, roll 1.1, ep 810, style .32. Eval support-off
  Newton 0/32 fell, PhysX 1/32 (26.9 s); sole width 185 / 189 mm (was 192-211), hip_roll mean +.11/-.10 (PhysX +.12/-.11),
  standing 184 mm (was 207-211). Torque rms all 1.17 Nm Newton vs 1.64 PhysX (hip_yaw .24 vs .64, same gap). RAW-SAT hip_roll
  .16/.10, arm_m8 .12/.32. Video auto_w155k.mp4 (front) 0 blank. Width moves only slowly under floor std -> started fresh run
  21-13-01 with the width reward in parallel (GPU fits two 4096 runs); keep the warm start as fallback, compare width at ~150k.
- ROOT CAUSE of the splay: the leg has no ankle roll and the hip_roll origin rpy -/+0.15 is not compensated below it, so the
  sole roll = hip_roll frame offset + joint (pinocchio FK, motion_generator URDF = arti_d1_local URDF): default stance
  (-/+0.0247) -> soles rolled -/+10.0 deg; joint +/-0.15 -> 0.0 deg (whole leg vertical). The reference clips themselves
  have soles rolled 10 deg at stance (frame 0) and 4.7-17 deg mid-gait (placo foot task soft). On the floor a flat sole
  wins (foot_flat, contact), so the policy cancels the offset -> vertical legs, 183-190 mm. The fresh run 21-13-01 with the
  width reward ended at .183 too (230k) -> width reward cannot fix it; stopped 21-13-01 (SIGINT).
  Warm 20-07-37 agent_300000 (cum 755k): Newton 0/32 fell, PhysX 1/32 (5.0 s); width 187/190 mm walking, 199 standing;
  torque rms all 1.13 vs 1.67 Nm (hip_yaw .22 vs .69). Video auto_w300k.mp4 0 blank. Asked the user how the real foot stays
  flat (sole wedge / ankle-frame +0.15 / feet really on the inner edge) before changing the model.
- 20-07-37 agent_605000 (cum 1.06M): plateau (width .182, tilt 2.7, roll 1.2, ep 888, style .28). Eval Newton 0/32,
  PhysX 0/32; width 187/190 mm; torque rms all 1.20 vs 1.74 Nm (hip_yaw .21 vs .77). Video auto_w605k.mp4 0 blank.
  No change: still waiting for the user's answer on the foot geometry (wedge / inner edge / vertical legs).
- 20-07-37 agent_915000 (cum 1.37M): plateau (width .180, tilt 3.0, roll 1.1, ep 885, style .28). Eval Newton 0/32,
  PhysX 1/32 (4.1 s); Newton width 183 mm, hip_roll +.099/-.099; torque rms all 1.21 vs 1.69 Nm (hip_yaw .23 vs .73).
  Video auto_w915k.mp4 0 blank. No change: waiting for the user's foot-geometry answer.
- 20-07-37 agent_1220000 (cum 1.68M): plateau since ~600k (width .181, tilt 2.8, roll 1.1, ep 884, style .27). Eval
  Newton 0/32, PhysX 2/32 (2.6, 31.2 s); width 187/192 mm; torque rms all 1.10 vs 1.81 Nm (hip_yaw .20 vs .73). Video
  auto_w1220k.mp4 0 blank. No change: waiting for the user's foot-geometry answer.
- 20-07-37 agent_1530000 (cum 1.99M): plateau (width .180, tilt 3.0, roll 1.1, ep 852, style .28). Eval Newton 0/32,
  PhysX 0/32; width 185/189 mm; torque rms all 1.10 vs 1.68 Nm (hip_yaw .20 vs .70). Video auto_w1530k.mp4 0 blank.
  No change: waiting for the user's foot-geometry answer.
- 20-07-37 agent_1835000 (cum 2.29M): plateau (width .180, tilt 2.9, roll 1.1, ep 881, style .25). Eval Newton 0/32,
  PhysX 3/32 (3.1, 26.7, 31.2 s); Newton width 185 mm; torque rms all 1.06 vs 1.66 Nm (hip_yaw .19 vs .70). Video
  auto_w1835k.mp4 0 blank. No change: waiting for the user's foot-geometry answer.
- USER: the real robot walks like the demo (legs inward, sole rolled ~10 deg); its sole pad is slightly soft and lands
  flat. -> 3432243: foot_flat / foot_tilt measured from the neutral sole up vector R_x(-/+0.1747) z (pinocchio FK at the
  reference stance) instead of world z. The old term charged the reference pose -30 sin^2(10 deg) = -0.9 per foot, far
  above the width term (-0.45), hence the outward hips. Smoke 64 env fresh (run dir 11-01_10-58-07): width .13, tilt from
  neutral 4 deg. Stopped 20-07-37 at 2125k; FRESH 11-01_11-03-31 (habits do not unlearn at floor std).
  Not modelled yet: the soft pad (rigid sole edge in sim). If the sim foot rocks on its inner edge, try a wedge/compliant
  sole collision next (A/B with compare_backends).
- 11-01_11-03-31 agent_10000 (12k, support 1.0): width .135 (ref .137), tilt from neutral 3.7, roll 3.6, swing height .99,
  reward +1.33, ep 899. Support-off eval reference only: Newton 32/32 and PhysX 32/32 fell in 1-4.7 s (expected this early).
  Video auto_n10k.mp4 0 blank. Continue.
- 11-01_11-03-31 agent_330000 (support .59): width .135, tilt-from-neutral 3.1, roll 1.3, swing height 1.09, air .18,
  style .25, ep 899. Eval Newton 0/32, PhysX 5/32 (1.4-11.7 s, ref only <600k); hip_roll mean -.041/+.045 (ref -/+.025),
  sole width 130/134 mm (ref 135) -> legs inward like the demo; hip_roll RAW-SAT 0 (was .10-.16). Torque rms all 1.32 vs
  1.74 Nm (hip_yaw .25 vs .74). Video auto_n330k.mp4 0 blank, legs visibly angled in, soles on inner edge. Continue.
- USER: "walk smoothly and stably". 330k eval showed big motions (hip_roll p-p 17 deg vs 6, knee 30, lateral vel osc
  194 mm/s, PhysX roll range 14 deg, PhysX feet lift ~10 mm while standing). The rigid sole stands on one edge at the
  10 deg stance roll (it is the OUTER edge: lowest flat-face corner y = -0.0079 on the left foot) while the real soft pad
  lands as a face. 52f2ff0: 8 mm box collider under each foot link, flat at the stance roll, bottom through the lowest
  sole corner (purpose guide). compare_backends nominal gait, no pad -> pad: PhysX stance foot spin 10.4 -> 5.0 deg/s,
  tilt 3.66 -> 3.20, min sole z 1.3 -> 2.8 mm; Newton spin 11.0 -> 7.7, tilt 3.45 -> 3.15, min sole 1.05 -> 1.21 mm;
  hip_yaw torque gap unchanged (.66 vs .19). Logs cmp_{pad,nopad}_*.log. Warm start 495k -> 14-16-37.
  Note: git show of base.usda gives the LFS pointer; use `git show HEAD:<f> | git lfs smudge`.
- 14-16-37 agent_120000 (cum 615k, support 0): width .134, tilt 2.8, roll 1.3, joint_acc 61, style .29, train ep 616 (the
  ep drop started before the pad: 11-03-31 899 -> 582 as support went .5 -> .23). Eval Newton 0/32, PhysX 0/32 (330k no
  pad: 0 / 5); forward roll range 5.7/5.6 deg (PhysX was 14.4), lateral osc 161/154 mm/s (194/172), hip_roll p-p 9-11 deg
  (17), knee 23 (30), feet 29-32 mm; PhysX stand roll .25 deg, no foot lifting (was ~10 mm). Torque rms all 1.24 vs 1.75
  (hip_yaw .24 vs .83). hip_roll mean -.038/+.044, width 131/133 mm. Video auto_p120k.mp4 0 blank. Continue.
- 14-16-37 agent_405000 (cum 900k): flat (width .136, tilt 2.7, roll 1.6, joint_acc 62, style .29, train ep ~600 for
  300k). Eval standstill reset (DR + pushes on): Newton 0/32, PhysX 2/32 (1.2, 10.7 s); hip_roll mean -.028/+.028 (ref .025),
  width 136 mm; hip_roll p-p 8-9, knee 21-24, lateral osc 160/150 mm/s; torque all 1.22 vs 1.78 (hip_yaw .25 vs .84).
  Newton --reset rsi: 15/32 fell at 0.4-1.2 s -> the short training episodes are RSI-start falls (random clip frame vs
  state), not mid-walk instability; vertical-leg policy had ep 858. Not acted on yet (real robot starts from standstill);
  candidate if it persists: check the RSI root height with the pad (sole up to 17 deg roll mid-gait). Video auto_p405k.mp4
  0 blank. Continue.
- USER: "could it walk with RL instead of imitation?" -> A/B: A = AMP 14-16-37 (keeps running), B = pure RL 17-21-00
  (style 0, task 1; discriminator still trains but pays nothing; nominal gait table kept). USER: stop the Newton vs PhysX
  comparison -> checks eval Newton only. Compare at B ~600k (support off): falls, ROM, sway, torque, front video.
- USER: A = AMP on PhysX, B = PPO on PhysX. Stopped both Newton runs. skrl_ppo_cfg.yaml is the stock template (32x32 nets,
  other std/lr) so B uses the AMP agent with style 0 / task 1 (same nets, PPO policy loss; the discriminator pays 0) for a
  fair A/B. Two PhysX 4096 runs do not fit (7.5 GB each, B OOM) -> both fresh at 2048 env, ~20 it/s each. Checks eval
  PhysX only (cron eafe77c2). Compare after both pass ~600k (support off).
- USER: does PPO learn to resist pushes while standing? Only 1 of 1001 clips stands (~0.1 % of training) and the pushes
  are +-0.05 m/s, so neither run trained it. b89eb4c: stand_command_probability .2, standing pushes 0.1-0.4 m/s every
  2-4 s, eval --mode standpush (push .2/.3/.4/.5 m/s x 8 directions at 3 s). Baseline AMP 14-16-37 agent_440000 PhysX:
  0/8, 1/8, 4/8, 7/8 fell, no recovery step (logs/tuning/eval_standpush_p440k_physx.log). Restarted A and B fresh
  (17-53-46 / 17-54-36), cron f936f757 adds the standpush eval.
- A/B check 95k (support still on, support-off evals are reference only). check.sh A/B: width 130.5/131.6 mm, roll 4.2/4.9 deg,
  foot tilt 3.1/3.1 deg, standing_fraction 0.32/0.32, air ratio .19/.20 / .18/.21, ep len 899/883, AMP style 0.30/0.23.
  PhysX sequence (eval_{A,B}_95k.log): both fell 32/32 within 0.6-2.0 s (A median ~1.25 s, B ~0.82 s), sole width A 126-141 / B 126-128 mm.
  Standpush (standpush_{A,B}_95k.log), fell at .2/.3/.4/.5 m/s: A 0/8, 1/7, 4/8, 7/8 (= old baseline, survivors take
  0.3-1.0 foot lifts); B 2/8, 3/6, 4/5, 5/7 (6 fell before the push). Videos skipped: OOM with both trainings running
  (Kit and newton_gl both need >3.5 GB free). No changes.
- A/B check A 225k / B 235k (support fading from 150k, evals reference only). check.sh A/B: width 126.6/125.8 mm, roll 2.5/2.5 deg,
  foot tilt 3.2/3.3, air .18/.18, joint_acc 71.6/70.5, ep len 899/898, style 0.27/0.25. PhysX sequence: both 32/32 fell, now later
  (A median ~2.1 s, B ~2.0 s, was ~1 s at 95k); fwd-walk sole width A 93 / B 109 mm (narrowing when stumbling), standing ~129 / 132-146 mm
  (B standing hip_roll noisier, roll range up to 25 deg in some segs). Standpush fell .2/.3/.4/.5: A 0/8, 2/8, 4/8, 6/8; B 0/8, 1/7, 4/8,
  6/8 (B caught up; still ~baseline). Videos skipped (OOM with both trainings). No changes.
- New eval_gait.py --mode score (walking scorecard over the 8-segment play sequence, metrics only before each env's
  first fall, walk/stand separate): stability (falls, time up, roll/pitch and rate rms), tracking (segment-mean velocity
  error), sway, smoothness (action rate/jerk, joint acc, pelvis acc), feet (touchdown speed, stance slip, swing height /
  time / symmetry, width), effort (torque, power, CoT). --train_step N evaluates with the balance support of step N.
  A/B 235k, support as trained (0.81): both 0/32 fell; walk roll rms 3.5/3.5, pitch .88/1.32 deg; track err x 11.9/14.2
  y 18.9/19.9 mm/s; action rate .091/.089, joint acc 25.0/25.4; touchdown 202/210 mm/s, slip 43/51 mm/s, swing sym
  .92/.95, power 9.2/9.5 W; stand roll .41/.63 pitch .73/.99 deg. Support off: 32/32 fell both (time up 3.1 / 2.0 s).
  (logs/tuning/score_{A,B}_235k_{sup,off}.log)
- A/B check at 350k (A ckpt 350k, B 365k latest; support 0.55/0.52). check.sh A/B: roll 1.2/3.0 deg, foot tilt 2.6/6.7 deg
  (B tilt jumped ~3 -> 7.7 at 276k), ep len 893/774 (B down from 898 since ~184k), width 128.7/127.0 mm. Score @350k
  support off: falls 14/32 both (was 32/32 at 235k), time up 30.0/30.6 s; walk roll rms 2.6/2.9, pitch 1.2/2.0 deg; track
  err x 17/17 y 14/17 mm/s; joint acc 24.3/25.3; touchdown 257/192 mm/s, slip 30/41 mm/s, swing h 19/17 mm, sym .91/.94,
  power 9.1/11.4 W (CoT 1.0/1.2); stand roll 1.1/2.2 pitch 4.0/4.2 deg, sway 30/50 mm/s. Support as trained: 0/32 both.
  Standpush .2/.3/.4/.5: A 0/8,1/8,3/8,6/8; B 0/8,1/8,3/8,6/8 (both a bit better than the old baseline 0,1,4,7).
  Sequence: A fell 19/32, B 26/32. Videos skipped (OOM). No changes.
- A/B check at 475k (support 0.28/0.24). check.sh A/B: roll 1.1/0.9, foot tilt 2.5/2.5 (B recovered from 6-7), width 133/129 mm,
  air .10/.11, ep len 711/804 (A down from 897 at 358k). Score support off: falls 20/32 / 18/32 (350k: 14/14), time up 27.9/30.4 s;
  walk roll rms 1.5/1.4, pitch 1.4/1.7 deg, sway 142/156 mm/s, joint acc 23.6/23.6, touchdown 307/319 mm/s, slip 24/26 mm/s,
  swing h 27/27 mm, sym .87/.97, power 7.9/8.8 W; stand pitch rms 5.8/7.1 deg. Support as trained (0.28): falls 2/32 / 6/32.
  Standpush A 0,2,3,6/8, B 0,1,3,6/8. Sequence fell 17/17; falls binned by segment: A 475k all 17 in stand segments,
  B 12/17 in stand (350k: A 17/19, B 23/26 in stand). => walking is mostly stable; the failure is stopping (walk -> zero
  command), the pitch rms in stand (5.8-7 deg) shows it. Standpush from a standstill is fine. No changes (comparison).
- A/B VERDICT at 600k (support 0 in both). Score (support off): falls A 13/32 B 4/32, time up 32.0/37.5 s; walk roll rms
  1.32/1.84, pitch .86/1.12 deg, sway 147/153 mm/s, joint acc 23.3/23.4, slip 28/25 mm/s, swing h 29/27 mm, touchdown
  417/345 mm/s, power 6.9/7.1 W; stand pitch rms 4.5/2.0 deg. Sequence fell A 20/32 (19 in stand segments) / B 4/32.
  Standpush .2/.3/.4/.5: A 0,1,4,6/8 (=baseline), B 0,0,1,3/8 with real recovery steps (0.5 m/s: 6 foot lifts, 176 mm drift).
  => B (pure RL) is clearly more stable (stopping + pushes); A walks marginally smoother (less roll, lower power).
  Single checkpoint, 32 envs: B's jump from 475k (18/32) to 600k (4/32) should be confirmed on later ckpts.
- User chose "continue B, stop A" on the 600k verdict; A stopped (SIGINT) at agent_920000 (it had kept training while the
  question waited). Videos 0 blank: logs/tuning/videos/auto_{A,B}_{600k,920k}.mp4. Recheck at 920k REVERSES the 600k
  verdict: score (support off) A 0/32 fell / B 3/32; walk roll rms 1.49/1.77, pitch .90/1.05; stand pitch rms 1.08/1.64 deg;
  standpush A 0,0,0,2/8 (recovery steps, 2.2 lifts at .5) / B 0,0,1,3/8. A learned stopping/push recovery later than B
  (A 600k: 13/32, 11/32). Asked the user whether to resume A from agent_920000.
- User then said: stop A, resume B. A resume (run 2026-11-02_08-13-13, no ckpt) stopped after ~1 min. B resumed from
  17-54-36 agent_955000 (offset 955000, same overrides, 2048 envs) -> run 2026-11-02_08-13-47, log
  logs/train/train_physx_pureRL_push_resume955k.log. A's final ckpt stays 17-53-46 agent_920000.
- 2026-11-02 CORRECTION of the 920k "A better" recheck (user saw A fall right at the start of auto_A_920k.mp4). Contact
  sheet of the video: env 0 in A collapses at 0.5 s and is reset at ~1.5 s; B from the same start state keeps standing.
  The 920k scorecard used standstill starts only (score mode forces reset_standstill_probability=1), and the videos
  were only blank-checked, never watched. Sequence eval with RSI starts (play/train-like mix): A fell 7/32 (5 of them at
  0.48-0.60 s after reset) vs B 3/32 (first at 1.25 s) -> eval_{A,B}_920k_rsi.log. A is NOT more robust overall: it
  fails starts from mid-gait states. The decision to keep B stands. Lesson: verdicts need --reset rsi sequence too and
  a watched video (frame contact sheet), not only the blank check.
- 2026-11-02 08:34 B + touchdown speed penalty (USER approved): -5 * (downward sole speed the step before contact)^2 per
  touchdown (cfg foot_touchdown_speed_reward_scale, TB Reward/touchdown_speed), same sample as eval "touchdown speed".
  Reason: eval touchdown speed rose 192 -> 319 -> 345 -> 363 mm/s (350k..920k). B stopped (SIGINT) at 08-13-47
  agent_25000 (cum 980k); warm start from it, offset 980000, same overrides, 2048 envs -> run 2026-11-02_08-34-40, log
  logs/train/train_physx_pureRL_touchdown.log. First TB: touchdown_speed -0.0086/step vs feet_air_time +0.018.
  Watch: eval touchdown speed (target < 250 mm/s), air ratio / swing height not collapsing, falls (920k: score 3/32,
  rsi 3/32, standpush 0,0,1,3/8). Warm start can hide side effects: if the swing shrinks, check with a short fresh A/B.
- 2026-11-02 10:43 check, touchdown run 08-34-40 at 235k (cum 1.215M). TB 0->235k: Reward/touchdown_speed -.0084 -> -.0079
  (only -6 %), air ratio .101/.104 -> .106/.107, foot tilt 2.0, roll 1.05 deg, width 131 mm, ep len ~875.
  Score (support off): falls 2/32 (920k: 3/32); walk touchdown 360 mm/s p95 434 (920k 363/453) -> NO real effect;
  swing h 26.7 mm, air .18, roll rms 1.90 pitch .96, sway 150, track err x 22.3 y 12.8, power 6.4 W (7.7); stand pitch
  rms 1.33. RSI sequence fell 1/32 (920k 3/32). Standpush 0,0,1,1/8 (920k 0,0,1,3/8). Stability a bit better, gait
  unchanged: the -5 term is ~-7 per episode (~2 % of return), too weak. Asked the user before raising the scale.
- 2026-11-02 12:43 check, touchdown run 08-34-40 at 465k (cum 1.445M). TB: touchdown_speed -.0082 (flat since 235k),
  air ratio .108/.112 (up slightly), feet_air_time .0189, foot tilt 2.2, roll 1.1, ep len ~880.
  Score: falls 2/32; walk touchdown 360 mm/s p95 434 (same as 1.215M), swing h 27.4 mm, air .18, sym 1.00, roll rms 1.79
  pitch .91, sway 150, track err x 23.2 y 12.4, power 6.8 W; stand pitch rms 1.23. RSI sequence fell 3/32 (20.8, 21.5,
  34.7 s; 1.215M 1/32). Standpush 0,0,1,3/8 (1.215M 0,0,1,1). Plateau: the -5 touchdown term still has no effect;
  scale decision still pending with the user (proposed -15). No changes.
- 2026-11-02 13:22 touchdown scale -5 -> -15 (USER approved after two flat checks). Run 08-34-40 stopped (SIGINT) at
  agent_540000 (cum 1.52M); warm start from it, offset 1520000, same overrides, 2048 envs -> run 2026-11-02_13-22-49, log
  logs/train/train_physx_pureRL_touchdown15.log. First TB: touchdown_speed -.0236/step (3x), feet_air_time .0188.
  Watch: touchdown < 250 mm/s; swing h (27 mm) and air ratio (.108/.112 TB, .18 eval) must not drop.
- 2026-11-02 14:53 check, -15 run 13-22-49 at 170k (cum 1.69M). TB: touchdown_speed -.024 -> -.022 (barely moves),
  air_time .0186 -> .0175, air ratio .097/.109, ep len ~780 (prev run ~860). Score: falls 1/32, walk touchdown 361
  p95 432 mm/s (NO effect even at -15), swing h 27.1, air .17, lifts/s 1.43, roll 1.80, sway 140, track x 25.1.
  Standpush 0,0,0,3/8. RSI sequence REGRESSED: 10/32 fell, 7 within 0.5-1.0 s of reset (the A-type start failure).
  Confirmed with 64 envs: -15 1.69M fell 9/64 (7 at 0.48-0.83 s) vs -5 1.52M (08-34-40 agent_540000) 2/64 (one at
  0.98 s). Video auto_B_td15_1690k.mp4 (+ _sheet.png): env 0 walks the whole 33 s, no visible problem (its start
  state happens to be benign). Likely cause: mid-swing RSI starts now brake the descending foot instead of planting it.
  Asked the user: revert to 1.52M (term off) vs keep vs reference-based swing descent shaping. Training left running.
- 2026-11-02 15:38 USER: B walks too stiffly ("the demo motion is smooth"). Demo forward clips (22, per 1.625 s cycle):
  ROM hip_roll 15.3, hip_pitch 28.7, knee 29.4, ankle 28.4, arm 20.8 deg; touchdown vz 21 mm/s (p95 35); swing ~0.57 s.
  B 1.69M forward: hip_roll 7.5, hip_pitch 8-11, knee 18-20, ankle 11-12, arm 4-8; touchdown 361; swing 0.27 s.
  A 920k is as small (knee 21-22, ankle 13) -> AMP alone did not fix it. The ILC command table moves the knee ~45 deg
  p-p, so the residual policy cancels part of the nominal stepping.
  Change (USER approved): touchdown term off (0.0), new reference_pose reward 0.5 * exp(-wmse(q - q_ref(phase))/0.11^2),
  q_ref = default + nominal_gait_reference (raw demo mean per phase), weights hip_roll/knee/ankle 1, hip_pitch .5,
  moving commands only (TB Reward/reference_pose, Gait/reference_pose_err_deg). -15 run 13-22-49 stopped at 245k.
  Warm start 08-34-40 agent_540000 (1.52M, best RSI starts), offset 1520000 -> run 2026-11-02_15-38-20, log
  logs/train/train_physx_pureRL_refpose.log. First TB: reference_pose .17/step (kernel ~.53 on moving steps), err 4.9 deg.
  Watch: forward ROM toward demo, touchdown speed, swing time, RSI falls (1.52M: 2/64).
- 2026-11-02 17:13 check, refpose run 15-38-20 at 175k (cum 1.695M). TB: reference_pose .169 -> .198/step, pose err
  5.0 -> 4.1-4.4 deg (flat since ~119k), air ratio .101/.108, tilt 2.1, roll 1.07, ep len 887.
  Score: falls 0/32 (best so far), walk touchdown 377 p95 464 mm/s, swing h 26.9, swing time .26/.27 s, air .17, roll
  rms 1.71, sway 144, power 6.1 W. RSI 64: 2/64 (20.9, 36.6 s; = 1.52M). Standpush 0,0,0,3/8.
  Forward ROM L/R: hip_roll 8.4/7.6, hip_pitch 10.3/7.9, knee 22.0/17.9, ankle 11.5/12.5, arm 2.9/6.2 -> UNCHANGED vs
  1.52M (7.0/8.1, 9.8/8.9, 19.2/20.0, 11.4/12.0); demo 15/29/29/28. The "knee lift" lines vary 12-41 mm between
  evals of similar gaits: not usable for comparison. Video auto_B_ref_1695k(.mp4/_sheet.png): no fall in 33 s, same
  look as before. => 0.5 is too weak to move the warm-started gait (max gain ~+.2/step). Asked the user (raise to 2.0).
- 2026-11-02 19:13 check, refpose run 15-38-20 at 400k (cum 1.92M). TB: reference_pose .211/step, pose err 3.94 deg
  (4.4 at 175k; slow), air ratio .110/.112 (up from .101/.108), tilt 2.3, roll 1.07, ep len 889.
  Score: falls 1/32; walk touchdown 409 p95 489 mm/s (WORSE: 377 at 1.695M, 360 at 1.52M), swing h 29.3, swing time
  .28/.28 s, lifts/s 1.38, air .18, roll rms 1.75, sway 149, power 6.4 W. RSI 64: 2/64 (6.2, 25.8 s). Standpush
  0,0,0,1/8 (best so far). Forward ROM L/R: hip_roll 8.0/7.8, hip_pitch 9.4/9.0, knee 22.4/20.1, ankle 12.5/11.3,
  arm 2.4/6.5 -> still ~unchanged (demo 15/29/29/28/21). Video auto_B_ref_1920k: no fall in 33 s, same look.
  Scale 0.5 -> 2.0 decision still pending with the user. No changes.
- 2026-11-02 21:13 check, refpose run 15-38-20 at 630k (cum 2.15M). TB: reference_pose .205/step, pose err 4.07 deg
  (flat ~3.9-4.1 since 400k), air ratio .105/.111, tilt 2.15, roll 1.09. EP LEN DROP: mean ~888 until ~580k, then
  834 -> 755 in the last ~50k steps.
  RSI 64 REGRESSED: 630k fell 24/64, 22 of them at 0.37-0.50 s (the A-type start failure). Bisect: 560k (cum 2.08M)
  2/64 (1.7, 35.7 s) = good; 600k (cum 2.12M) 8/64 (5 at 0.40-0.42 s). So the collapse started between 560k and 600k.
  Score (96 envs this time; earlier checks used 32): falls 1/96, walk touchdown 387 p95 482 mm/s, swing h 27.3, swing
  time .27/.25 s, lifts/s 1.52, air .17, roll rms 1.55, sway 143, power 6.3 W. Standpush (96 envs, 24/level):
  0,0,0,5/24 (~ same rate as 1/8 at 1.92M).
  Forward ROM L/R: hip_roll 7.3/7.9, hip_pitch 10.5/8.6, knee 21.9/19.4, ankle 10.2/10.0, arm 3.1/6.0 -> unchanged
  (demo 15/29/29/28/21). Video auto_B_ref_2150k(.mp4/_sheet.png): env 0 (standstill-ish start) no fall in 33 s, same
  look. One eval segfaulted (exit 139) when two ran in parallel next to training; rerun alone was fine.
  Asked the user: stop and keep 560k as the best refpose checkpoint vs let it run. Scale 0.5 -> 2.0 still pending.
- 2026-11-02 23:13 check, refpose run 15-38-20 at 855k (cum 2.375M): COLLAPSED. TB ep len 887 (~550k) -> 449 -> 213
  (last 25 logs), pose err 3.9 -> 5.2 deg, roll 1.05 -> 2.4 deg, tilt 2.2 -> 2.8, reference_pose .215 -> .174.
  Evals 855k: score falls 19/32 (time up 16.5/40 s), RSI 64 fell 53/64 (almost all 0.37-0.45 s), standpush 17/32 fell
  before the push. Walk touchdown 363 p95 472, swing .26/.24 s. Forward ROM unchanged (hip_roll 8.0/7.2, hip_pitch
  9.0/9.5, knee 19.2/21.3, ankle 11.4/11.6, arm 4.3/6.7). Video auto_B_ref_2375k_sheet: env 0 tumbles at 1.0-1.5 s,
  episode reset at ~2 s, then stands/steps for the rest.
  Best refpose checkpoint stays 560k (cum 2.08M, RSI 2/64). Training NOT stopped (user approval pending); recommended
  stop + restart from 560k with reference_pose 2.0 (or keep 560k as final).
- 2026-11-03 01:13 check, refpose run 15-38-20 at 1080k (cum 2.60M): RECOVERED from the 600k-900k collapse. TB ep len
  213 -> 889/871-884 (last 125 logs), pose err 3.95 deg, roll 1.09, tilt 2.11, air ratio .107/.111, reference_pose .211.
  Evals 1080k: score falls 0/32 (40.0/40 s), RSI 64 fell 3/64 (0.68-0.78 s), standpush 0,0,0,0/8 (best so far; 0.5 m/s
  drift 93 mm, 3.6 lifts). Walk touchdown 409 p95 480, swing h 27.4, swing time .27/.27 (sym .99), roll rms 1.63,
  power 6.4 W. Forward ROM: hip_roll 8.2/8.0, hip_pitch 10.6/8.8, knee 21.8/20.1, ankle 11.1/11.4, arm 2.8/11.2 ->
  gait still unchanged vs 1.52M (demo 15/29/29/28/21). Video auto_B_ref_2600k_sheet: no fall in 33 s, same stiff look.
  Lesson: a mid-run collapse can recover on its own; 2 h is too short to call it. Best stability checkpoint now
  1080k. Gait smoothness goal not met by 0.5; still waiting on the user (raise to 2.0 from 1080k recommended).
- 2026-11-03 03:13 check, refpose run 15-38-20 at 1310k (cum 2.83M). TB stable: ep len ~870-890 (one dip to 753 ~100k
  ago, recovered), pose err 4.0-4.2 deg (flat since ~400k), air ratio .108/.110, tilt 2.2, roll 1.05.
  Evals 1310k: score falls 0/32, RSI 64 fell 2/64 (5.8, 10.9 s; no early falls), standpush 0,0,0,2/8. Walk touchdown
  399 p95 479, swing h 27.5, swing time .27/.26, roll rms 1.71, power 6.9 W. Forward ROM: hip_roll 8.8/7.7, hip_pitch
  9.3/9.5, knee 20.5/20.0, ankle 10.8/9.8, arm 3.7/9.9 -> unchanged. Video auto_B_ref_2830k_sheet: no fall, same look.
  Plateau: stability ~= 2.60M, gait unchanged for 1.3M steps at scale 0.5. Still waiting on the user (2.0 recommended).
- 2026-11-03 05:13 check, refpose run 15-38-20 at 1540k (cum 3.06M). TB: ep len 813-886, pose err 3.94-4.08 deg,
  air ratio .109/.111, tilt 2.25, roll 1.05 (flat).
  Evals 1540k: score falls 2/32, RSI 64 fell 3/64 (0.75, 0.87, 25.5 s), standpush 0,0,0,3/8. Walk touchdown 416 p95 491,
  swing h 28.7, swing time .28/.29, roll rms 1.68, power 6.4 W. Forward ROM: hip_roll 9.0/8.0, hip_pitch 8.4/10.0,
  knee 20.6/19.2, ankle 12.2/10.7, arm 4.1/11.5 -> unchanged. Video auto_B_ref_3060k_sheet: no fall, same look.
  Plateau continues; best stability checkpoints 1080k (2.60M) / 1310k (2.83M). Still waiting on the user.
- 2026-11-03 07:13 check, refpose run 15-38-20 at 1765k (cum 3.285M). TB: another ep-len dip 886 -> 711-727 in the
  last ~75k (tilt 2.2 -> 2.6), pose err 4.1-4.3 deg, air ratio .106/.113, roll 1.03-1.13.
  Evals 1765k: score falls 1/32, RSI 64 fell 20/64 (8 at 0.38-0.47 s, 10 at 0.6-1.1 s: start failure again),
  standpush 0,0,0,4/8. Walk touchdown 391 p95 484, swing h 26.2, swing time .27/.25, roll rms 1.49, power 6.7 W.
  Forward ROM: hip_roll 8.5/8.4, hip_pitch 10.0/8.7, knee 18.6/21.7, ankle 10.9/11.8, arm 4.7/11.3 -> unchanged.
  Video auto_B_ref_3285k_sheet: env 0 no fall, same look. Run oscillates (collapse/recover cycles ~600k, ~1650k);
  1310k (2.83M) remains the best checkpoint. Still waiting on the user.
- 2026-11-03 07:3x USER approved: stopped run 15-38-20 (PID 398967) at ~1.79M of its own steps. reference_pose_reward_scale
  0.5 -> 2.0 (cfg default). Warm start 15-38-20 agent_1310000 (cum 2.83M; RSI 2/64, score 0/32, push 0,0,0,2/8),
  offset 2830000, log logs/train/train_physx_pureRL_refpose2.log. Watch: forward ROM toward demo (knee 29, ankle 28,
  hip_pitch 29, hip_roll 15), pose err < 3.9 deg, touchdown, and RSI start falls / collapse cycles.
- 2026-11-03 08:xx USER question: "can't it hold one leg, so the body tips and the foot comes down fast?" New tools:
  eval_gait --mode score --dump <npz> (per-step pelvis/feet traces), scripts/tuning/swing_balance.py (per-swing
  pelvis/XCoM offset from the stance foot, roll rate to the swing side; pelvis as CoM proxy for both). 2.83M (15-38-20
  agent_1310000) forward vs demo forward clips:
    swing 0.28 vs 0.65 s, touchdown 421 vs 22 mm/s
    pelvis offset from stance foot toward swing side: mean +56 vs +28 mm (B barely shifts its weight onto the stance foot)
    XCoM (y + vy/w0) mid-swing +65 vs +18 mm (sole half width 17 mm), at touchdown +121 vs +65 mm
    roll rate toward the swing side: late swing +23, at touchdown +46 deg/s vs 0 (demo trunk held level)
  => USER is right: B never gets its CoM over the stance foot, tips toward the swing side and must catch itself, hence
  the short swing and the hard landing. Lower foot height would not fix it. Candidate fix: reward tracking the demo's
  lateral pelvis-over-stance-foot shift (and/or level trunk roll) per gait phase.
  CORRECTION (same day): demo left/right_toe_position are stored relative to the root in the heading frame, the first
  swing_balance run treated them as world positions. Fixed demo numbers: pelvis offset from stance foot +46 mm (B +56),
  XCoM mid-swing +40 (B +65), at touchdown +75 (B +121). Demo pelvis shift onto the stance foot = +-27.6 mm from the
  soles' midpoint (B ~11 mm). Conclusion holds (B shifts less and tips toward the swing side, +23 -> +46 deg/s), the
  gap is smaller than first reported.
- 2026-11-03 USER chose "add it now": new pelvis_shift reward 1.0 * exp(-(shift - shift_ref(phase))^2 / 0.015^2), shift =
  pelvis lateral offset from the soles' midpoint (heading frame), shift_ref = demo mean per clock phase (+-27.6 mm),
  moving commands only (TB Reward/pelvis_shift, Gait/pelvis_shift_err_mm). reference_pose stays 2.0.
  refpose2 run 07-58-16 stopped (~30 min, last ckpt agent_30000, never evaluated). Warm start 15-38-20 agent_1310000 (cum 2.83M),
  offset 2830000, log logs/train/train_physx_pureRL_shift.log, PID 471586.
- 2026-11-03 08:55 check, pelvis_shift run 08-18-13 at 65k (cum 2.895M, 37 min). TB: pelvis_shift err 8.3 -> 2.8 mm,
  reward .40 -> .62, reference_pose err 3.7 -> 2.9-3.1 deg, foot tilt 2.4 -> 3.4-3.6 (worse), roll 1.06 -> 1.40,
  ep len 663 -> 881 -> dip 701-736 in the last logs.
  swing_balance: pelvis off stance +44.6 mm (2.83M +56, demo +46) -> weight shift reached; but XCoM mid +56 / end +107
  (demo +40 / +75), roll rate to swing side late +30, td +52 deg/s (2.83M +23/+46), swing 0.30 s, touchdown 467.
  => it now shifts the pelvis by hip roll (ROM 17.5/18.2, demo 15) with too much lateral speed / trunk roll, still
  tips toward the swing foot.
  Evals: score falls 1/32, roll rms 2.19 (1.71), touchdown 374 p95 553, swing .27/.26, lifts/s 1.76. RSI 64: 13/64
  (11 at 0.32-0.5 s; 2.83M 2/64). Standpush 0,0,0,2/8. Forward ROM: hip_roll 17.5/18.2, hip_pitch 7.4/8.5, knee
  22.1/21.4, ankle 12.4/13.2, arm 5.5/10.5. Video auto_B_shift_2895k_sheet: no fall, visible sideways trunk lean.
  Early (37 min); no changes. Candidate if it persists: level-trunk-roll term (demo roll rate 0) / XCoM-based target.
- 2026-11-03 10:5x USER: "landing stability is insufficient". New scripts/tuning/landing.py (per touchdown from a --dump
  trace): 2.83M touchdowns 1.44/s (clock 1.23/s -> extra corrective steps), bounce 1%, roll-rate peak after td 35 deg/s,
  impact 408 (p90 456); pelvis_shift 65k: 1.70/s, bounce 5%, roll 31, impact 360 (p90 532).
  Demo left sole per phase: stance 0-0.46, lift 0.50 (64 -> 153 mm/s), 28 mm at 0.65-0.69, decelerated descent
  (-158 mm/s at 0.81 -> -12 at 0.88), touchdown ~0.88 (0.62 s swing).
  Change (USER approved): foot_height reward 1.0 * exp(-sum_f ez^2/0.012^2 - sum_f ev^2/0.15^2), ez = sole z -
  (demo z(phase) + 3.4 mm sim stance offset), ev = sole vz - demo vz(phase); moving only (TB Reward/foot_height,
  Gait/foot_height_err_mm). pelvis_shift run 08-18-13 stopped at agent_260000 (unevaluated past 65k). Warm start
  08-18-13 agent_260000, offset 3090000, log logs/train/train_physx_pureRL_foot.log, PID 490208.
- 2026-11-03 11:1x USER: "when a foot lands, the other foot drags". Measured on traces (world xy slide of the planted
  foot B from A's touchdown until B leaves contact): 2.83M median 6.2 / p90 11.4 mm, pelvis_shift 65k 9.7 / 20.8 mm;
  double support ~0.50 s (demo: trailing foot lifts ~0.26 s after the other's touchdown); stance slip 22 -> 28 mm/s.
  Swing sole tilt ~8 deg, edge-drag 0. scripts/tuning/drag.py added (its demo slide number uses root-relative toe
  velocities -> not comparable; use the world-position slide above).
  Change (USER approved): foot_slip_reward_scale -8 -> -30. foot_height run 10-52-36 stopped at agent_45000 (25 min).
  Warm start 10-52-36 agent_45000, offset 3135000, log logs/train/train_physx_pureRL_slip.log, PID 493233.

## 2026-11-03 13:13 check — foot_slip -30 run (11-17-17, agent_165000 = cum 3.300M)
- Alive (PID 493233, ~31 it/s). TB: foot_slip reward -0.37 -> -0.33 (barely moved), foot_height err 3.8 -> 4.5 mm, pelvis_shift err ~3.0 mm,
  refpose err 3.3 -> 2.6 deg, foot tilt 3.6, roll 1.5, ep len mean 811 (85k) -> 651 (168k).
- score: falls 5/32 (2.895M 1/32, 2.83M 0/32), touchdown 356 p95 581, stance slip 23.5 mm/s (28), swing L 0.22 R 0.28, lifts/s 1.81.
- RSI 64: fell 25/64 (22 within 0.38-0.83 s start transient; 3 later). standpush: 2 fell before push, 0,0,0,2/8.
- ROM fwd: hip_roll 17.5/18.1, hip_pitch 10.4/10.6, knee 23.0/24.2, ankle 12.5/14.7 (slightly up vs 2.895M).
- swing_balance: swing 0.27, td 463, off +44.1, XCoM +56/+97, roll late +25 td +48 (2.895M +30/+52).
- landing: 1.57 td/s, bounce 4%, roll peak 31, impact 412/541.
- drag (same inline script for all; double-support slide of planted foot): 2.83M 6.5/12.0, 2.895M 6.9/19.6, 3.300M 6.8/23.5 mm;
  double support 0.48 s. drag.py: 10.0/15.3, 16.8/26.0, 12.3/25.3 mm. -> -30 slip penalty has not reduced drag in 165k.
- Video auto_B_slip_3300k: main robot upright 6.7 s, sideways trunk lean in single support persists.

## 2026-11-03 15:13 check — foot_slip -30 run (11-17-17, agent_375000 = cum 3.510M)
- Alive (~4 h). TB 168k->375k: foot_slip -0.33 -> -0.30, ep len 651 -> 794 (recovered), foot tilt 3.6 -> 3.3, roll 1.53 -> 1.42,
  refpose err 2.6, pelvis_shift err 3.0 -> 3.3, foot_height err ~4.0.
- score: falls 1/32 (3.300M 5/32), touchdown 381 p95 546, stance slip 24.6, swing L 0.27 R 0.30 (sym 0.90), lifts/s 1.61.
- RSI 64: 12/64 (7 within 0.38-0.53 s, 5 after 20 s). standpush 0,0,0,3/8, none before the push.
- ROM fwd: hip_roll 17.5/18.6, hip_pitch 11.1/10.7, knee 24.0/23.5, ankle 13.8/13.3 (slowly up).
- swing_balance: swing 0.32, td 489, off +44.4, XCoM +54/+108, roll late +27 td +52.
- landing: 1.48 td/s, bounce 5%, roll peak 37, impact 467/536.
- drag (ds_slide inline): median 11.3 p90 17.2 mm (3.300M 6.8/23.5, 2.83M 6.5/12.0), double support 0.50 s; drag.py 15.4/21.5.
- Video auto_B_slip_3510k: all robots upright 6.7 s; sideways trunk lean in single support persists.
- Deploy side (same day): humanoid_controller origin/arti-d1-2830k = 2.83M ONNX + real servo zero offsets / axis flips.

## 2026-11-03 17:13 check — foot_slip -30 run (11-17-17, agent_580000 = cum 3.715M)
- Alive (~6 h). TB 375k->580k: flat (foot_slip -0.30..-0.32, foot tilt 3.3->3.5, roll 1.42->1.59, ep len ~770-800).
- score: falls 0/32, touchdown 421 p95 556, stance slip 23.2, swing L 0.28 R 0.28 (sym 0.98), lifts/s 1.35, swing height 28.6.
- RSI 64: 18/64 (13 within 0.4-0.8 s start transient). standpush 0,0,0,1/8 (best so far).
- ROM fwd: hip_roll 16.8/17.9, hip_pitch 12.2/8.3, knee 24.4/21.8, ankle 13.3/13.7.
- swing_balance: swing 0.30, td 490, off +46.2, XCoM +55/+109, roll late +29 td +53. landing: 1.25 td/s (= clock), bounce 3%, roll 31, impact 476/532.
- drag (ds_slide): 10.3/17.2 mm, double support 0.52 s (3.510M 11.3/17.2; 2.83M 6.5/12.0). drag.py 14.7/21.7.
- Video auto_B_slip_3715k: upright 6.7 s; trunk lean to the stance side in single support persists.
- Verdict: foot_slip -30 has not reduced drag in 580k; stability fine (score 0/32, push best), RSI start transient high.

## 2026-11-03 17:26 — drag analysis -> pelvis lateral sway velocity reward (USER "분석해서 끌리지 않게 해")
- trace B_slip_3715k: the planted (trailing) foot slides 9.6 mm in the FIRST third of double support (2nd 0.1, last 1.6),
  toward the landing foot (+10.2 mm lateral, fwd -1.9). Correlated 0.71 with the pelvis lateral velocity toward the
  landing foot at touchdown (policy 263 mm/s, demo 126) and 0.66 with landing speed (484 mm/s). Demo trail foot lifts
  0.17 s after touchdown (policy 0.50 s). foot_slip -8 -> -30 punished the effect, not the cause: no change in 580k.
- New pelvis_lat_vel reward (scale 1.0, sigma 0.08 m/s): exp(-(v_lat_heading - cmd_y - demo_sway(phase))^2/sigma^2),
  demo table = per-clip heading lateral velocity minus clip mean, by gait phase (sinusoid +-131 mm/s).
  TB: Reward/pelvis_lat_vel, Gait/pelvis_lat_vel_err_mm_s.
- Run 2026-11-03_17-26-22 (log train_physx_pureRL_latvel.log, PID 532287), warm start 11-17-17 agent_600000, offset 3735000.

- 11-03 19:13 check, run 17-26-22 (pelvis_lat_vel 1.0) agent_185000 = cum 3.920M (185k steps after the change). PID 532287 alive.
  TB (first->last fifth): pelvis_lat_vel rew .246->.267, err 116->113 mm/s (slow); foot_slip -.30 flat; ref_pose err 2.9 deg;
  pelvis_shift err 3.1 mm; foot_height err 4.0 mm; roll 1.5 deg; air ratio .11/.12; ep len 712->749.
  EVAL: score falls 1/32 (3.715M 0/32); RSI 14/64 (18); standpush 0,0,0,0/8 (0,0,0,1). Walk: touchdown 376 mean / p95 534,
  stance slip 22 mm/s, swing 0.26/0.30 s, lateral sway vel rms 145 mm/s.
  swing_balance: touchdown 463 (490), XCoM +54/+102 (+55/+109), roll td +53 (same). landing impact 448/514 (476/532).
  DRAG (the target): ds_slide 6.9 / p90 12.9 mm (3.715M 10.3/17.2; 2.83M 6.5/12.0); drag.py slide before lift 11.7/17.9;
  new td_latvel.py (scratchpad): pelvis lat vel toward landing foot at td 243 / p90 317 mm/s (same script on 3.715M trace:
  257/332; demo 126); trailing lateral slide 0.17 s after td +5.2 / p90 +10.7 mm (3.715M same script +9.2/+14.6).
  Trailing lift delay still 0.40 s (demo 0.17).
  ROM forward: hip_roll 17/18, hip_pitch 10/9, knee 23/22, ankle 14/14 (demo 15/29/29/28) - hip_pitch/ankle still small.
  VIDEO auto_B_latvel_3920k: 0 blank, upright whole clip, alternating steps, no fall; wide lateral sway visible.
  Verdict: drag down ~40 % early in the run while the lat-vel error itself barely moved -> keep running, no change.

- 11-03 21:13 check, run 17-26-22 agent_395000 = cum 4.130M. PID 532287 alive (3h47m).
  TB (first/mid/last tenth): pelvis_lat_vel rew .246/.259/.269, err 116/115/114 mm/s (barely moving); foot_slip -.30/-.32/-.34;
  ref_pose err 2.89/2.94/3.16 deg (up); pelvis_shift err 3.2/3.3/2.6 mm; foot_height err 4.0/4.1/4.4; roll 1.47/1.49/1.77 deg (up);
  tilt 3.7/3.8/3.9; air .11; ep len 714/709/668 (down) -> watch, not one drop.
  EVAL: score 0/32; RSI 7/64 (3.92M 14, 3.715M 18; fall times .42-1.03 s = start-up); standpush 0,0,0,1/8 (back).
  Walk: touchdown mean 333 p95 550, lifts/s 1.88 (3.92M 1.56) and touchdowns 1.73/s per env vs clock 1.23 -> more extra steps.
  swing_balance td 473, XCoM +55/+104, roll td +49. landing impact 408/526, bounce 7 %.
  DRAG: ds_slide 7.3 / p90 15.8 mm (3.92M 6.9/12.9, 3.715M 10.3/17.2); drag.py 13.1/20.8; td_latvel 256 / p90 340 mm/s
  (3.92M 243, 3.715M 257, demo 126); trailing slide 0.17 s after td +6.6 / p90 +15.1 (3.92M +5.2/+10.7, 3.715M +9.2/+14.6).
  Lift delay still 0.40 s. ROM fwd hip_roll 19/19, hip_pitch 12/8, knee 25/22, ankle 14/14 (demo 15/29/29/28).
  VIDEO auto_B_latvel_4130k: 0 blank; main robot upright, alternating steps whole clip; one background env tumbles at 0.5 s
  (start-up, matches RSI fall times) and is reset by 1.0 s.
  Verdict: drag gain from 3.92M partly given back; the lat-vel error itself is not being learned (116 -> 114 mm/s in 395k).
  err 114 mm/s = 1.4 sigma, near the steepest part of the kernel, so the weight (1.0 vs ref_pose 2.0) is the likelier limit. No change made (needs user OK); option to propose: scale 2-3.

- 11-03 23:13 check, run 17-26-22 agent_605000 = cum 4.340M. PID 532287 alive (5h47m).
  TB (sixths): pelvis_lat_vel rew .255->.276, err 114.3/113.6/115.2/113.7/111.4/110.5 mm/s (slow but now falling);
  foot_slip -.29/-.30/-.31/-.34/-.31/-.28; ref_pose err 2.9-3.1; pelvis_shift err 2.9; foot_height err 4.0; roll 1.46 (the
  4.13M bump 1.67 went back); tilt 3.8; air .11/.12; ep len 747/752/709/658/758/879 (the 4.13M dip recovered -> was noise).
  EVAL: score 1/32; RSI 2/64 (fall times 5.8, 25.8 s; no start-up falls) (4.13M 7, 3.92M 14, 3.715M 18); standpush
  0,0,0,2/8 (0.5 m/s, back). Walk: touchdown mean 399 p95 548, lifts/s 1.47, touchdowns 1.30/s vs clock 1.23 (extra steps of
  4.13M gone), bounce 1 %, swing sym 0.98.
  swing_balance td 473, XCoM +54/+103, roll td +49. landing impact 464/523, roll-rate peak after td 27/49 (4.13M 32/62).
  DRAG: ds_slide 7.1 / p90 12.0 mm (= 2.83M 6.5/12.0; 3.715M 10.3/17.2); drag.py 10.2/15.3; td_latvel 236 / p90 316 mm/s
  (lowest so far; 3.715M 257, demo 126); trailing slide 0.17 s after td +5.0 / p90 +8.9 mm (3.715M +9.2/+14.6). Lift delay 0.40.
  ROM fwd hip_roll 18/18, hip_pitch 10/10, knee 22/23, ankle 13/13 (demo 15/29/29/28).
  VIDEO auto_B_latvel_4340k: 0 blank, every robot upright the whole clip (no start-up tumble this time), alternating steps.
  Verdict: best checkpoint of this run (RSI 2/64, drag p90 back to the 2.83M level, slide 0.17 s +5.0). Lat-vel error is
  finally falling -> keep scale 1.0 running; the scale 2-3 proposal from 21:13 is still open for the user.

- 11-04 01:13 check, run 17-26-22 agent_815000 = cum 4.550M. PID 532287 alive (7h47m).
  TB (eighths): pelvis_lat_vel rew .255->.282, err 114/114/115/113/111/110/109/110 mm/s (slow, flattening); foot_slip -.27;
  ref_pose err 2.88 -> 3.30 deg (slow creep up, watch); pelvis_shift err 2.9; foot_height err 4.1; roll 1.46-1.60;
  tilt 3.7 -> 4.05 (creep up); air .11/.12; ep len 882/825/775 (noisy).
  EVAL: score 0/32; RSI 1/64 (6.1 s) - best so far; standpush 0,0,0,0/8 - best. Walk touchdown mean 378 p95 543, lifts/s 1.55,
  touchdowns 1.34/s vs clock 1.23, bounce 1 %; swing time L .240 R .319 (sym 0.75, 4.34M 0.98) - asymmetric this ckpt.
  swing_balance td 481, XCoM +56/+104, roll td +52. landing impact 460/535.
  DRAG: ds_slide 8.6 / p90 13.3 mm (4.34M 7.1/12.0, 3.715M 10.3/17.2); drag.py 11.9/17.7; td_latvel 233 / p90 315 mm/s
  (3.715M 257, demo 126); trailing slide 0.17 s after td +7.2 / p90 +11.6 (4.34M +5.0/+8.9, 3.715M +9.2/+14.6). Lift delay 0.40.
  ROM fwd hip_roll 19/18, hip_pitch 10/11, knee 23/24, ankle 13/14 (demo 15/29/29/28).
  VIDEO auto_B_latvel_4550k: 0 blank, all robots upright the whole clip, alternating steps, no visible fall.
  Verdict: stability keeps improving (RSI 1/64, push 0/32) but drag stopped improving - 4.34M stays the best drag ckpt
  (keep agent_605000). Lat-vel err plateaus ~110 mm/s (demo match would need ~0) at scale 1.0. No change (user OK pending
  on scale 2-3).

- 11-04 03:13 check, run 17-26-22 agent_1025000 = cum 4.760M. PID 532287 alive (9h47m).
  TB (tenths): lat_vel err 114 -> 108.9 (7th) -> 112.5 / 110.7 = plateau ~110 mm/s, rew .28 flat; ref_pose err 2.88 -> 3.37/3.14;
  roll 1.46 -> 1.73/1.61; tilt 3.7 -> 4.17/3.93; foot_slip -.29; air .12; ep len 883/822/778/688/752 (noisy, no trend).
  EVAL: score 1/32; RSI 15/64 - 13 start-up falls at 0.37-0.55 s + 2 at 36 s (4.55M 1/64, 4.34M 2/64, 4.13M 7, 3.92M 14);
  standpush 0,0,0,2/8. Walk touchdown 395/553, lifts/s 1.64, td 1.41/s vs clock 1.23, swing sym 0.95.
  swing_balance td 490, XCoM +56/+102, roll td +56. landing 474/529.
  DRAG: ds_slide 6.4 / p90 13.2 (best median of run; 4.55M 8.6/13.3, 4.34M 7.1/12.0); drag.py 11.9/16.2; td_latvel 229 / p90
  320 (3.715M 257); trailing slide 0.17 s +5.2 / p90 +11.2. Lift delay 0.40.
  ROM fwd hip_roll 17/18, hip_pitch 12/11, knee 25/23, ankle 14/13 (demo 15/29/29/28).
  VIDEO auto_B_latvel_4760k: 0 blank; main robot upright whole clip, alternating steps; the right background robot is
  displaced at 0.5-1.5 s (likely start-up fall + reset, as in RSI).
  Verdict: start-up (RSI) robustness swings ckpt to ckpt (18/14/7/2/1/15) -> not a trend; drag median ~6-9 mm, p90 12-13,
  lat-vel err plateau 110 mm/s. Scale 1.0 has converged; further drag gain needs the user-pending scale 2-3 change.
  Best this run: 4.34M agent_605000 (drag p90 12.0, RSI 2) / 4.55M agent_815000 (RSI 1, push 0).

- 11-04 05:13 check, run 17-26-22 agent_1235000 = cum 4.970M. PID 532287 alive (11h47m).
  TB (twelfths): lat_vel err plateau 110 mm/s (rew .28); ref_pose err back to 2.87; pelvis_shift err up 2.9 -> 3.39 mm;
  foot_height err 4.2; roll 1.53; tilt 3.86; air .11/.12; ep len 677/716 (noisy, below the 883 peak).
  EVAL: score 0/32; RSI 26/64 - 24 start-up falls 0.38-1.17 s + 11.8 s + 39.3 s (4.76M 15, 4.55M 1, 4.34M 2) -> second
  worse ckpt in a row; standpush 0,0,0,1/8. Walk touchdown 375/527, lifts/s 1.63, td 1.52/s vs clock 1.23 (extra steps up),
  swing sym 0.82. swing_balance td 469, XCoM +55/+105, roll td +52. landing 456/509, roll peak after td 38/61.
  DRAG: ds_slide 8.5 / p90 16.6 (4.76M 6.4/13.2, 4.34M 7.1/12.0, 3.715M 10.3/17.2); drag.py 12.3/19.2; td_latvel 259 / p90 335
  (back at the 3.715M level 257); trailing slide 0.17 s +6.7 / p90 +13.7. Lift delay 0.40.
  ROM fwd hip_roll 16/17, hip_pitch 10/10, knee 24/22, ankle 13/13 (demo 15/29/29/28).
  VIDEO auto_B_latvel_4970k: 0 blank; main robot upright whole clip; right background robot tumbles at 0.5 s (start-up fall,
  matches RSI) and is reset by 1.0 s.
  Verdict: past its best - RSI start-up robustness 1 -> 15 -> 26 and drag back near the 3.715M level. The run is drifting
  with scale 1.0. Keep agent_605000 (4.34M, best drag) and agent_815000 (4.55M, best stability). Training left running
  (stop/restart = user decision); asked the user: stop + restart from agent_815000 with pelvis_lat_vel scale 2-3?

- 11-04 05:43 user OK: stopped run 17-26-22 (PID 532287, verified train.py) at agent_1235000. pelvis_lat_vel_reward_scale
  1.0 -> 2.0 (sigma 0.08 kept, everything else unchanged). New run 2026-11-04_05-43-10 from 17-26-22 agent_815000
  (4.55M, best stability), env.curriculum_step_offset=4550000, log logs/train/train_physx_pureRL_latvel2.log, PID 573730.
  Iterating, demo sway table printed.

- 11-04 05:50 user: "microduck 코드를 참고해서 아예 새로운 학습" (choices: Isaac Lab port, stop the latvel2 run, no demo motion).
  Stopped run 05-43-10 (PID 573730, verified train.py) after ~7 min. Read pollen-robotics/microduck_rl (mjlab 1.3.0, rsl_rl
  PPO, AGENTS.md playbook) + mjlab v1.3.0 velocity base. New task Template-Arti-D1-Vel-Direct-v0
  (arti_d1_vel_env.py / _cfg.py, agents/skrl_vel_ppo_cfg.yaml), commits 8392100, 946020b:
  * no reference motion / clock / nominal table; reward stack = mjlab velocity + microduck overrides: track lin vel 2.0
    (std 0.08 m/s, rescaled from 0.32 for +-0.4), track yaw 2.0 (std 0.28), upright 2.0 (std^2 .05), variable posture 1.0
    (microduck stds, hip_roll walking 0.05 -> 0.1), body ang vel -0.05, dof limits -1, action rate -0.1 -> -1.0 by iter 1500,
    air time 3.0 in [0.15, 0.45] s (microduck [0.125, 0.3]), foot clearance -2.0 / swing height -0.25 at 30 mm (microduck
    20 mm), foot slip -0.1; all x step_dt. Commands x -0.12..0.15, y +-0.08, yaw +-0.4, resample 3-8 s, 15 % turn in place,
    standing fraction .02 -> .25 by iter 2000. Obs actor 51 = ang vel, gravity, q rel, qd, last action, command (noise
    .03/.01/.001/.25, IMU 6 deg misalignment, 0-1 step IMU lag, qd one step old, encoder bias +-.015); critic 60 adds lin
    vel, sole heights, air time, contact. DR: foot friction .7-1.3, mass x.95-1.05, pelvis CoM +-5 mm, armature / Coulomb
    x.9-1.1, Dynamixel delay/stiction/voltage model kept; pushes +-0.15 m/s every 3-6 s. Arms held at 0 (as deployed).
    PPO 512-256-128 elu, std 1, lr 1e-3 KL .01, entropy .01, 5 epochs x 4 mb, 24 rollouts, 10000 iters.
  * stand check (scripts/vel/stand_check.py): zero action 3 s, 0/64 fell, tilt <= 5.6 deg, stance contact detected.
  * scripts/vel/eval_vel.py (score / standpush), record_video.sh TASK/ALGO, play.py play_command_sequence.
  Run 2026-11-04_05-51-37_ppo_torch, 4096 envs, PID 575406, log logs/train/train_vel_microduck.log (~2.5 h).
  agent_6000 (250 iters) smoke eval: 0/32 falls, not tracking yet, flailing legs (expected this early).
  Cron f8dee69e removed, new hourly check 677ad9dd.

- 11-04 06:19 check, vel run 05-51-37 agent_36000 (1500 iters of 10000). PID 575406 alive (28 min).
  TB (sixths): track_lin .115->.192 (of 2.0), track_yaw .010->.022 (of 2.0!), upright 1.86, pose .72->.66, air_time 1.35->1.72
  ->1.47, action_rate -.59->-.74 (weight ramp -.1->-.96), penalties all <= 0 OK. Gait/lin_vel_err .24 m/s flat, yaw_rate_err
  1.0-1.2 rad/s, air ratio .51/.50, peak height 27 mm, slip 252->136 mm/s, roll 2.2 deg, ep len 1185 (= max, no falls).
  Policy std 1.35 -> 2.25: per joint legs .18-.24 but the 4 IGNORED arm outputs 7.4 (entropy bonus with no reward signal).
  LR ~3e-4 (KL adaptive).
  EVAL score 64: 0/64 falls but no tracking: forward .12 -> .037, backward -.10 -> -.079, side .06 -> -.004, turn .3 ->
  -.23; constant yaw drift -.23..-.26 rad/s in EVERY segment incl. stand; touchdowns 2.2/s, swing .42 s, peak 29.7 mm,
  touchdown 506/609 mm/s, stance slip 98 mm/s, double support only 0.02 s (DS slide n/a), stand foot lifts 2.5/s,
  ROM ~41 deg on every leg joint (thrashing). standpush 0,1,2,3/8 = 6/32.
  VIDEO: Kit RTX out of GPU memory next to the 4096-env run (blank); newton_gl recording OK (VIDEO_SOURCE=newton_gl):
  upright, marching/hopping in place with big leg swings, turning; no falls.
  Port bugs found: (1) ignored arm outputs keep a policy dim each -> std 7.4, pollutes entropy/KL; fix = 10 leg actions.
  (2) lin vel tracked at the pelvis LINK origin, which on this USD sits at foot level (z 4.5 mm), so body roll/yaw
  wobble x 0.25 m lever is in the tracked velocity; mjlab tracks root_link_lin_vel_b of a trunk-level root -> track CoM
  (or trunk frame) velocity. Asked user before restarting.

- 11-04 07:19 check, vel run agent_180000 (7500 iters). PID 575406 alive (1h28m). No user answer on the 06:19 restart
  question yet -> run left as is.
  TB (sixths): a jump in the 4th sixth (~iter 4000): track_yaw .05 -> 1.41 (of 2.0), track_lin .23 -> .73, upright 1.98,
  pose .81, body_ang_vel -.002, action_rate -.08 (weight -1.0), slip 127 -> 40 mm/s, roll 2.1 -> 0.7 deg, yaw err .87 ->
  .08 rad/s, lin err .24 -> .18 m/s (pelvis link origin, see port bug 2), air ratio .35/.35, peak height 20 mm, ep len 1196.
  Penalties all <= 0. Arm std still 7.4 (bug 1).
  EVAL score 64: 0/64 falls; forward .12 -> .054, backward -.10 -> -.060, side .06 -> -.002 (none), diagonal (.10,-.05) ->
  (.036, +.014), turn .30 -> .294, stand: drift 0, foot lifts 0/s. touchdowns 1.93/s, swing .42 s, peak 19 mm (target 30),
  touchdown 476/550 mm/s, stance slip 51 mm/s; contact both-down only 10 % of walking time (DS median 0.02 s);
  DS trailing-foot slide 0.4 / p90 1.9 mm (AMP best 6.5/12.0). ROM fwd L/R: hip_roll 18/18, hip_yaw 6/10, hip_pitch 14/8,
  knee 18/31, ankle 26/12 (asymmetric).
  standpush 0,0,5,8/8 = 13/32 (weak at 0.4-0.5 m/s; trained pushes only +-0.15).
  eval_vel.py fixes: ROM indexed the env axis with the joint id (all ~45 deg before, wrong); DS slide window was 0 for
  1-step DS; added contact both-down / both-up fractions.
  VIDEO (newton_gl) auto: all upright whole clip, crouched knees, short steps, visible limp (one leg does most stepping).

- 11-04 08:19 check: vel run 05-51-37 FINISHED (240000 steps = 10000 iters, 1h51m). Final agent_240000.
  TB (eighths): plateau since ~iter 5000: track_lin .73, track_yaw 1.42, upright 1.98, pose .80, slip 38 mm/s, roll .78 deg,
  lin err .18 m/s (pelvis-link point), yaw err .086, peak height 20 -> 23.6 mm, ep len 1196 (max). Penalties all <= 0.
  EVAL score 64: 0/64 falls; forward .12 -> .081 (68 %), backward -.10 -> -.041, side .06 -> -.011 (none),
  diagonal (.10,-.05) -> (.069, -.008), turn .30 -> .298, stand drift 0, stand foot lifts 0/s.
  touchdowns 2.07/s, swing .42 s, peak 24.9 / p10 20.7 mm (target 30), touchdown 460/505 mm/s (AMP 3.715M 490),
  stance slip 54 mm/s, both feet down only 3 % of walking time; DS trailing slide 0.4 / p90 1.6 mm (AMP best 6.5/12.0).
  ROM fwd L/R hip_roll 19/17, hip_yaw 7/19, hip_pitch 14/13, knee 24/34, ankle 25/20 (demo 15/29/29/28; L/R asymmetric).
  standpush 0,0,4,6/8 = 10/32.
  VIDEO vel_mduck_240000 (Kit, GPU free now): 0 blank; all robots upright the whole clip, crouched alternating steps,
  no fall, no visible foot drag.
  Verdict: microduck recipe gives a stable gait with almost no drag on the first try, but lateral walking is not learned,
  backward/forward are under-tracked, swing 25 mm < 30 and the gait is L/R asymmetric. Port bugs 1 (arm outputs, std 7.4)
  and 2 (pelvis-link velocity point) still in; awaiting user decision on a fixed rerun. Hourly cron 677ad9dd removed
  (nothing left to check).

- 11-05 07:18 user: "1, 2 둘 다 반영해서 다시 학습 시작해". Vel task changes (previous run 05-51-37 finished, agent_240000 kept):
  (1) port bugs: 10 leg-only actions (arms no policy output, held at 0); actor obs leg joints only -> policy obs 39
  (ang vel 3, gravity 3, leg q rel 10, leg qd 10, last action 10, command 3), critic 48. Linear velocity (tracking reward,
  critic obs, eval) taken at the trunk point (pelvis + trunk_offset, lin + w x r) instead of the foot-level pelvis origin.
  (2) lateral_fraction 0.15 pure sideways commands (|vy| 0.032-0.08), push_velocity 0.15 -> 0.25 m/s.
  Stand check: 0/64 fell, tilt <= 5.6 deg. Run 2026-11-05_07-18-50_ppo_torch, 4096 envs, PID 659962,
  log logs/train/train_vel_microduck_v2.log, ~1h50m, 38 it/s.

- 11-05 08:34 check, v2 run 07-18-50 (seed 42) at iter ~7800: COLLAPSED. Never learned to stand: ep len 78 -> 43-60 steps,
  policy std 1 -> 7.39 (clamp) on all 10 leg outputs by iter 800, action_rate -25000/ep, value loss -> 1e-6, upright .18.
  (v1 at iter 60 already had ep len 584.) Falling early is a local optimum here (no alive bonus, per-step penalties).
  Debug run A (same code, seed 7, 250 iters, logs/skrl/arti_d1_vel_debug): ep len 1140 by iter 110, std .45 -> healthy,
  so seed-dependent, not a code bug. Added cfg lin_vel_at_trunk toggle (default True, unchanged behaviour) for A/B.
  Stopped 07-18-50 (PID 659962 verified train.py) at 197k steps; no eval/video (unusable policy).
  New run 2026-11-05_08-41-26_ppo_torch, agent.seed=7, everything else as v2, PID 664689,
  log logs/train/train_vel_microduck_v2s7.log; first 210 iters identical to debug run A (deterministic). ETA ~1h35m.

- 11-05 09:34 check, v2 seed-7 run 08-41-26 agent_132000 (5500 iters of 10000). PID 664689 alive (53 min), healthy.
  TB (sixths): ep len 1092 -> 1194, track_lin .034 -> .619, track_yaw .012 -> 1.24 (jump in 5th sixth as in v1), upright
  1.97, pose .83, action_rate -.09, lin err .41 -> .21 m/s (trunk point), yaw err 1.16 -> .10, slip 197 -> 41 mm/s, roll
  .8 deg, peak 23 mm, std .41 -> .036. Penalties all <= 0.
  EVAL score 64 (lin vel now at the trunk point; v1 numbers were pelvis origin): 0/64 falls; forward .12 -> .052,
  backward -.10 -> -.021, side .06 -> .010 (still ~none), diagonal (.10,-.05) -> (.048,+.005), turn .30 -> .285, stand drift
  ~0, foot lifts 0. touchdowns 2.16/s, swing .42 s, peak 24 mm, touchdown 495/637 mm/s, slip 61 mm/s, DS slide .3/p90 2.4 mm,
  both-down .02. ROM L/R: hip_roll 16/18, hip_yaw 10/16, hip_pitch 8/23 (asymmetric), knee 34/30, ankle 17/17.
  standpush 0/32 (v1 final 10/32; 0.4-0.5 m/s now survived).
  VIDEO newton_gl (record_video.sh needs PY=/opt/isaac/IsaacLab-main/.venv/bin/python): all 4 upright the whole clip,
  crouched alternating steps.
  vs v1 final: push much better; forward/backward/side tracking not better yet (mid-training).

- 11-05 10:34 check, v2 seed-7 run 08-41-26 FINISHED (10000 iters, agent_240000).
  TB 2nd half (sixths): track_lin .627 .647 .667 .691 .707 .717 (still rising), lin err .206 -> .189, track_yaw 1.27 ->
  1.37, std .035 -> .032, slip 39 mm/s, peak 27 mm, ep len 1197.
  EVAL score 64: 0/64 falls; forward .12 -> .064 (132k .052), backward -.10 -> -.017, side .06 -> .000 (none),
  diagonal (.10,-.05) -> (.062,-.012), turn .30 -> .296, stand drift ~0, foot lifts 0. touchdowns 2.11/s, swing .42 s,
  peak 24 mm, touchdown 552/682 mm/s, slip 60 mm/s, DS slide .5/p90 1.5 mm, both-down .05. ROM L/R: hip_roll 16/19,
  hip_yaw 16/19, hip_pitch 23/31, knee 37/30, ankle 21/24 (more symmetric than 132k / v1).
  standpush 0/32 (v1 10/32).
  VIDEO Kit: all upright whole clip, crouched alternating steps, no drag visible.
  Decision (user pre-approved): still improving -> continuation, no setting changes.
  Run 2026-11-05_10-37-33_ppo_torch from 08-41-26 agent_240000, agent.seed=7, env.curriculum_step_offset=240000,
  +240000 steps, PID 671276, log logs/train/train_vel_microduck_v2_cont1.log. Checkpoint load verified (std .031,
  track_lin .75 from the start). Cumulative step at its agent_N = 240000 + N.

- 11-05 11:34 check, continuation 1 run 10-37-33 agent_144000 (cumulative 384000). PID 671276 alive (57 min).
  TB (sixths): flat - track_lin .737 -> .723, lin err .186 -> .185, track_yaw 1.41 -> 1.39, slip 38 mm/s, std .030-.032;
  only peak height rose 27 -> 30 mm. -> plateau so far.
  EVAL score 64: 0/64 falls; forward .12 -> .048 (vy -.025), backward -.10 -> -.060 (was -.017), side .06 -> .005 (none),
  diagonal -> (.042,-.019), turn .30 -> .286, stand drift ~0, lifts 0. touchdowns 1.98/s, peak 32 mm (target 30), touchdown
  436/545 mm/s (better), slip 51 mm/s, DS slide .4/p90 1.4 mm, both-down .10. ROM L/R: hip_roll 18/18, hip_yaw 20/20,
  hip_pitch 28/29, knee 41/33, ankle 20/20 -> nearly symmetric now.
  standpush 1/32 (one fall at 0.5 m/s).
  VIDEO newton_gl: all upright whole clip, crouched alternating steps.
  Trade-off vs 240000: backward + symmetry + swing height + softer touchdown better, forward slightly lower.

- 11-05 12:34 final check, continuation 1 run 10-37-33 FINISHED (agent_240000 = cumulative 480000). PID 671276 exited.
  TB (sixths): plateau - lin err .187 .185 .183 .185 .186 .186, track_lin .73 flat, track_yaw 1.42 -> 1.43, peak 28 -> 31 mm,
  slip 38 -> 35 mm/s, std .031.
  EVAL score 64: 0/64 falls; forward .12 -> .053, backward -.10 -> -.042, side .06 -> .002 (none), diagonal -> (.045,-.014),
  turn .30 -> .294, stand drift ~0, lifts 0. touchdowns 2.14/s, peak 29 mm, touchdown 415/549 mm/s (softest so far), slip
  55 mm/s, DS slide .7/p90 1.7 mm. ROM L/R: hip_roll 18/18, hip_yaw 22/20, hip_pitch 27/28, knee 38/32, ankle 22/21.
  standpush 3/32 (all at 0.5 m/s).
  VIDEO Kit: all upright whole clip, crouched alternating steps.
  Decision: plateau -> no further continuation (pre-approved rule). Cron 625147a6 deleted.
  Candidates: v2s7 08-41-26 agent_240000 (best forward .064, push 0/32) vs cont1 agent_144000 (best backward -.060, symmetric,
  push 1/32) vs cont1 agent_240000 (softest touchdown). Lateral walking not learned in any (policy std ~.03).

- 11-05 20:1x user: "1, 2 반영해서 다시 학습" + mid-turn: right-foot kick unlike left -> chose "both" (symmetry + kick penalty).
  Measured on cont1 agent_240000 (new per-foot lines in eval_vel.py): swing max xy speed R 1519 / L 810 mm/s, step length
  32.5 / 30.0 mm, peak 34 / 25 mm, touchdown 385 / 508 mm/s -> right foot flicks out and back.
  v4 changes (all in arti_d1_vel_env*.py, skrl_vel_ppo_cfg.yaml, scripts/skrl/train.py):
  * lateral: extra track_lat_vel term 1.0, std 0.03 m/s on vy; track_lin_vel std 0.08 -> 0.06.
  * exploration: policy std floor 0.1 (min_log_std -2.3026), cap 1.5 (max_log_std 0.405; was 7.39).
  * seed robustness: alive 1.0/s + one-off termination -10.
  * kick: swing_speed penalty -5 x sum over airborne feet of (xy speed - 0.5 m/s)+^2.
  * symmetry: PPO mirror loss 0.5 x mse(mean(mirror obs), mirror(mean obs).detach()) added to every mini-batch backward
    (train.py hook on policy.act / scaler.scale; mirror = swap L/R, negate hip_yaw/hip_roll, wx, wz, gy, cmd vy, wz).
  * kick + symmetry start at iter 500 (symmetry_kick_start_step).
  Short tests (4096 envs): v3 (lateral/std floor/alive) seed 42 stands by iter 110. v4 from iter 0: seeds 42 and 123
  collapsed (std to the cap, ep len < 80), seed 7 OK; nosym / nokick alone OK on 42. With the iter-500 start: seeds 42
  and 123 both stable through 800 iters (std .36, ep len 1187, sym loss .005).
  Stopped v3 full run 20:2x (PID 695994 verified, 6 min in).
  Run 2026-11-05_21-09-49_ppo_torch (seed 42), PID 703207, log logs/train/train_vel_microduck_v4.log, ~1h40m.

- 11-05 21:26 check, v4 run 21-09-49 agent_30000 (1250 iters of 10000; early). PID 703207 alive (16 min).
  TB (sixths): ep len 917 -> 1194, termination -.53 -> -.003, std .72 -> .23, sym loss .018 -> .003 (on from iter 500),
  swing_speed -.10 -> -.05, track_lat .055 -> .073, track_lin .011 -> .024, track_yaw ~.02 (tracking not learned yet; v1/v2
  jumped at iter ~4000), lin err .45 -> .37.
  EVAL score 64: 0/64 falls, no tracking yet (all segments vx ~ -.01, yaw drift -.02..-.06), stand foot lifts 2.8/s
  (marching in place). Per foot L/R: swing max xy speed 565/570 mm/s (v2 cont1 810/1519: kick gone), step 13/14 mm, peak
  35/34 mm, touchdown 653/659 mm/s. ROM L/R identical within 1 deg (hip_roll 14/14, hip_yaw 25/25, hip_pitch 38/39, knee
  41/41, ankle 25/26). standpush 0/32 (but stepping 7-9 lifts/s).
  VIDEO newton_gl: all upright, marching in place with wide crouched stance.

- 11-05 22:3x v4 check (run 21-09-49, PID 703207 alive, 161k/240k steps).
  COLLAPSE from ~133k steps (iter ~5540): Learning rate pinned at the 1e-6 floor from ~134k (KL over target even at
  min LR), then termination reward -0.01 -> -13, action_rate -0.9 -> -21, mean total reward 77 -> -18, symmetry loss
  .003 -> 1.5 (159.8k). std flat .23 (not the std-explosion type of collapse). Still getting worse at 160k.
  Pre-collapse agent_126000 (eval_v4_126000_score/push.log, video vel_v4_126000.mp4):
    falls 0/64 but NO tracking in any segment (vx +.02..+.025 for fwd/back/side/stand, vy ~0, wz +.03..+.08 drift);
    TB track_lin_vel peaked .040 (v1/v2 learned tracking ~iter 4000; v4 never did by iter 5250).
    marching in place, stand foot lifts 2.85/s; per foot L/R swing max xy 587/598 mm/s, peak 28.5/31.0 mm,
    touchdown 595/594, step 18.5/22.4 mm; ROM L/R within 3 deg (hip_pitch 23.9/25.9, ankle 17.8/20.8).
    standpush fell 1/32 (0.5 m/s 1/8). Video: upright, wide crouched stance, stepping in place, slow yaw drift.
  vs v1 05-51-37 agent_240000 (fwd .081/back -.041, push 10/32) and cont1 (fwd .048-.053): v4 symmetry fixed the kick
  but tracking was lost. Suspects (untested): tighter track std (.08 -> .06) + lat std .03 give ~0 gradient when
  standing still, alive bonus makes marching in place a good optimum.
  Decision: not improving -> no continuation. Asked the user before any change / stopping the run.

- 11-05 22:51 user: "추천으로 진행" -> v4 run (PID 703207) stopped at ~165k (collapsed). v5 = v4 with tracking stds
  restored: track_lin_vel_std .06 -> .08 (v2 value), track_lat_vel_std .03 -> .06. Symmetry .5 / swing-speed penalty /
  alive 1.0 / termination -10 / seed 42 unchanged. Run 2026-11-05_22-51-32_ppo_torch, PID 710488,
  log logs/train/train_vel_microduck_v5.log ([SYMMETRY] coef 0.5 seen). Cron 6dcf78db -> c44fc749 (hourly :13).

- 11-05 23:4x v5 check (run 22-51-32, PID 710488 alive, 60k/240k, iter ~2500). TB by sixths: track_lin .027 -> .077,
  track_lat .120 -> .183 (v4 peak .040 / .084), track_ang .008 -> .016, termination -.02, std .20, symmetry loss .003,
  swing_speed -.047, LR 8e-5..2e-4 (healthy, not at the 1e-6 floor).
  agent_60000 (eval_v5_60000_score/push.log, video vel_v5_60000.mp4): falls 0/64, no tracking yet (vx -.014..-.024 in
  every segment = slow backward drift, wz drift up to +.09); per foot L/R swing max xy 581/534 mm/s, peak 28/32 mm,
  step 13/14 mm; ROM L/R within 2.4 deg. standpush fell 2/32 (0.5 m/s 2/8). Video: upright, stepping in place, one
  robot drifting back toward the camera. Too early (v1/v2 learned tracking ~iter 4000).

- 11-06 00:4x v5 check (PID 710488 alive, 138k/240k). COLLAPSE from ~106k steps (iter ~4400), same kind as v4:
  termination -.02 -> -5.6 (109k) then -0.3..-1.5, action_rate -0.8 -> -25..-151, total reward min -3.8e5 (huge actions in
  some envs), mean 82 -> -1080, std .20 -> .36 rising, LR dropped to 1e-6..2e-5. v4 collapsed at ~133k the same way.
  Pre-collapse agent_102000 (eval_v5_102000_score/push.log, video vel_v5_102000.mp4): falls 0/64, NO command
  response at all (vx +.002..+.006 in every segment, vy ~0, wz +.02..+.07 drift); TB track_lin .093 there is mostly
  zero-command envs. L/R swing max xy 576/544 mm/s, peak 30/32 mm, step 14/15 mm, ROM L/R within 3.4 deg (hip_pitch
  29.2/25.8); standpush 0/32. Video: upright, marching in place, slow yaw drift.
  Std restore did not bring back tracking. Common to v4/v5 and absent in v1/v2 (both tracked by iter ~4000, never
  collapsed): symmetry loss .5, swing-speed penalty, alive +1/s, termination -10, lateral term. Mirror map rechecked
  (obs/act perm and signs match sysid expect_positive), no bug found.
  Decision: collapse -> no continuation, cron c44fc749 deleted. Run left running (ends ~01:50); asked the user.

- 11-06 07:58 user chose "v2 + symmetry only" and "stop v5" (v5 had already finished 240k by then, collapsed).
  v6 = v2 config + symmetry loss only: track_lat_vel_weight 0, alive 0, termination 0, swing_speed_weight 0,
  symmetry_loss_coef .5 -> .2 (start iter 500), log_std bounds back to v2 (-20 / 2; v4/v5 had floor .1 / cap 1.5),
  track_lin_vel_std .08 (v2), agent.seed=7 (v2s7 lineage; seed 42 v2 collapsed). Env code unchanged (zero weights).
  Run 2026-11-06_07-58-49_ppo_torch, PID 737585, log logs/train/train_vel_microduck_v6.log
  ([SYMMETRY] coef 0.2, seed 7 seen). Hourly check cron recreated.

- 11-06 08:3x v6 check (run 07-58-49, PID 737585 alive, 72k/240k, iter 3000, ~36 it/s). TB by sixths: track_lin
  .031 -> .043, track_ang .010 -> .034 (rising), std .20, symmetry loss .002, action_rate -.76, LR normal (1e-4 range).
  agent_72000 (eval_v6_72000_score/push.log, video vel_v6_72000.mp4): falls 0/64. Walks FORWARD (unlike v4/v5) but
  mostly regardless of command: vx fwd +.064 / stand +.045 / side +.058 / back +.035 / diag +.060 / turn +.051;
  vy ~0; wz +.04 on fwd/diag. Weak modulation only. Per foot L/R swing max xy 940/928 mm/s (symmetric, no kick),
  step 42/48 mm, peak 30/33 mm, touchdown 672/680; ROM L/R within 2 deg (hip_yaw 6.6/8.2). standpush fell 3/32
  (0.5 m/s 3/8). Video: all upright, walking forward steadily past the camera. Command following expected ~iter 4000.

- 11-06 09:3x v6 check (PID 737585 alive, 211k/240k, ~10 min left). TRACKING LEARNED: TB jump in the 4th sixth
  (track_lin .109 -> .566 -> .742, track_ang .109 -> 1.08 -> 1.39, mean reward 72 -> 118, std .16 -> .029,
  action_rate -.54 -> -.13), symmetry loss .001, no collapse, LR normal. Last sixth still slightly rising.
  agent_210000 (eval_v6_210000_score/push.log, video vel_v6_210000.mp4): falls 0/64; fwd +.075 (cmd .12),
  back -.057 (cmd -.10), diag vx +.059 vy -.001, turn wz +.299 (cmd .30), stand -.002 with foot lifts 0.00/s
  (stands still), side_left vy +.002 (lateral still not learned). Per foot L/R: swing max xy 991/1033 mm/s (p90
  1086/1162; no one-sided kick, cf. cont1 810/1519), peak 22.0/17.9 mm, touchdown 669/538 mm/s, step 28.4/28.6 mm,
  swing .33 s; ROM L/R hip_roll 14.9/18.4, hip_yaw 16.7/19.3, hip_pitch 14.6/18.4, knee 26.8/27.8, ankle 17.0/18.2.
  standpush 0/32, foot lifts 0-2.5 (stands through pushes). Video: walks fwd, stops, walks back, turns; upright.
  vs v1 240000 (fwd .081 / back -.041 / push 10/32) and v2s7 240000 (.064 / -.017 / 0/32): best backward and push so far.

- 11-06 17:24 v6 finished 240k (last sixth still slightly rising: track_lin .726 -> .742, track_ang 1.353 -> 1.391).
  User: "다른 학습도 또 진행해봐" -> chose "v6 + swing-speed penalty", GPU: "new experiment only now" (v6 continuation
  deferred until after v7; 2 x 4096 envs won't fit in 16 GB).
  v7 = v6 + swing_speed_weight -1.0 (limit .5 m/s; v4/v5 had -5 from scratch), fine-tuned from v6 agent_240000
  (env.curriculum_step_offset=240000, agent.seed=7, 240k more steps). Run 2026-11-06_17-24-53_ppo_torch, PID 772771,
  log logs/train/train_vel_microduck_v7.log (checkpoint load + [SYMMETRY] coef 0.2 seen). Cron switched to v7.
  Baseline to beat (v6 agent_210000): swing max xy L/R 991/1033 mm/s, touchdown 669/538, fwd .075 / back -.057 /
  turn .299, push 0/32.

- 11-06 18:3x v7 check (run 17-24-53, PID 772771 alive, 149k/240k). TB by sixths: swing_speed -.042 -> -.019
  (decreasing), track_lin .767 -> .786, track_ang 1.449 -> 1.486, std .025, symmetry .001, action_rate -.13, LR normal,
  mean reward 110 -> 120. Healthy, no collapse.
  agent_144000 (cumulative 384k; eval_v7_144000_score/push.log, video vel_v7_144000.mp4): falls 0/64; fwd +.053 (v6
  .075), back -.071 (v6 -.057), diag vx +.038, turn wz +.301, stand -.005 (0 foot lifts), side vy -.015 (no lateral).
  Per foot L/R swing max xy 859/830 mm/s (v6 991/1033, -15%), touchdown 573/513 (v6 669/538), peak 22.5/20.8 mm,
  step 29.7/23.7 mm; ROM L/R within 3.5 deg. standpush 1/32 (0.5 m/s 1/8). Video: upright, walks fwd/back, turns.
  Trade-off so far: softer swings, slower forward.

- 11-06 19:3x v7 FINISHED (240k more, cumulative 480k). Last sixth still slightly rising (track_lin .793 -> .798,
  track_ang 1.494 -> 1.503, swing_speed -.016 -> -.015, reward 121.4); healthy throughout.
  agent_240000 (eval_v7_240000_score/push.log, video vel_v7_240000.mp4): falls 0/64; fwd +.055, back -.078, diag vx
  +.038, turn wz +.290, stand -.003 (0 foot lifts), side vy -.016 (no lateral). Per foot L/R swing max xy 779/834 mm/s
  (v6 991/1033), touchdown 525/552 (v6 669/538), peak 20.4/23.7 mm, step 29.0/23.9 mm, stance slip 54.6 (v6 70.9);
  ROM L/R within 3.8 deg (hip_pitch 14.7/18.5). standpush 0/32. Video: upright, walks fwd/back, turns, stands.
  v7 continuation NOT started (needs user approval); asked.
- 11-06 19:29 v6 cont1 started as deferred (swing_speed_weight back to 0.0 = v6 config): from v6 agent_240000,
  env.curriculum_step_offset=240000, agent.seed=7. Run 2026-11-06_19-29-54_ppo_torch, PID 780477,
  log logs/train/train_vel_microduck_v6_cont1.log (checkpoint load, seed 7, [SYMMETRY] .2 seen). Cron switched to it.

- 11-06 20:4x v6 cont1 check (run 19-29-54, PID 780477 alive, 176k/240k). TB by sixths: track_lin .760 -> .770 (flat
  last 3 sixths .773/.771/.770), track_ang 1.443 -> 1.475 (slowly rising), reward 110 -> 120.7, std .026, symmetry .001,
  action_rate -.134, LR normal. Near plateau.
  agent_174000 (cumulative 414k; eval_v6c1_174000_score/push.log, video vel_v6c1_174000.mp4): falls 0/64; fwd +.063,
  back -.073, diag vx +.047, turn wz +.293, stand -.002 (0 foot lifts), side vy -.012 (no lateral). Per foot L/R swing
  max xy 998/1045 mm/s (= v6, v7 779/834), touchdown 586/526, peak 21/22 mm, step 33/30 mm, slip 61.6; ROM L/R
  hip_pitch 15.0/21.7 (6.7 deg gap, largest), others within 2.7. standpush 0/32. Video: upright, fwd/back/turn/stand.

- 11-06 21:2x v6 cont1 FINISHED (cumulative 480k). Last sixth TB +1% (track_lin .778 -> .788, track_ang 1.480 -> 1.489).
  agent_240000 (eval_v6c1_240000_score/push.log, video vel_v6c1_240000.mp4, kit render): falls 0/64; fwd +.059, back
  -.075, diag vx +.044, turn +.297, stand -.004 (0 lifts), side vy -.013 (no lateral). L/R swing max xy 1000/1008,
  touchdown 545/512, peak 20/22 mm, step 32/28 mm, slip 59.8; ROM hip_pitch 14.7/20.3, others within 2.3.
  standpush 2/32 (0.5 m/s 2/8). Video: upright, fwd/back/turn/stand.
  Eval not better than agent_174000 (fwd .063, push 0/32) -> plateau: no further continuation, cron f03cee68 deleted.
  Candidates: v7 17-24-53 agent_240000 (softest swings 779/834, push 0/32, fwd .055) vs v6 07-58-49 agent_210000
  (fastest fwd .075, push 0/32) vs v6 cont1 agent_174000 (balanced fwd .063 / back -.073, push 0/32).

- 11-07 08:49 user: "계속 진행해" -> option 1, v7 cont1 (same v7 config: swing_speed_weight back to -1.0) from v7
  17-24-53 agent_240000, env.curriculum_step_offset=480000, agent.seed=7, 240k more. Run 2026-11-07_08-49-37_ppo_torch,
  PID 820595, log logs/train/train_vel_microduck_v7_cont1.log (checkpoint load, seed 7, [SYMMETRY] .2 seen).

- 11-07 09:1x v7 cont1 check (run 08-49-37, PID 820595 alive, 37k/240k, 26 it/s: GPU shared with an H1 vel training
  PID 820964 started outside this session, left alone). TB flat: track_lin .80-.81, track_ang 1.52, swing_speed
  -.013 -> -.011, std .024, symmetry .001, LR normal; reward 100 -> 117 (first sixth = restart transient).
  agent_36000 (cumulative 516k; eval_v7c1_36000_score/push.log, video vel_v7c1_36000.mp4): falls 0/64; fwd +.053,
  back -.081, diag vx +.037, turn +.290, stand -.004 (0 lifts), side vy -.016 (no lateral). L/R swing max xy 780/777
  (v7 779/834, now symmetric), touchdown 532/535 (v7 525/552), peak 22/23 mm, step 30/24 mm, slip 52.2; ROM hip_pitch
  14.6/18.6, others within 1.5. standpush 0/32. Video: upright, fwd/back/turn/stand. Same as v7 final, L/R more even.

- 11-07 10:3x v7 cont1 check (PID 820595 alive, 121k/240k, ~25 it/s, H1 run still sharing GPU). TB: tracking flat
  (lin .807/.801, ang 1.52), swing_speed -.012 -> -.008 (still falling), reward 121.6, std .024, LR normal.
  agent_120000 (cumulative 600k; eval_v7c1_120000_score/push.log, video vel_v7c1_120000.mp4): falls 0/64; fwd +.052,
  back -.081, diag vx +.040, turn +.292, stand -.004 (0 lifts), side vy -.015. L/R swing max xy 686/728 mm/s (v7
  779/834, v6 991/1033), touchdown 521/554, peak 23/23 mm, step 31/25 mm, slip 52.1; ROM L/R within 2.4 deg
  (hip_pitch 15.6/18.0). standpush 2/32 (0.4 1/8, 0.5 1/8; 36000 was 0/32). Video: upright, fwd/back/turn/stand.
  Swings keep getting softer with tracking held; push slightly worse.

- 11-07 11:3x v7 cont1 check (PID 820595 alive, 206k/240k, ~22 min left). TB: tracking flat (lin .805, ang 1.532),
  swing_speed -.008 -> -.007 (flattening), reward 122.0, std .024, LR normal.
  agent_204000 (cumulative 684k; eval_v7c1_204000_score/push.log, video vel_v7c1_204000.mp4): falls 0/64; fwd +.050,
  back -.086, diag vx +.037, turn +.296, stand -.005 (0 lifts), side vy -.016. L/R swing max xy 716/703 (120000:
  686/728), touchdown 532/548, peak 23/24 mm, step 31/26 mm, slip 50.8; ROM L/R within 2.6 deg. standpush 1/32
  (0.5 1/8). Video: upright, fwd/back/turn/stand. Mostly plateaued vs 120000.

- 11-07 12:1x v7 cont1 FINISHED (cumulative 720k). Last sixth flat (track_lin .808 -> .804, ang 1.532 -> 1.533,
  swing_speed -.006, reward 122.0).
  agent_240000 (eval_v7c1_240000_score/push.log, video vel_v7c1_240000.mp4, kit render): falls 0/64; fwd +.055, back
  -.078, diag vx +.043, turn +.306, stand -.004 (0 lifts), side vy -.009 (no lateral). L/R swing max xy 708/699 mm/s,
  touchdown 529/556, peak 23/25 mm, step 34/29 mm; ROM L/R within 2.8 deg. standpush 0/32. Video: upright,
  fwd/back/turn/stand.
  PLATEAU -> no further continuation, cron 0c0b3a57 deleted.
  Deployment candidate: v7 cont1 08-49-37 agent_240000 (softest symmetric swings 708/699, push 0/32, all tracking
  except lateral). Alternatives: v6 07-58-49 agent_210000 (fastest fwd .075, swings ~1000).

- 11-09 09:58 user: "2 진행해" -> v8 lateral retry: fine-tune from v7 cont1 08-49-37 agent_240000 with
  track_lat_vel_weight 0 -> 1.0 (std 0.06); everything else = v7 (swing_speed -1.0, symmetry 0.2, alive/termination 0,
  v2 log-std bounds). env.curriculum_step_offset=720000, agent.seed=7, 240k more. Run 2026-11-09_09-58-13_ppo_torch,
  PID 938814, log logs/train/train_vel_microduck_v8_lat.log (checkpoint load, seed 7, [SYMMETRY] .2 seen), ~42 it/s
  (GPU free, H1 run gone). Cron 4699b3bd (:17/:47). Baseline side vy: v7c1 240000 -.009.

- 11-09 10:2x v8 check (PID 938814 alive, ~36k/240k at check time). Speed fell 42 -> 24 it/s: a new external H1 vel
  training (PID 941153, Template-Arti-H1-Velocity-Direct-v0, started outside this session) shares the GPU; left alone.
  TB rising: track_lat_vel .46 -> .55, track_lin .89 -> 1.03, track_ang 1.52 -> 1.45 (slight drop), swing_speed -.004,
  action_rate -.10, reward 102 -> 124, LR normal.
  agent_30000 (cumulative 750k; eval_v8_30000_score/push.log, video vel_v8_30000.mp4): falls 0/64; fwd +.060,
  back -.021 (v7c1 -.078: regressed), side_left vy +.004 (v7c1 -.009: right sign, ~7% of .06), diag vx +.047 vy -.014
  (cmd -.05), turn +.238 (v7c1 .306), stand 0 lifts. Swing max L/R 715/696, touchdown 525/546, ROM L/R within 1 deg.
  standpush 1/32 (one fall at .5 m/s). Video: all upright, walking/turning, no visible sideways stepping yet.
  Early: lateral term rising but backward/turn traded away; keep watching.
