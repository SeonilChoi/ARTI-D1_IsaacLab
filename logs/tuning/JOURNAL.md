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
| 2026-10-31_13-12-41 | extra liftoff penalty -10 per liftoff after a foot's first in its gait cycle (1cca970), warm start 12-38-06 agent_80000, offset 1680000 | running (log train_newton_extraliftoff.log) |

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
