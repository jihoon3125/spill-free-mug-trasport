# figs/ — 발표/리포트용 figure 모음

자료 세션 (다른 Claude Code) 가 사용. 코딩 세션이 채우고, 자료 세션이 슬라이드/리포트에 가져다 씀.

## 핵심 결과 요약 (한 문장)

**1초간 70 cm 이동 (mug 운반) — naive (min-jerk) 는 66% 시간 동안 spill, CHOMP+smooth 는 42%, STOMP+spill 은 14%, CHOMP+spill (ours) 는 0%.** 모든 시간 budget (0.5–2.0초) 에서 ours 만 ~0% spill 달성.

## Figure 인벤토리 (2026-06-07 밤)

### ★ 핵심 결과 (proposal + final 둘 다 사용)
| 파일 | 용도 | 비고 |
|---|---|---|
| `fig_tilt_timeseries.png` | tilt(t) for 4 methods + θ_max threshold. 가장 임팩트 큰 single-T 결과 | 4-way |
| `fig_spill_ratio.png` | spill ratio bar 4-method | 66% / 42% / 14% / 0% |
| `fig_multispeed.png` | T ∈ {0.5, 0.8, 1.0, 1.5, 2.0} × 4 method (2 panel: spill ratio + max tilt) | 4-way × 5 budget — paper-quality |
| `fig_summary_table.png` | 4-method table (Method / Spill / Max tilt / Mean tilt) | T=1s scenario |

### 정성적 / 시각 자료
| 파일 | 용도 | 비고 |
|---|---|---|
| `scene_init.png` | 졸논에서 가져온 raw grasp pose (mug 옆으로 누움) | "데이터 출처" |
| `scene_transport_start.png` | IK 로 풀어 mug upright + 손잡이 잡힌 transport 시작 자세 | 문제 정의 슬라이드 |
| `fig_3way_frames.png` | 3 method × 5 keyframe sapien 비교 (arm posture 시간 흐름) | 정성 비교 |
| `fig_particles.png` | 3 method × 5 keyframe + mug 내부 물방울 파티클 (blue=safe, red=spill) | 가장 직관적 데모 |
| `fig_mug_traj_3d.png` | 4 method mug 중심 3D trajectory | 보조 |

## 권장 슬라이드 매핑 (Proposal — 3–4분, 5–7 슬라이드)

| Slide | 내용 | 사용 figure |
|---|---|---|
| 1. Title | 제목 / 저자 / 수업 | — |
| 2. Problem & Motivation | "물체 transport 시 spill 안 나는 trajectory 가 필요. 졸논의 affordance-grasp 가 input." | `scene_transport_start.png` (오른쪽) |
| 3. Approach | CHOMP smooth + obs + **spill** cost. 수식 box. | (텍스트, box diagram — 자료 세션이 그림) |
| 4. Preliminary result | "1초 안에 70cm 운반 — CHOMP+spill 만 0% spill" | `fig_tilt_timeseries.png` (메인) |
| 5. Quantitative (4-way + multi-speed) | 4 method × 5 time budget — robust | `fig_multispeed.png` + `fig_summary_table.png` |
| 6. Schedule / next steps | 남은 8일: obstacle 시나리오, particle 정량 측정, report | — |

## 권장 슬라이드 매핑 (Final — 3–4분, 5–7 슬라이드)

대부분 동일하되 다음 추가:
- `fig_3way_frames.png` 또는 `fig_particles.png` (영상 대체 정성 비교)
- (Day 2 추가 작업) `fig_obstacle_*.png` — obstacle scenario 결과

## 권장 리포트 매핑 (3–6 page, conference style)

- **Methodology**: spill cost 수식 + CHOMP update rule 다이어그램
- **Setup**: `scene_transport_start.png`
- **Main result (single T)**: `fig_tilt_timeseries.png` + `fig_summary_table.png`
- **Robustness (multi-T)**: `fig_multispeed.png` ← 핵심 paper figure
- **Qualitative**: `fig_3way_frames.png` 또는 `fig_particles.png`
- **3D trajectory**: `fig_mug_traj_3d.png` (선택)

## 실험 셋업 (자료 세션이 텍스트로 인용할 때)

- **Object**: 머그 컵 (졸논 mug_0, scale 0.12, OBJ Y-up convention)
- **Robot**: UR5e 6-DoF (xhand 18-DoF 중 finger 12 joint 은 grasp config 로 고정)
- **Grasp**: `mug_0_left_whole_200` — whole-hand 손바닥으로 본체 측면 감싸기
- **Path**: start (0.5, −0.3, 0.25) → goal (0.5, 0.4, 0.40) — 70 cm lateral + 15 cm lift
- **Spill threshold**: θ_max = 18° (effective gravity vs mug local +y axis cone)
- **Trajectory**: N=50 waypoints, dt = T/(N−1)
- **Spill model**: quasi-static "waiter problem". `effective gravity = g - a_mug` 이 mug local up axis 와 이루는 각 > θ_max → spill 카운트.

## 방법별 하이퍼파라미터

| Method | 핵심 설정 |
|---|---|
| Min-jerk (naive) | 5차 다항식, rest-to-rest |
| CHOMP (smooth only) | α_smooth=1.0, γ_spill=0, Adam lr=0.005, 300 iter, M⁻¹ preconditioning, grad clip 10 |
| STOMP+spill | K=50 samples, n_iter=200, σ_init=0.04, decay=0.995, λ=0.2, γ_spill=5.0, noise correlated by Σ=M⁻¹ |
| **CHOMP+spill (ours)** | α_smooth=1.0, γ_spill=2.0, Adam lr=0.003, 1000 iter |

## Obstacle scenario (NEW)

`results/obstacle.npz` 기반 figure:

| 파일 | 용도 | 비고 |
|---|---|---|
| `fig_obstacle_metrics.png` | ★ 3-way spill+collision bar | 32%/25% / 53%/0% / **0%/0%** |
| `fig_obstacle_3d.png` | mug 중심 3D path + box | obstacle 회피 시각화 |

**핵심 story**: obstacle cost 만 추가하면 우회 → 더 큰 spill 발생 (32→53%).
spill cost 도 함께 추가하면 **둘 다 0%** 달성. → "두 cost 의 simultaneous optimization 이 필요한 사실" 증명.

## Animation GIF (NEW)

| 파일 | 용도 | 비고 |
|---|---|---|
| `anim_3way.gif` | ★ 3-method side-by-side 50frame@25fps GIF | 슬라이드/web 용 |
| `anim_minjerk.gif` / `anim_chomp.gif` / `anim_chomp_spill.gif` | per-method GIF | 개별 |

각 GIF 에 method label (상단) + 실시간 tilt 값 (하단, spill 시 빨강) annotation.
