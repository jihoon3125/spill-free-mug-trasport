# Motion Planning Term Project — 코딩 세션 가이드

## 한 줄
**Affordance-grasp 조건부 spill-free trajectory optimization for liquid transport (mug)** — 졸논 grasp pose 를 입력으로 받아, 컵 안의 물이 쏟아지지 않는 6-DoF end-effector trajectory 를 생성.

## 사용자
- 윤지훈 (SNU, hubo3125@snu.ac.kr)
- SNU 모션플래닝 수업 (H. Jin Kim) term project

## 일정
- 2026-06-08 (내일) 수업: Proposal 발표 3–4분
- 2026-06-15 수업: Final lightning talk 3–4분 + Report 마감 (3–6 page)

## 평가
- Technical depth 40% / Experimental validation 40% / Clarity 20%
- + Original/creative bonus

## 폴더 구조
```
/home/dongjae/motion_planning_termproject/   ← root (독립 git repo)
├── code/        ← 이 세션 작업: CHOMP/STOMP/spill cost 구현
├── sim/         ← 이 세션 작업: sapien/pybullet scene
├── data/        ← 졸논에서 1회만 복사 (mug mesh, grasp pose)
├── results/     ← ablation 출력 (CSV/JSON)
├── figs/        ← 결과 figure (PNG) — 자료 세션이 가져다 사용
└── docs/        ← 자료 세션 영역 (수정 X)
```

## 졸논 repo 와의 관계
- /home/dongjae/UltraDexGrasp/UltraDexGrasp_jihoon/ 는 READ-ONLY 참조만
- 절대 수정하지 않음
- 필요한 것 (mesh, grasp pose) 은 `data/` 로 1회 복사

## 작업 분담
- **이 세션 (코딩)**: code/, sim/, data/, results/, figs/ 수정
- **자료 세션 (다른 Claude)**: docs/ 만 수정. figs/ 는 읽기만.
- 공유: README.md, HANDOFF_TERMPROJECT.md, CLAUDE.md 는 사용자 확인 후 수정

## 코딩 현황 (2026-06-07 저녁 기준)

### ✅ 완료
1. **Data 추출** — `data/mug_mesh.obj`, `data/grasp_info.txt` (whole_200), `data/ee_to_mug.npy`, `data/qpos_transport_start.npy`, `data/xhand_qpos.npy`
2. **Sapien scene** — `sim/scene.py` (UR5e + xhand + mug + table). render headless 검증 (`figs/scene_init.png`).
3. **PyTorch FK** — `mp/kinematics.py` URDF chain walking. sapien 결과와 10⁻⁷ 오차로 검증 (`mp/_test_fk.py`).
4. **IK** — `mp/ik.py` Adam-based gradient IK. pos err < 1mm, rot err < 1e-4. `mp/make_transport_pose.py` (upright transport 자세 생성, `figs/scene_transport_start.png`).
5. **Spill cost** — `mp/spill_cost.py`. 효과적 중력 vs mug local +y axis. mesh 가 Y-up 컨벤션. θ_max=18° default.
6. **CHOMP** — `mp/chomp.py`. Joint-space, CHOMP 의 M^-1 preconditioner + Adam + gradient clipping (안정성). min-jerk baseline 도 같이 제공.
7. **3-way ablation runner** — `mp/run_ablation.py`. 결과 `results/ablation.npz` + `results/ablation.csv`.
8. **Figures** — `mp/make_figures.py`. 4종:
   - `figs/fig_tilt_timeseries.png` ★ — 3 method tilt over time + threshold
   - `figs/fig_spill_ratio.png` ★ — bar chart (66% / 42% / 0%)
   - `figs/fig_mug_traj_3d.png` — 3 method mug 위치 3D
   - `figs/fig_summary_table.png` — table

### 📊 핵심 결과 (1초간 70cm 이동, θ_max=18°)
| Method | Spill ratio | Max tilt | Mean tilt |
|---|---|---|---|
| Min-jerk (naive) | 66% | 38.8° | 23.3° |
| CHOMP (smooth) | 42% | 21.2° | 15.6° |
| STOMP + spill (gradient-free) | 14% | 32.1° | 11.9° |
| **CHOMP + spill (ours)** | **0%** | **15.1°** | **11.6°** |

### 📈 Multi-speed robustness (5 time budgets × 4 methods)
T=0.5/0.8/1.0/1.5/2.0s 에서 spill ratio:
- Min-jerk: 100% → 88 → 66 → 32 → 22%
- CHOMP smooth: 100 → 56 → 42 → 26 → 20%
- STOMP+spill: 16 → 28 → 34 → 48 → 40%
- **CHOMP+spill (ours): 0 → 4 → 0 → 0 → 0%** ← 모든 T 에서 견고

### ✅ 추가 완료 (저녁 D-C-B-A 진행 후)
1. ✅ figs/README.md 갱신 — 자료 세션 매핑 가이드
2. ✅ Git initial commit (543d087) + 메모리 갱신
3. ✅ Animation GIF (3-method side-by-side, 50frame, 25fps)
4. ✅ Obstacle scenario (SDF cost) — 3-way trade-off 검증

### 📊 Obstacle scenario 결과
| Method | Spill | Collision |
|---|---|---|
| Min-jerk | 32% | **25%** (penetrates) |
| CHOMP smooth+obs | **53%** | 0% (detour induces spill!) |
| **CHOMP+obs+spill (ours)** | **0%** | **0%** |

→ Obstacle 만 막으면 우회 → 더 큰 spill. Spill+Obstacle 동시 처리가 필요한 사실.

### ⏳ 남은 작업 (선택)
1. Quantitative particle spill metric (단순 binary 가 아닌 회복 가능 비율)
2. Plan-time 그래프 (CHOMP iter 별 수렴 곡선)
3. 다양한 grasp pose 로 robust 성 평가

## 핵심 수식 (spill cost)

End-effector 위치 p(t), 회전 R(t) → mug 의 z-axis 단위벡터 n(t) = R(t) e_z
유효중력 g_eff(t) = g - a(t) (가속도 a(t) = p̈(t))
유효중력 정규화: ĝ_eff(t) = g_eff(t) / ||g_eff(t)||

**No-spill 조건**: ĝ_eff(t) · n(t) ≥ cos(θ_max)
  (즉 유효중력 방향이 mug 의 위쪽 cone 안에 들어와야 함)

**Spill cost (smooth penalty)**:
  c_spill(t) = max(0, cos(θ_max) - ĝ_eff(t) · n(t))²

전체 cost:
  J = α · J_smooth + β · J_obs + γ · J_spill
       ──────────────  ───────  ──────────
        CHOMP 기본       기본    추가 (ours)

## 의존성 (예상)
- sapien (졸논 환경) 또는 pybullet
- numpy, scipy
- trimesh
- matplotlib (figure)

## 사용자 선호 (졸논 작업에서 학습)
- 짧고 명확한 응답
- 화려한 figure X, 학회 paper 스타일
- 한국어로 소통
- 코멘트는 최소화 (왜 / 비자명한 것만)
- emoji 사용 X
- 영어로 작성하는 부분: figure 캡션, code docstring (선택), report

## 첫 단계
1. data/ 에 mug mesh 1개 복사 (졸논 mug_0)
2. data/ 에 grasp pose 1개 추출 (졸논 reward model top-1)
3. sim/ 에 sapien scene 셋업 시도
