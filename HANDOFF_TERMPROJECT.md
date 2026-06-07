나는 윤지훈 (SNU, hubo3125@snu.ac.kr), 졸업논문으로 task-conditional dexterous
grasp pose generation 을 진행 중인 사람이야. 이번에 모션플래닝 수업 term project
가 있는데, 너랑 작업할 내용은 다음과 같아.

═══════════════════════════════════════════════════════════════
[1] 프로젝트 주제 (확정)
═══════════════════════════════════════════════════════════════
"Affordance-grasp 조건부 spill-free trajectory optimization
 for liquid transport"

- Object: 머그컵 (단일 카테고리)
- Input: mug mesh + 졸논 reward model 이 고른 grasp pose (taxonomy/region
  반영) + start/goal end-effector pose
- Output: 6-DoF end-effector SE(3) trajectory 가
    (a) tilt angle ≤ θ_max
    (b) 유효중력벡터 (g − a_ee) 가 mug rim cone 안 (no-spill)
    (c) obstacle 회피
    (d) smooth
- Approach: CHOMP / STOMP / MPPI 등 trajectory optimization 에
  spill cost term 을 추가. 강의에서 다룬 framework 그대로.
- 평가: 3-way ablation
    Baseline 1: Linear interp
    Baseline 2: CHOMP (smooth + obs cost, no spill)
    Ours: CHOMP + spill cost
  Metric: spill ratio, traj length, smoothness, plan time
- Demo: sapien/PyBullet 컵 위 파티클 N개 → "rim 넘은 개수" 카운트
  (full fluid sim 없이 시각적 직관)

═══════════════════════════════════════════════════════════════
[2] 일정 (TIGHT)
═══════════════════════════════════════════════════════════════
- 2026-06-08 (내일) 수업: Proposal 발표 3–4분 ← 최우선
- 2026-06-15 수업: Final lightning talk 3–4분
- 2026-06-15 마감: Final report (3–6 page, conference style)

평가: technical depth 40 / experimental validation 40 / clarity 20
+ original/creative idea bonus

═══════════════════════════════════════════════════════════════
[3] 너의 역할 (다른 Claude 세션은 코딩 담당)
═══════════════════════════════════════════════════════════════
우선순위 순:

1. Proposal 슬라이드 (내일 발표용, 3–4분, PDF/PPT 둘 다 제출)
   - 문제 정의 → motivation → 방법 → 예상 결과 → schedule
   - 5–7 슬라이드. Marp 권장 (졸논 미팅 자료가 Marp 였음)
   - 화려하지 않게, 명확하게

2. 선행연구 조사 (proposal + final report 모두에 들어감)
   - "Waiter motion planning"
   - Non-prehensile liquid transport
   - CHOMP / STOMP extensions with task-specific costs
   - Tilt-constrained trajectory optimization
   - DLO/sloshing-aware planning (있다면)
   - 우리와의 차별점 명시 → 우리는 "affordance-aware grasp 가
     spill 여유에 영향" 을 contribution 으로 잡고 싶음

3. Final lightning talk 슬라이드 (proposal 슬라이드 기반 + 실험 결과)

4. Final report (3–6 page, conference style: intro / related work /
   method / experiments / results / conclusion)

═══════════════════════════════════════════════════════════════
[4] 프로젝트 폴더 (독립 repo)
═══════════════════════════════════════════════════════════════
/home/dongjae/motion_planning_termproject/   ← 이 프로젝트 root (독립 git)
├── README.md
├── HANDOFF_TERMPROJECT.md    ← 이 파일
├── CLAUDE.md                 ← 코딩 세션 가이드
├── code/                     ← CHOMP/STOMP/spill cost 구현 (코딩 세션)
├── sim/                      ← sapien/pybullet scene (코딩 세션)
├── data/                     ← mesh + grasp pose (졸논에서 1회 복사)
├── results/                  ← ablation 출력 (코딩 세션)
├── figs/                     ← figure (공유 — 코딩 세션이 채움, 너가 사용)
└── docs/                     ← 너의 작업 공간
    ├── slides_proposal.md
    ├── slides_proposal.pdf
    ├── slides_final.md
    ├── slides_final.pdf
    ├── report.tex (or .md)
    ├── related_work_notes.md
    └── _scripts/

폴더 충돌 방지:
- 코딩 세션: code/, sim/, data/, results/ 수정. figs/ 에 figure 저장.
- 너 (자료 세션): docs/ 만 수정. figs/ 는 읽기만.
- 양쪽 다: README.md, HANDOFF_TERMPROJECT.md, CLAUDE.md 는 함부로 수정 X
  (필요하면 사용자에게 먼저 확인)

졸논 repo 와의 관계:
- /home/dongjae/UltraDexGrasp/UltraDexGrasp_jihoon/ 는 READ-ONLY 참조만.
  절대 수정 X.

═══════════════════════════════════════════════════════════════
[5] 참고 자료 (외부, 이미 로컬에 있음)
═══════════════════════════════════════════════════════════════
- 수업 강의자료: ~/Downloads/motionplanning/
  * 1–3 Path planning, 4 Constraints, 5–6 Traj/poly, 7 Collision,
    8 CHOMP, 9 Drone corridor, 10 iLQR, 11 LinMPC, 12 NMPC, 13 NMPC-contact
- 졸논 프로젝트 (참고만, 수정 X): /home/dongjae/UltraDexGrasp/UltraDexGrasp_jihoon/
  * CLAUDE.md — 졸논 한 줄 요약 + 데이터/모델 위치 + 환경
  * docs/meeting_2026-06-04/ — Marp 슬라이드 style 참고 (slides.md, figs/)
  * reward_model.pt — Stage 1 reward model (mug, testset 91%)
  * data_grasp_pose/mug/mug_0/ — 900 grasp pose

═══════════════════════════════════════════════════════════════
[6] 스타일 / 톤
═══════════════════════════════════════════════════════════════
- 짧고 명확하게, 한국어로 소통
- figure / table 화려하지 않게 (학회 paper 스타일)
- 슬라이드는 Marp; 편집 가능한 pptx 가 필요하면 python-pptx 별도 생성
- 영어로 작성: report 본문, figure 캡션 (슬라이드는 자유)

═══════════════════════════════════════════════════════════════
[7] 첫 작업
═══════════════════════════════════════════════════════════════
1. ~/Downloads/motionplanning/ 의 강의자료 훑고 CHOMP / 강의 슬라이드
   스타일 파악
2. 선행연구 빠르게 조사 (WebSearch 활용) → docs/related_work_notes.md
3. Proposal 슬라이드 초안 (5–7 슬라이드 Marp) 작성
   → 내일 수업 시간 전까지 완성

OK, 시작하자. 먼저 [7].1 강의자료 확인부터.
