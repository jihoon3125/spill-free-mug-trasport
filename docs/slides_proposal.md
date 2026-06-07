---
marp: true
theme: default
paginate: true
size: 16:9
math: katex
style: |
  section { font-family: 'Noto Sans KR', sans-serif; padding: 40px 60px; }
  h1 { color: #1a365d; }
  h2 { color: #2c3e50; border-bottom: 2px solid #ccc; padding-bottom: 4px; }
  table { font-size: 0.84em; }
  img { max-height: 500px; }
  .small { font-size: 0.8em; color: #555; }
  .hl { color: #c0392b; font-weight: bold; }
  .sub { font-size: 1.15em; color: #2c3e50; }
  .cols { display: flex; gap: 30px; align-items: center; }
  .cols > div { flex: 1; }
  .box { background: #fbeceb; border-left: 5px solid #c0392b; padding: 10px 18px; border-radius: 4px; }
  .tag { background: #1a365d; color: #fff; padding: 2px 10px; border-radius: 3px; font-size: 0.7em; }
  ul { line-height: 1.5; }
---

<!-- _paginate: false -->

# Affordance-Grasp Conditioned<br>Spill-Free Trajectory Optimization

<span class="sub">Does <b>how you grasp</b> a cup change <b>how safely you can carry</b> it?</span>

<br>

<span class="tag">Motion Planning · Proposal</span> &nbsp; Jihoon Yun · 2026-06-08

---

## Where this project fits — from grasp generation to the carry

![w:1120](figs/overview_thesis_project.png)

<span class="small"><b>My thesis (left):</b> object mesh + a natural-language task → a dexterous grasp (same mug, the task picks a different grasp). &nbsp; <b>This project (right):</b> take that grasp + start/goal poses → plan a <b>spill-free</b> carry trajectory.</span>

---

## Problem — carry liquid from A to B without spilling

Given a grasped mug, plan a 6-DoF end-effector trajectory that is:

- **(a)** spill-free &nbsp;&nbsp; **(b)** obstacle-avoiding &nbsp;&nbsp; **(c)** smooth &nbsp;&nbsp; **(d)** tilt $\le \theta_{max}$

<br>

<div class="box">

**Even a perfectly upright cup spills** under a fast stop or turn — the danger is *motion*, not just orientation.

</div>

<br>

<span class="small">Why? That is the next slide.</span>
