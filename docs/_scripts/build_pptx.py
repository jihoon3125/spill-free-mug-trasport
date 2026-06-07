#!/usr/bin/env python
"""Build a fully EDITABLE PowerPoint (native text boxes + tables + images)
for the term-project proposal. Marp's --pptx bakes slides into images;
this keeps every title/bullet/table editable in PowerPoint.

Output: docs/slides_proposal_editable.pptx
"""
import os
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.abspath(os.path.join(HERE, ".."))
FIGS = os.path.join(DOCS, "figs")

NAVY = RGBColor(0x1A, 0x36, 0x5D)
SLATE = RGBColor(0x2C, 0x3E, 0x50)
RED = RGBColor(0xC0, 0x39, 0x2B)
DRED = RGBColor(0xA9, 0x32, 0x26)
GRAY = RGBColor(0x66, 0x66, 0x66)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
REDBG = RGBColor(0xFB, 0xEC, 0xEB)
HEADBG = RGBColor(0x1A, 0x36, 0x5D)
ROWBG = RGBColor(0xF2, 0xF5, 0xF9)
FONT = "Arial"

SW, SH = 13.333, 7.5
prs = Presentation()
prs.slide_width = Inches(SW)
prs.slide_height = Inches(SH)
BLANK = prs.slide_layouts[6]


def slide():
    return prs.slides.add_slide(BLANK)


def textbox(s, text, left, top, width, height, size=18, color=SLATE,
            bold=False, italic=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
            line_spacing=1.05):
    tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    lines = text if isinstance(text, list) else [text]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        runs = ln if isinstance(ln, list) else [(ln, {})]
        for j, (t, st) in enumerate(runs):
            r = p.add_run()
            r.text = t
            r.font.name = st.get("font", FONT)
            r.font.size = Pt(st.get("size", size))
            r.font.bold = st.get("bold", bold)
            r.font.italic = st.get("italic", italic)
            r.font.color.rgb = st.get("color", color)
    return tb


def title(s, text, sub=False):
    textbox(s, text, 0.55, 0.3, 12.2, 0.95, size=30, color=NAVY, bold=True,
            anchor=MSO_ANCHOR.MIDDLE)
    ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(1.22),
                            Inches(12.1), Pt(2.2))
    ln.fill.solid(); ln.fill.fore_color.rgb = RGBColor(0xCC, 0xCC, 0xCC)
    ln.line.fill.background()
    return 1.45


def pagenum(s, n):
    textbox(s, str(n), 12.7, 7.0, 0.5, 0.35, size=12, color=GRAY, align=PP_ALIGN.RIGHT)


def place_image(s, name, top, max_w=12.0, max_h=4.6, center_x=None):
    path = os.path.join(FIGS, name)
    iw, ih = Image.open(path).size
    ar = iw / ih
    w = max_w; h = w / ar
    if h > max_h:
        h = max_h; w = h * ar
    left = (SW - w) / 2 if center_x is None else center_x
    s.shapes.add_picture(path, Inches(left), Inches(top), Inches(w), Inches(h))
    return top + h  # bottom


def callout(s, text, left, top, width, height):
    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top),
                             Inches(width), Inches(height))
    box.fill.solid(); box.fill.fore_color.rgb = REDBG
    box.line.color.rgb = RED; box.line.width = Pt(1.5)
    box.shadow.inherit = False
    tf = box.text_frame; tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
    for t, st in text:
        r = p.add_run(); r.text = t
        r.font.name = FONT; r.font.size = Pt(st.get("size", 17))
        r.font.bold = st.get("bold", False); r.font.color.rgb = st.get("color", SLATE)
    return box


# ============================ SLIDE 1 — Title ============================
s = slide()
textbox(s, "Affordance-Grasp Conditioned\nSpill-Free Trajectory Optimization".split("\n"),
        0.9, 2.35, 11.5, 1.7, size=36, color=NAVY, bold=True, align=PP_ALIGN.LEFT)
textbox(s, [[("Does ", {}), ("how you grasp", {"bold": True, "color": NAVY}),
             (" a cup change ", {}),
             ("how safely you can carry", {"bold": True, "color": NAVY}), (" it?", {})]],
        0.92, 4.15, 11.5, 0.6, size=20, color=SLATE)
textbox(s, "Jihoon Yun   ·   Motion Planning   ·   2026-06-08",
        0.92, 5.0, 11.5, 0.5, size=15, color=GRAY)

# ====================== SLIDE 2 — Big picture ===========================
s = slide()
t = title(s, "My thesis picks the grasp — this project plans the carry")
b = place_image(s, "overview_thesis_project.png", t + 0.05, max_w=12.2, max_h=4.7)
textbox(s, [[("My thesis (left): ", {"bold": True, "color": NAVY}),
             ("object mesh + a natural-language task → a dexterous grasp (same mug, the task "
              "picks a different grasp).   ", {}),
             ("This project (right): ", {"bold": True, "color": DRED}),
             ("take that grasp + start/goal poses → plan a spill-free carry trajectory.", {})]],
        0.7, b + 0.12, 12.0, 0.9, size=13, color=SLATE)
pagenum(s, 2)

# ========================= SLIDE 3 — Problem ============================
s = slide()
title(s, "Problem — carry liquid from A to B without spilling")
textbox(s, "Given a grasped mug, plan a 6-DoF end-effector trajectory that is:",
        0.7, 1.6, 12.0, 0.5, size=19, color=SLATE)
textbox(s, [[("(a) ", {"bold": True}), ("spill-free      ", {}),
             ("(b) ", {"bold": True}), ("obstacle-avoiding      ", {}),
             ("(c) ", {"bold": True}), ("smooth      ", {}),
             ("(d) ", {"bold": True}), ("tilt ≤ θmax", {})]],
        0.95, 2.3, 11.5, 0.6, size=18, color=SLATE)
callout(s, [("Even a perfectly upright cup spills", {"bold": True, "size": 19}),
            (" under a fast stop or turn — the danger is ", {"size": 18}),
            ("motion", {"size": 18, "color": RED, "bold": True}),
            (", not just orientation.", {"size": 18})],
        0.9, 3.5, 11.5, 1.2)
textbox(s, "Why? — next slide.", 0.95, 5.2, 6.0, 0.5, size=14, color=GRAY, italic=True)
pagenum(s, 3)

# ====================== SLIDE 4 — Key insight ===========================
s = slide()
t = title(s, "Key insight — liquid follows effective gravity  g − a")
b = place_image(s, "spill_cone.png", t + 0.1, max_w=11.0, max_h=4.4)
textbox(s, [[("No-spill ", {"bold": True, "color": NAVY}),
             ("⇔ the apparent-up vector  a_cup − g  stays inside the cup's ", {}),
             ("rim cone", {"bold": True}),
             ("  (half-angle θmax).  Acceleration is evaluated at the cup, not the wrist.", {})]],
        0.7, b + 0.15, 12.0, 0.7, size=14, color=SLATE)
pagenum(s, 4)

# ==================== SLIDE 5 — Why grasp matters =======================
s = slide()
t = title(s, "Why the grasp matters — it reshapes the spill margin")
b = place_image(s, "grasp_mechanism.png", t + 0.05, max_w=10.5, max_h=3.05)
textbox(s, [
    [("• The grasp fixes  T(hand→cup): ", {"bold": True}),
     ("the cup's offset r and orientation relative to the wrist.", {})],
    [("• Same arm motion → different cup acceleration  a_cup  (lever effect) → "
      "different spill margin.", {})],
    [("• Plus arm reachability: some grasps keep the cup upright easily, others don't.", {})],
    [("• The two effects can conflict → ", {}),
     ("which grasp is safest is measured, not assumed", {"bold": True, "color": DRED}),
     ("  (see Method).", {})],
], 0.7, b + 0.15, 12.1, 2.2, size=15, color=SLATE, line_spacing=1.15)
pagenum(s, 5)

# ====================== SLIDE 6 — Related work =========================
s = slide()
title(s, "Related work & the gap")
rows = [
    ["Line of work", "What they do", "Gap"],
    ["Anti-slosh / spill-free planning\n(Muchacho '22, Gandhi '24)",
     "spill-free trajectory for a fixed container pose",
     "grasp is a fixed boundary condition"],
    ["Waiter / non-prehensile\n(Heins '23, Selvaggio '23)",
     "keep object from tipping / sliding (friction cone)",
     "rigid object, no liquid, no grasp choice"],
    ["Affordance / task-oriented grasping\n(\"Grasping for a Purpose\" '14, Dang '12)",
     "pick a task-appropriate grasp",
     "stops at selection — never carries it"],
]
tbl_shape = s.shapes.add_table(4, 3, Inches(0.7), Inches(1.55),
                               Inches(11.9), Inches(3.2))
tbl = tbl_shape.table
tbl.columns[0].width = Inches(4.1)
tbl.columns[1].width = Inches(4.2)
tbl.columns[2].width = Inches(3.6)
for ci in range(3):
    for ri in range(4):
        cell = tbl.cell(ri, ci)
        cell.margin_left = Inches(0.12); cell.margin_right = Inches(0.1)
        cell.margin_top = Inches(0.06); cell.margin_bottom = Inches(0.06)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.fill.solid()
        cell.fill.fore_color.rgb = HEADBG if ri == 0 else (ROWBG if ri % 2 else WHITE)
        tf = cell.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]
        r = p.add_run(); r.text = rows[ri][ci]
        r.font.name = FONT
        r.font.size = Pt(14 if ri == 0 else 12.5)
        r.font.bold = (ri == 0)
        r.font.color.rgb = WHITE if ri == 0 else SLATE
textbox(s, [[("Nobody connects grasp choice to a physical spill margin. ",
              {"bold": True, "color": DRED, "size": 17}),
             ("We make the trajectory optimizer ", {"size": 16}),
             ("score grasps", {"bold": True, "size": 16}),
             (" by how safely they carry.", {"size": 16})]],
        0.7, 5.05, 12.0, 0.9, size=16, color=SLATE)
pagenum(s, 6)

# ================== SLIDE 7 — Method: spill-aware CHOMP =================
s = slide()
t = title(s, "Method — spill-aware CHOMP")
b = place_image(s, "method_pipeline.png", t + 0.1, max_w=12.2, max_h=3.0)
textbox(s, [[("J(ξ) = J_smooth + J_obs + λₛ · J_spill",
              {"bold": True, "size": 20, "color": NAVY}),
             ("          ξ ← ξ − η A⁻¹ ∇J(ξ)",
              {"size": 18, "color": SLATE})]],
        0.7, b + 0.2, 12.0, 0.6, size=20, align=PP_ALIGN.CENTER)
textbox(s, [[("Course CHOMP (Lecture 8) + ", {}),
             ("one differentiable spill term", {"bold": True, "color": DRED}),
             (".  a_cup comes from finite differences of ξ — the same operator as the "
              "smoothness term, so the spill gradient is analytic.", {})]],
        0.7, b + 0.95, 12.0, 0.8, size=14, color=SLATE, align=PP_ALIGN.CENTER)
pagenum(s, 7)

# ================== SLIDE 8 — Method: grasp evaluator ==================
s = slide()
t = title(s, "Method — the optimizer as a grasp evaluator")
b = place_image(s, "grasp_evaluator.png", t + 0.1, max_w=12.0, max_h=4.0)
textbox(s, [[("Run the same spill-aware planner ", {}),
             ("per grasp", {"bold": True, "color": NAVY}),
             (" → each grasp gets an achievable spill-margin score; the optimizer becomes a "
              "downstream-aware ", {}),
             ("grasp evaluator", {"bold": True, "color": DRED}),
             (".  Future (thesis): feed that score back into grasp generation — closing the loop.",
              {})]],
        0.7, b + 0.12, 12.0, 0.9, size=13.5, color=SLATE)
pagenum(s, 8)

# ===================== SLIDE 9 — Evaluation & plan =====================
s = slide()
t = title(s, "Evaluation & plan")
b = place_image(s, "ablation_plan.png", t + 0.05, max_w=11.5, max_h=3.0)
textbox(s, [
    [("• 3-way ablation: ", {"bold": True}),
     ("linear interp / CHOMP / CHOMP + spill — on a mug, with and without obstacles.", {})],
    [("• Metrics: ", {"bold": True}),
     ("spill ratio, path length, smoothness (jerk), plan time; run across grasps to show the "
      "margin differs.", {})],
    [("• Demo: ", {"bold": True}),
     ("SAPIEN particles on the liquid surface → count how many cross the rim.", {})],
    [("• Timeline: ", {"bold": True}),
     ("implementation + ablation within ~1 week  (proposal 06-08 → report 06-15).", {})],
], 0.7, b + 0.12, 12.1, 2.1, size=14.5, color=SLATE, line_spacing=1.12)
pagenum(s, 9)

out = os.path.join(DOCS, "slides_proposal_editable.pptx")
prs.save(out)
print("wrote", out, "(", len(prs.slides._sldIdLst), "slides )")
