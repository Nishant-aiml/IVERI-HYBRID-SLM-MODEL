"""
Generate Clean IEEE Formulas & Tables Companion DOCX
===================================================
Produces CLEAN_SUBMISSION_PACKAGE/IEEE_FORMULAS_AND_TABLES.docx
Contains all formal equations (1)-(18) and Tables I-V formatted strictly to IEEE standards.
"""

from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DOCX = REPO_ROOT / "CLEAN_SUBMISSION_PACKAGE" / "IEEE_FORMULAS_AND_TABLES.docx"

doc = docx.Document()

# Margins: Standard IEEE 0.75 in
for section in doc.sections:
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)

def set_cell_margins(cell, top=100, bottom=100, left=120, right=120):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_cell_shading(cell, color_hex):
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shd)

def add_heading(text, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.bold = True
    if level == 1:
        r.font.size = Pt(12)
        r.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    else:
        r.font.size = Pt(10.5)
        r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    return p

def add_equation(eq_text, eq_num):
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    # 6.0 inches equation, 1.0 inch number
    table.columns[0].width = Inches(5.8)
    table.columns[1].width = Inches(1.2)
    
    c0 = table.cell(0, 0)
    p0 = c0.paragraphs[0]
    p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r0 = p0.add_run(eq_text)
    r0.font.name = "Cambria Math"
    r0.font.size = Pt(10)
    
    c1 = table.cell(0, 1)
    p1 = c1.paragraphs[0]
    p1.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r1 = p1.add_run(f"({eq_num})")
    r1.font.name = "Times New Roman"
    r1.font.size = Pt(10)
    
    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(0)
    p_spacer.paragraph_format.space_after = Pt(4)

# Title
p_title = doc.add_paragraph()
p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
r_t = p_title.add_run("IEEE COMPANION: ALL MATHEMATICAL FORMULAS & FORMAL TABLES")
r_t.font.name = "Times New Roman"
r_t.font.size = Pt(16)
r_t.font.bold = True
r_t.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

p_sub = doc.add_paragraph()
p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_sub.paragraph_format.space_after = Pt(16)
r_s = p_sub.add_run("State and Memory Based Sub-Quadratic Sequence Modeling with Entropy-Gated Associative Neural Networks\nReady for Direct Copy-Paste into IEEE MS Word Templates")
r_s.font.name = "Times New Roman"
r_s.font.size = Pt(10)
r_s.font.italic = True

# ==============================================================================
# SECTION A: ALL MATHEMATICAL EQUATIONS (1) - (18)
# ==============================================================================
add_heading("PART I: COMPLETE EQUATION DIRECTORY (1)–(18)")

add_heading("1. Pre-RMSNorm Normalization", level=2)
add_equation("RMSNorm(x) = ( x / sqrt( (1/D) * sum_{i=1}^D x_i^2 + eps ) ) * gamma", 1)

add_heading("2. Continuous-Time State Space System", level=2)
add_equation("h'(t) = A h(t) + B u(t),    y(t) = C h(t) + D u(t)", 2)

add_heading("3. Zero-Order Hold (ZOH) Discrete Recurrence", level=2)
add_equation("h_t = exp(Delta_t * A) * h_{t-1} + (Delta_t * B_t) * x_t", 3)
add_equation("y_t = C_t * h_t + D * x_t", 4)

add_heading("4. S4D Real Diagonal Eigenvalue Spectrum", level=2)
add_equation("A_n = -n,    for n in {1, 2, ..., N}, with N = 16", 5)

add_heading("5. Selective Input-Dependent Projections", level=2)
add_equation("Delta_t = softplus(W_Delta * x_t + b_Delta),    B_t = W_B * x_t,    C_t = W_C * x_t", 6)

add_heading("6. Decayed Associative Neural Memory Outer-Product Update", level=2)
add_equation("M_t = alpha_t * M_{t-1} + (v_t (x) k_t^gated)", 7)

add_heading("7. Causal Information-Density Key Write Gate", level=2)
add_equation("k_t^gated = k_t * sigma(W_store * e_t + b_store)", 8)

add_heading("8. Bounded Associative Forget Factor", level=2)
add_equation("alpha_t = sigma(W_decay * [x_t; e_t] + b_decay) * 0.149 + 0.85 in (0.85, 0.999)", 9)

add_heading("9. Associative Memory Readout & Zero-Initialized Projection", level=2)
add_equation("r_t = M_t * q_t,    y_t^titans = W_out * r_t    (with W_out|_init = 0)", 10)

add_heading("10. Strictly Causal 1D Convolutional Entropy Predictor", level=2)
add_equation("x_pad = Pad_left(x_embed, pad=2)", 11)
add_equation("h_ent = GELU(Conv1D(x_pad, W_1)) in R^{B x S x 128}", 12)
add_equation("e_t = sigma(Conv1D(Pad_left(h_ent, 2), W_2)) in (0, 1)", 13)

add_heading("11. Unified Dynamic Compute Block Transformation", level=2)
add_equation("x^{(l+1)} = x^{(l)} + sigma(W_g * e_t) * SSM(Norm(x^{(l)})) + Titans(Norm(x^{(l)}), e_t) + sigma(W_f * e_t) * FFN(Norm(x^{(l)}))", 14)

add_heading("12. GPU Reverse-Time Adjoint Equations for Triton Compilation", level=2)
add_equation("lambda_t = dh_t + lambda_{t+1} * dA_{t+1}", 15)
add_equation("d(Delta_t) = sum_n lambda_{t, n} * [ A_n * h_{t-1, n} * exp(Delta_t * A_n) + B_{t, n} * x_t ]", 16)
add_equation("dB_t = Delta_t * (lambda_t * x_t),    dx_t = Delta_t * (lambda_t * B_t) + (dL/dy_t) * D", 17)

add_heading("13. Extended Context 32K KV-Cache Footprint Algebra", level=2)
add_equation("KV_Size = 2 (K, V) * 4 layers * 448 dim * 32,768 tokens * 2 bytes (fp16) = 234,881,024 bytes approx 234.9 MB", 18)

# ==============================================================================
# SECTION B: FORMAL IEEE TABLES I - V
# ==============================================================================
add_heading("PART II: FORMAL IEEE TABLES I–V")

# TABLE I
add_heading("TABLE I: COMPLEXITY, PATH LENGTH, AND MEMORY SCALING ACROSS PARADIGMS", level=2)
t1 = doc.add_table(rows=6, cols=5)
t1.alignment = WD_TABLE_ALIGNMENT.CENTER
t1_headers = ["Layer Type", "Complexity per Layer", "Sequential Operations", "Maximum Path Length", "Inference Memory Class"]
t1_data = [
    ["Self-Attention (Transformer) [1]", "O(S^2 * d)", "O(1)", "O(1)", "O(S) Linear KV-Cache"],
    ["Recurrent Neural Network (LSTM) [11]", "O(S * d^2)", "O(S)", "O(S)", "O(1) Constant State"],
    ["Convolutional (ByteNet / ConvS2S) [24]", "O(k * S * d^2)", "O(1)", "O(log_k(S))", "O(k * L) Buffer"],
    ["Linear Attention [3]", "O(S * d * d_k)", "O(1)", "O(1)", "O(1) Constant State"],
    ["IVERI-HYPERION (ENSN, Ours)", "O(S * d * N + S * d * d_mem)", "O(1)", "O(1)", "O(1) Constant State"],
]
for j, h in enumerate(t1_headers):
    c = t1.cell(0, j)
    set_cell_shading(c, "E8F0FE")
    set_cell_margins(c, 100, 100, 100, 100)
    p = c.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(h)
    r.font.name = "Times New Roman"
    r.font.size = Pt(8.5)
    r.font.bold = True

for i, row in enumerate(t1_data):
    for j, val in enumerate(row):
        c = t1.cell(i+1, j)
        set_cell_margins(c, 70, 70, 80, 80)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(val)
        r.font.name = "Times New Roman"
        r.font.size = Pt(8)
        if i == 4:
            r.font.bold = True

# TABLE II
add_heading("TABLE II: MULTI-SEED 4-WAY PARADIGM BENCHMARK AT 10M PARAMETER SCALE", level=2)
t2 = doc.add_table(rows=5, cols=7)
t2.alignment = WD_TABLE_ALIGNMENT.CENTER
t2_headers = ["Architecture", "Paradigm", "Parameters", "Delta (%)", "Val Loss (Mean +/- Std)", "Byte-PPL (Mean +/- Std)", "Speed (tok/s)"]
t2_data = [
    ["Dense Transformer [1]", "Softmax Attention", "9,984,556", "-0.0089%", "3.1286 +/- 0.0299", "22.85 +/- 0.69", "32,973.9"],
    ["Linear Attention [3]", "Kernel Associative", "9,986,816", "+0.0137%", "3.1896 +/- 0.0013", "24.28 +/- 0.03", "2,653.3"],
    ["Pure Mamba SSM [2]", "State Space Model", "9,985,664", "+0.0022%", "1.2462 +/- 0.0017", "3.47 +/- 0.01", "2,419.2"],
    ["IVERI-HYPERION (Ours)", "SSM + Titans Hybrid", "9,985,449", "0.0000%", "1.2102 +/- 0.0168", "3.35 +/- 0.06", "2,400.8"],
]
for j, h in enumerate(t2_headers):
    c = t2.cell(0, j)
    set_cell_shading(c, "E8F0FE")
    set_cell_margins(c, 100, 100, 80, 80)
    p = c.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(h)
    r.font.name = "Times New Roman"
    r.font.size = Pt(8.5)
    r.font.bold = True

for i, row in enumerate(t2_data):
    for j, val in enumerate(row):
        c = t2.cell(i+1, j)
        set_cell_margins(c, 70, 70, 80, 80)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 1 else WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(val)
        r.font.name = "Times New Roman"
        r.font.size = Pt(8)
        if i == 3:
            r.font.bold = True

# TABLE III
add_heading("TABLE III: MULTI-SEED ARCHITECTURAL ABLATIONS", level=2)
t3 = doc.add_table(rows=6, cols=5)
t3.alignment = WD_TABLE_ALIGNMENT.CENTER
t3_headers = ["Variation", "Configuration", "Val Loss (Mean)", "Byte-PPL (Mean)", "Rel. Delta vs Baseline"]
t3_data = [
    ["(A) Baseline", "Full IVERI-HYPERION (mem=64, Triton)", "1.2102", "3.35", "Baseline (0.0%)"],
    ["(B) No Titans", "Titans bypassed (W_out <- 0)", "1.2462", "3.47", "+3.0% Loss Penalty"],
    ["(C) Reduced Mem", "d_mem = 32 (reduced capacity)", "1.2284", "3.42", "+1.5% Loss Penalty"],
    ["(D) No Gating", "Uniform compute (gate = 1.0)", "1.2215", "3.39", "+0.9% Loss Penalty"],
    ["(E) No Triton", "PyTorch interpreted recurrence loop", "1.2102", "3.35", "4.3x Speed Penalty"],
]
for j, h in enumerate(t3_headers):
    c = t3.cell(0, j)
    set_cell_shading(c, "E8F0FE")
    set_cell_margins(c, 100, 100, 80, 80)
    p = c.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(h)
    r.font.name = "Times New Roman"
    r.font.size = Pt(8.5)
    r.font.bold = True

for i, row in enumerate(t3_data):
    for j, val in enumerate(row):
        c = t3.cell(i+1, j)
        set_cell_margins(c, 70, 70, 80, 80)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 1 else WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(val)
        r.font.name = "Times New Roman"
        r.font.size = Pt(8)
        if i == 0:
            r.font.bold = True

# TABLE IV
add_heading("TABLE IV: LOCKSTEP CONVERGENCE TRAJECTORY OVER 1,000 STEPS", level=2)
t4 = doc.add_table(rows=6, cols=6)
t4.alignment = WD_TABLE_ALIGNMENT.CENTER
t4_headers = ["Training Step", "IVERI Val Loss", "IVERI Byte-PPL", "Transformer Val Loss", "Transformer Byte-PPL", "Perplexity Advantage"]
t4_data = [
    ["Step 100", "1.4437", "4.24", "3.2335", "25.37", "5.98x"],
    ["Step 300", "1.2239", "3.40", "2.8123", "16.65", "4.90x"],
    ["Step 500", "1.1051", "3.02", "2.6314", "13.89", "4.60x"],
    ["Step 700", "1.0260", "2.79", "2.5352", "12.62", "4.52x"],
    ["Step 1,000", "0.9625", "2.62", "2.4493", "11.58", "4.42x"],
]
for j, h in enumerate(t4_headers):
    c = t4.cell(0, j)
    set_cell_shading(c, "E8F0FE")
    set_cell_margins(c, 100, 100, 80, 80)
    p = c.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(h)
    r.font.name = "Times New Roman"
    r.font.size = Pt(8.5)
    r.font.bold = True

for i, row in enumerate(t4_data):
    for j, val in enumerate(row):
        c = t4.cell(i+1, j)
        set_cell_margins(c, 70, 70, 80, 80)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(val)
        r.font.name = "Times New Roman"
        r.font.size = Pt(8)
        if j == 5:
            r.font.bold = True

# TABLE V
add_heading("TABLE V: UNCURATED QUALITATIVE TEXT GENERATION COMPLETIONS (1,000 STEPS)", level=2)
t5 = doc.add_table(rows=3, cols=3)
t5.alignment = WD_TABLE_ALIGNMENT.CENTER
t5_headers = ["Prompt", "IVERI-HYPERION Completion (1,000 st)", "Dense Transformer Completion (1,000 st)"]
t5_data = [
    ["\"Once upon a time, there was a little girl named\"", "\"...Bobe. He started to paint the swim and yellow home. They play! <|endoftext|>\"", "\"...ont|> \\n\\n Orape wile sis. Thid ad thenkiy ay Mandas kolen re pldo mo theier\""],
    ["\"The big dog ran to the\"", "\"...sound. The bird like cars, a big cloud when he was playing with that, Jake and Tom away. So he kn\"", "\"...anaued imet bt shis. Thudise o topkenng <are! venthe oum in moowe. thin dd heery.\""],
]
for j, h in enumerate(t5_headers):
    c = t5.cell(0, j)
    set_cell_shading(c, "E8F0FE")
    set_cell_margins(c, 100, 100, 80, 80)
    p = c.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(h)
    r.font.name = "Times New Roman"
    r.font.size = Pt(8.5)
    r.font.bold = True

for i, row in enumerate(t5_data):
    for j, val in enumerate(row):
        c = t5.cell(i+1, j)
        set_cell_margins(c, 70, 70, 80, 80)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(val)
        r.font.name = "Consolas" if j > 0 else "Times New Roman"
        r.font.size = Pt(8)
        if j == 1:
            r.font.bold = True

doc.save(str(OUTPUT_DOCX))
print(f"[SUCCESS] Clean IEEE Formulas & Tables DOCX saved at: {OUTPUT_DOCX}")
