import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

try:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
except ImportError:
    install("fpdf2")
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos


class PDF(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(160, 160, 160)
        self.cell(0, 10, "AgroCarbon AI  |  Capstone Project  |  Confidential",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.set_text_color(0, 0, 0)


pdf = PDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.set_margins(0, 0, 0)

# ─────────────────────────────────────────────────────────────
# DARK HEADER BAND
# ─────────────────────────────────────────────────────────────
pdf.set_fill_color(18, 32, 58)
pdf.rect(0, 0, 210, 42, 'F')

# Green accent stripe
pdf.set_fill_color(34, 197, 94)
pdf.rect(0, 42, 210, 3, 'F')

pdf.set_y(8)
pdf.set_text_color(255, 255, 255)
pdf.set_font("Helvetica", "B", 20)
pdf.cell(0, 11, "AgroCarbon AI -- Seed Funding Application",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
pdf.set_font("Helvetica", "", 11)
pdf.set_text_color(180, 220, 180)
pdf.cell(0, 8, "Tree Carbon Assessment System  |  Capstone Project  2026",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
pdf.ln(10)

# ─────────────────────────────────────────────────────────────
# META INFO
# ─────────────────────────────────────────────────────────────
pdf.set_text_color(0, 0, 0)
pdf.set_left_margin(18)
pdf.set_right_margin(18)
pdf.set_y(51)

meta = [
    ("Date",  "July 2026"),
    ("To",    "The Seed Funding Committee / Project Guide"),
    ("From",  "[Your Name / Team Name]  |  [Roll Number]"),
]
for label, value in meta:
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(22, 7, label + ":", border=0)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, value, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(3)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def section_title(pdf, text):
    pdf.set_fill_color(18, 32, 58)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 9, "  " + text, fill=True,
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)


badge_colors = [
    (59, 130, 246),    # blue
    (168, 85, 247),    # purple
    (249, 115, 22),    # orange
    (20, 184, 166),    # teal
]


def expense_card(pdf, idx, letter, title, amount, details):
    y_start = pdf.get_y()
    card_h = 36

    # Card background + border
    pdf.set_fill_color(248, 249, 252)
    pdf.set_draw_color(220, 225, 235)
    pdf.rect(18, y_start, 174, card_h, 'DF')

    # Left badge strip
    r, g, b = badge_colors[idx]
    pdf.set_fill_color(r, g, b)
    pdf.rect(18, y_start, 7, card_h, 'F')

    # Badge letter
    pdf.set_xy(18, y_start + card_h / 2 - 4)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(7, 8, letter, align="C")

    # Title
    pdf.set_xy(28, y_start + 5)
    pdf.set_text_color(18, 32, 58)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(115, 7, title)

    # Amount (badge-colored)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(r, g, b)
    pdf.cell(0, 7, amount, align="R",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Detail text
    pdf.set_xy(28, y_start + 16)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(80, 85, 105)
    pdf.multi_cell(160, 5, details)

    pdf.set_y(y_start + card_h + 4)
    pdf.set_text_color(0, 0, 0)


# ─────────────────────────────────────────────────────────────
# EXPENSE SECTION
# ─────────────────────────────────────────────────────────────
section_title(pdf, "  Breakdown of Expenses")

expenses = [
    (
        "A", "Cloud Computing & AI Model Training", "INR 6,000",
        "Google Colab Pro / Pro+ (multiple months) for A100/V100 GPUs to train SegFormer-B5 and YOLOv11"
        " models on the NEON dataset through 80-epoch cycles. Prevents OOM errors and session timeouts."
    ),
    (
        "B", "Cloud Hosting & Database Infrastructure", "INR 5,000",
        "AWS S3 for storing image uploads, AI-generated masks and metadata. AWS EC2 for hosting the"
        " FastAPI backend pipeline and serving ML models 24/7 without cold-start delays."
    ),
    (
        "C", "Web Application Deployment & Database Management", "INR 4,000",
        "Custom domain name + SSL certificates for the web dashboard. Managed PostgreSQL (AWS RDS) for"
        " geospatial data, user metadata, and final AI carbon assessment reports."
    ),
    (
        "D", "AI-Assisted Development Tools Subscription", "INR 3,000",
        "GitHub Copilot for code completion across Python, Flutter & YAML. ChatGPT Plus / Claude Pro"
        " for architecture design, code review and complex ML debugging during development."
    ),
]

for i, (letter, title, amount, detail) in enumerate(expenses):
    expense_card(pdf, i, letter, title, amount, detail)

pdf.ln(2)

# ─────────────────────────────────────────────────────────────
# TOTAL BANNER
# ─────────────────────────────────────────────────────────────
y_total = pdf.get_y()
pdf.set_fill_color(18, 32, 58)
pdf.rect(18, y_total, 174, 14, 'F')
pdf.set_fill_color(34, 197, 94)
pdf.rect(18, y_total, 5, 14, 'F')

pdf.set_xy(28, y_total + 3)
pdf.set_font("Helvetica", "B", 12)
pdf.set_text_color(255, 255, 255)
pdf.cell(120, 8, "TOTAL REQUESTED SEED FUNDING")
pdf.set_font("Helvetica", "B", 13)
pdf.set_text_color(34, 220, 100)
pdf.cell(0, 8, "INR 18,000", align="R",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)

pdf.ln(6)

# ─────────────────────────────────────────────────────────────
# IMPACT NOTE
# ─────────────────────────────────────────────────────────────
section_title(pdf, "  Impact & Justification")
pdf.set_font("Helvetica", "", 10)
pdf.set_text_color(50, 50, 65)
pdf.set_x(18)
pdf.multi_cell(174, 5.5,
    "The requested seed funding of INR 18,000 directly offsets the financial burden of building and deploying "
    "AgroCarbon AI. Without cloud GPUs, the SegFormer and YOLOv11 models could not achieve viable accuracy. "
    "Without robust cloud hosting, the end-to-end pipeline - from mobile image capture to final carbon reports "
    "- would not function in a real-world setting. AI coding tools accelerated our development cycle, while "
    "web deployment and database management ensure the platform is secure, scalable, and accessible."
)
pdf.ln(5)

# ─────────────────────────────────────────────────────────────
# SIGNATURE BLOCK
# ─────────────────────────────────────────────────────────────
y_sig = pdf.get_y()
pdf.set_fill_color(240, 244, 255)
pdf.set_draw_color(200, 210, 235)
pdf.rect(18, y_sig, 174, 26, 'DF')
pdf.set_fill_color(59, 130, 246)
pdf.rect(18, y_sig, 5, 26, 'F')

pdf.set_xy(26, y_sig + 5)
pdf.set_font("Helvetica", "B", 11)
pdf.set_text_color(18, 32, 58)
pdf.cell(0, 7, "Sincerely,", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_x(26)
pdf.set_font("Helvetica", "", 10)
pdf.set_text_color(60, 65, 85)
pdf.cell(0, 6, "[Your Name / Team Leader]", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_x(26)
pdf.cell(0, 6, "[Roll Number  |  Department  |  College Name]",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)

# ─────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────
output_path = "d:/Capston Zip Files/tree_carbon_assessment_system/Seed_Funding_Expenses_Summary.pdf"
pdf.output(output_path)
print(f"PDF saved: {output_path}")
