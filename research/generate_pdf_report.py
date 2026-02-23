
from fpdf import FPDF
import math

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, 'Technical Specification: Unified v2 Pruning', border=False, new_x="LMARGIN", new_y="NEXT", align='C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT", fill=True)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Helvetica', '', 11)
        self.multi_cell(0, 6, body, align='L')
        self.ln()

    def formula_box(self, formula):
        self.set_font('Courier', '', 10)
        self.set_fill_color(245, 245, 245)
        self.multi_cell(self.epw, 6, formula, fill=True, border=1, align='L')
        self.ln(2)

def create_pdf(filename):
    pdf = PDF()
    pdf.add_page()
    
    # Introduction
    pdf.chapter_body(
        "This document provides the mathematical foundations and implementation details for the Unified v2 Pruning method. "
        "It addresses the limitations of standard pruning metrics (Magnitude, WANDA, SparseGPT) on noisy time-series data."
    )

    # Section 1
    pdf.chapter_title('1. Error-Weighted Gram Matrix Accumulation')
    pdf.chapter_body(
        "Standard pruning often calculates the Hessian (X^T X) using unweighted inputs. "
        "Unified v2 modifies this by weighting input windows based on the model's error."
    )
    
    pdf.set_font('Helvetica', 'B', 11); pdf.cell(0, 8, "A. Weight Calculation", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', '', 11)
    
    pdf.multi_cell(0, 6, 
        "For each calibration window i, we compute the Mean Squared Error (MSE) of the dense model's prediction against the target.\n"
        "Let e_i = MSE(pred_i, target_i)."
    )
    
    pdf.formula_box(
        "Relative Error Ratio:  r_i = e_i / (mean(e) + epsilon)\n"
        "Sample Weight:         w_i = (r_i)^P\n\n"
        "where P (error_power) = 1.0 by default. High P focuses pruning on 'hard' examples."
    )

    pdf.set_font('Helvetica', 'B', 11); pdf.cell(0, 8, "B. Gram Matrix Update", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', '', 11)
    
    pdf.multi_cell(0, 6, 
        "The Gram matrix G (approx Hessian) for each group of 4 inputs is computed as a weighted sum of outer products:"
    )
    
    pdf.formula_box(
        "G = Sum[ w_i * (X_i * X_i^T) ]\n\n"
        "- X_i: Input activation vector for the group (4x1)\n"
        "- w_i: Computed sample weight\n"
        "- G:   Resulting 4x4 matrix"
    )

    # Section 2
    pdf.chapter_title('2. Adaptive Unified Scoring')
    pdf.multi_cell(0, 6, 
        "We blend two metrics: Gram-based scoring (standard) and Spectral Quality scoring (noise-robust)."
    )

    pdf.set_font('Helvetica', 'B', 11); pdf.cell(0, 8, "A. Noise Fraction Estimation", new_x="LMARGIN", new_y="NEXT")
    pdf.formula_box(
        "Activation Energy via FFT:\n"
        "  E_signal = E_trend + E_season\n"
        "  E_noise  = Energy in high-frequency band (>70% Nyquist)\n\n"
        "Noise Fraction (N_frac) = E_noise / (E_signal + E_noise)"
    )

    pdf.set_font('Helvetica', 'B', 11); pdf.cell(0, 8, "B. Blending Factor (Alpha) and Unified Score", new_x="LMARGIN", new_y="NEXT")
    pdf.formula_box(
        "Alpha = clamp( (N_frac - 0.15) / 0.15, 0, 1 )\n\n"
        "Unified_Score = Alpha * Z(Score_spectral) + (1-Alpha) * Z(Score_ratio)\n\n"
        "Where Z() is Z-normalization across the layer."
    )

    # Section 3
    pdf.chapter_title('3. Auto-Ridge Regularization')
    pdf.multi_cell(0, 6, 
        "To prevent the refit step (Optimal Brain Surgeon) from overfitting to noise, we adapt the regularization strength (lambda)."
    )
    
    pdf.formula_box(
        "Effective Ridge (lambda_eff):\n"
        "  lambda_eff = lambda_base * 10^(3 * t)\n\n"
        "  where t = clamp( (max(N_frac_global) - 0.15) / 0.15, 0, 1 )\n\n"
        "  Scaling: 1e-5 -> 1e-2 as noise increases."
    )

    # Section 4
    pdf.chapter_title('4. Performance Summary')
    pdf.multi_cell(0, 6, 
        "Unified v2 achieves superior results across all ETT datasets (2:4 Sparsity)."
    )
    
    data = [
        ["Dataset", "Unified v2 (MSE)", "SparseGPT", "Magnitude", "WANDA"],
        ["ETTm1", "-0.14 (Best)", "+0.03", "+0.02", "+0.49"],
        ["ETTm2", "-0.18 (Best)", "+3.17", "+4.65", "+6.39"],
        ["ETTh1", "-0.31 (Best)", "-0.26", "-0.06", "+0.44"],
        ["ETTh2", "+0.61 (Best)", "+2.47", "+5.49", "+5.16"],
    ]
    
    # Table
    pdf.ln(2)
    pdf.set_font('Helvetica', 'B', 10)
    # Header
    col_width = pdf.epw / 5
    for row in data:
        for datum in row:
            pdf.cell(col_width, 8, str(datum), border=1, align='C')
        pdf.ln()
        pdf.set_font('Helvetica', '', 10)

    pdf.output(filename)
    print(f"PDF generated: {filename}")

if __name__ == "__main__":
    create_pdf("unified_v2_technical_report.pdf")
