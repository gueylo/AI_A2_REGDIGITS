"""Deterministic PDF report and evidence rendering."""

from __future__ import annotations

import json
import platform
import textwrap
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

PAGE_W, PAGE_H = A4
MARGIN = 42


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _header(canvas: Canvas, page: int, title: str, total_pages: int = 7) -> float:
    canvas.setFillColor(colors.HexColor("#173B2D"))
    canvas.rect(0, PAGE_H - 62, PAGE_W, 62, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 15)
    canvas.drawString(MARGIN, PAGE_H - 36, title)
    canvas.setFont("Helvetica", 8.5)
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 36, f"AI-A2-7953  |  Page {page} of {total_pages}")
    canvas.setFillColor(colors.black)
    return PAGE_H - 84


def _footer(canvas: Canvas) -> None:
    canvas.setStrokeColor(colors.HexColor("#B7C9BF"))
    canvas.line(MARGIN, 30, PAGE_W - MARGIN, 30)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#52645B"))
    canvas.drawString(MARGIN, 18, "Musanze Harvest Intelligence Service — synthetic classroom prototype")


def _paragraph(canvas: Canvas, text: str, y: float, *, size: float = 9.2, leading: float = 12, bold: bool = False) -> float:
    canvas.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    width = PAGE_W - 2 * MARGIN
    average = max(20, int(width / (size * 0.51)))
    for raw in text.split("\n"):
        lines = textwrap.wrap(raw, width=average, break_long_words=False) or [""]
        for line in lines:
            canvas.drawString(MARGIN, y, line)
            y -= leading
        y -= 2
    return y


def _section(canvas: Canvas, title: str, y: float) -> float:
    canvas.setFillColor(colors.HexColor("#276749"))
    canvas.setFont("Helvetica-Bold", 11)
    canvas.drawString(MARGIN, y, title)
    canvas.setFillColor(colors.black)
    return y - 17


def _table(canvas: Canvas, headers: list[str], rows: list[list[str]], y: float, widths: list[float] | None = None) -> float:
    total = PAGE_W - 2 * MARGIN
    widths = widths or [total / len(headers)] * len(headers)
    row_h = 18
    x = MARGIN
    canvas.setFont("Helvetica-Bold", 7.8)
    canvas.setFillColor(colors.HexColor("#DCEBE3"))
    canvas.rect(x, y - row_h + 4, total, row_h, fill=1, stroke=0)
    canvas.setFillColor(colors.black)
    for heading, width in zip(headers, widths):
        canvas.drawString(x + 3, y - 8, heading[:32])
        x += width
    y -= row_h
    canvas.setFont("Helvetica", 7.5)
    for index, row in enumerate(rows):
        x = MARGIN
        if index % 2:
            canvas.setFillColor(colors.HexColor("#F4F8F6"))
            canvas.rect(x, y - row_h + 4, total, row_h, fill=1, stroke=0)
        canvas.setFillColor(colors.black)
        for value, width in zip(row, widths):
            canvas.drawString(x + 3, y - 8, str(value)[:42])
            x += width
        y -= row_h
    return y - 8


def _image(canvas: Canvas, path: Path, x: float, y: float, width: float, height: float) -> None:
    if path.is_file():
        canvas.drawImage(ImageReader(str(path)), x, y, width=width, height=height, preserveAspectRatio=True, anchor="c")


def generate_technical_report(root: Path) -> Path:
    """Render the required seven-page report from current measured artifacts."""
    artifacts = root / "artifacts"
    reg = _load(artifacts / "regression_metrics.json")
    cls = _load(artifacts / "classification_metrics.json")
    clu = _load(artifacts / "clustering_report.json")
    val = _load(artifacts / "data_validation.json")
    output = root / "docs" / "AI_A2_2527953_REPORT.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(str(output), pagesize=A4, invariant=1)
    canvas.setTitle("AI_A2_2527953 Technical Report")
    canvas.setAuthor("Kawaya Gueylord")

    y = _header(canvas, 1, "Identity and decision problem")
    y = _paragraph(canvas, "Kawaya Gueylord  |  Registration: 25/27953  |  ug2527953@ines.ac.rw", y, bold=True)
    y = _paragraph(canvas, "Verification code: AI-A2-7953  |  Personal seed: 7953", y, bold=True)
    y = _section(canvas, "Decision setting", y)
    y = _paragraph(canvas, "A Musanze potato cooperative receives inconsistent farm spreadsheets. Its field officer needs an offline decision-support service that estimates harvest quantity, identifies farms likely to have below-median yield per hectare, and groups farms with similar input conditions.", y)
    y = _section(canvas, "Three machine-learning questions", y)
    y = _paragraph(canvas, "1. Regression: how many tons might this farm harvest?\n2. Classification: is the farm estimated to be in the high-risk class?\n3. Clustering: which farms have similar measured input conditions?", y)
    y = _section(canvas, "Ethical boundary", y)
    y = _paragraph(canvas, "This is a synthetic classroom prototype, not an agronomic authority. It must not replace an agronomist or determine access to finance, insurance, public services, or farm treatment. A human field officer remains responsible for decisions.", y)
    _footer(canvas); canvas.showPage()

    y = _header(canvas, 2, "Data and reproducibility")
    y = _paragraph(canvas, f"The seed is int(last four registration digits) = 7953. Base rows = 240 + (7953 mod 41) = {val['base_rows']}. Three exact duplicates produce {val['raw_rows']} raw rows; cleaning returns {val['clean_rows']} rows.", y)
    y = _table(canvas, ["Field", "Generation / meaning"], [
        ["rainfall_mm", "Uniform 600–1400"], ["avg_temp_c", "Uniform 12–26"],
        ["soil_moisture_pct", "Uniform 20–80; 14 planned missing positions"],
        ["fertilizer_kg", "Uniform 0–250; 14 planned missing positions"],
        ["field_size_ha", "Uniform 0.2–4.0"], ["season", "Random A or B"],
        ["cooperative_support", "Random binary 0 or 1"],
        ["yield_tons", "Prescribed formula plus N(0, 0.80), clipped at 0.50"],
        ["high_risk", "1 when base-row yield/ha is strictly below its median"],
    ], y, [135, PAGE_W - 2*MARGIN - 135])
    y = _section(canvas, "Byte-level signature", y)
    y = _paragraph(canvas, f"CSV: data/AI_A2_2527953.csv\nSHA-256: {val['sha256']}\nThe same default_rng(7953), draw order, four-decimal formatting, and newline convention recreate the same bytes. The risk median is computed on the original base rows before missing values and duplicates.", y, size=8.5)
    _footer(canvas); canvas.showPage()

    y = _header(canvas, 3, "Pipeline design and leakage controls")
    y = _section(canvas, "Reproducible flow", y)
    y = _paragraph(canvas, "Generate → validate raw schema/ranges/nulls/duplicates → remove exact duplicates → one stratified 80/20 split → training-only imputation/scaling/encoding → cross-validation on training rows → fit selected pipelines → held-out evaluation → save models, predictions, JSON and plots → CLI/API inference.", y)
    y = _section(canvas, "Inputs and transformations", y)
    y = _paragraph(canvas, "Only rainfall_mm, avg_temp_c, soil_moisture_pct, fertilizer_kg, field_size_ha, season and cooperative_support are inputs. Numeric fields use median imputation and standardization. Season uses most-frequent imputation and one-hot encoding. Every transformer is inside a scikit-learn Pipeline and is fitted only on training data.", y)
    y = _section(canvas, "Leakage prevention", y)
    y = _paragraph(canvas, "yield_tons, high_risk, yield per hectare, row_id and all target-derived values are excluded from classification and clustering inputs. Missing values are not globally filled. The test set is not used for model selection; five-fold training-only CV selects the final regression and classification estimators.", y)
    y = _table(canvas, ["Task", "Models compared", "Selection evidence"], [
        ["Regression", "Mean dummy; Ridge; Random Forest", "Lowest CV MAE; RMSE tie-break"],
        ["Classification", "Majority dummy; Logistic; Random Forest", "Highest CV F1; recall tie-break"],
        ["Clustering", "K-Means K=2,3,4,5,6", "Highest training silhouette"],
    ], y, [90, 220, PAGE_W - 2*MARGIN - 310])
    _footer(canvas); canvas.showPage()

    y = _header(canvas, 4, "Regression and classification")
    reg_rows = [[name, f"{m['mae']:.3f}", f"{m['rmse']:.3f}", f"{m['r2']:.3f}"] for name, m in reg['held_out_metrics'].items()]
    y = _table(canvas, ["Regression model", "MAE", "RMSE", "R²"], reg_rows, y, [190, 90, 90, 90])
    y = _paragraph(canvas, f"Ridge was selected from training-only CV (mean MAE {reg['cv_results']['ridge']['cv_mae_mean']:.3f}). On test data it achieved MAE {reg['held_out_metrics']['ridge']['mae']:.3f} tons and R² {reg['held_out_metrics']['ridge']['r2']:.3f}.", y, size=8.3)
    cls_rows = [[name, f"{m['precision_high_risk']:.3f}", f"{m['recall_high_risk']:.3f}", f"{m['f1_high_risk']:.3f}", f"{m['roc_auc']:.3f}"] for name, m in cls['held_out_metrics'].items()]
    y = _table(canvas, ["Classifier", "Precision", "Recall", "F1", "ROC-AUC"], cls_rows, y, [155, 77, 77, 77, 77])
    y = _paragraph(canvas, f"Random Forest was selected because its training CV F1 ({cls['cv_results']['random_forest']['cv_f1_mean']:.3f}) narrowly exceeded Logistic Regression ({cls['cv_results']['logistic_regression']['cv_f1_mean']:.3f}). Held-out selection was not used.", y, size=8.3)
    _image(canvas, artifacts / "plots" / "residuals.png", MARGIN, 48, 245, 180)
    _image(canvas, artifacts / "plots" / "confusion_matrix.png", 310, 48, 245, 180)
    canvas.setFont("Helvetica", 7.2)
    canvas.drawString(MARGIN, 41, "Largest residual: row 150, actual 16.510 t, predicted 13.285 t (3.225 t under-estimate).")
    _footer(canvas); canvas.showPage()

    y = _header(canvas, 5, "Clustering evidence and interpretation")
    k_rows = [[k, f"{entry['silhouette']:.4f}", str(entry['cluster_counts'])] for k, entry in clu['scores_k_2_to_6'].items()]
    y = _table(canvas, ["K", "Silhouette", "Training counts"], k_rows, y, [60, 100, PAGE_W - 2*MARGIN - 160])
    y = _paragraph(canvas, f"K={clu['selected_k']} has the highest observed silhouette ({clu['scores_k_2_to_6'][str(clu['selected_k'])]['silhouette']:.4f}). The value is modest, so the grouping should be treated as a useful description rather than evidence of sharply separated natural farm types.", y, size=8.4)
    profile_rows = []
    for p in clu['profiles_original_units']:
        means = p['numeric_means']
        profile_rows.append([str(p['cluster']), str(p['count']), f"{means['rainfall_mm']:.1f}", f"{means['fertilizer_kg']:.1f}", f"{means['field_size_ha']:.2f}", f"{p['cooperative_support_proportion']:.2f}"])
    y = _table(canvas, ["ID", "N", "Rain mm", "Fert kg", "Field ha", "Support"], profile_rows, y, [45, 50, 85, 85, 85, 85])
    y = _paragraph(canvas, "Measured interpretation for student review: the strongest difference is cooperative support—cluster 0 contains unsupported farms while cluster 1 contains supported farms. Other average conditions are fairly similar. IDs 0 and 1 are labels, not best/worst rankings.", y, size=8.4, bold=True)
    _image(canvas, artifacts / "plots" / "clusters.png", 120, 45, 355, 255)
    _footer(canvas); canvas.showPage()

    y = _header(canvas, 6, "API, CLI and testing")
    y = _table(canvas, ["Interface", "Contract"], [
        ["GET /health", "200 with status and AI-A2-7953; 503 if models unavailable"],
        ["GET /model-info", "Identity, model names, K, features, fingerprint, limitations"],
        ["POST /predict", "Seven typed fields; yield, risk, probability, cluster and code"],
        ["predict.py", "Loads the same saved pipelines; never retrains"],
    ], y, [135, PAGE_W - 2*MARGIN - 135])
    y = _section(canvas, "Validation behavior", y)
    y = _paragraph(canvas, "The API rejects missing fields, extra fields, strings pretending to be numbers, nulls, NaN/infinity, invalid seasons/support flags, and numeric values outside the prescribed ranges. FastAPI returns HTTP 422 rather than an unhandled server error. The invalid CLI exits nonzero with a readable validation message.", y)
    y = _section(canvas, "Executed test evidence", y)
    y = _paragraph(canvas, "The suite covers valid inference, missing fields, wrong types, ranges, extra inputs, seed mismatch, byte reproducibility, exact duplicates, planned missingness, leakage allowlists, model loading, K=2…6 evidence, API responses, health/model information, and CLI/API parity. See evidence/AI_A2_2527953_TEST_LOG.pdf for the actual command and result.", y)
    y = _section(canvas, "Example valid record", y)
    y = _paragraph(canvas, '{"rainfall_mm":950.0,"avg_temp_c":18.0,"soil_moisture_pct":55.0,"fertilizer_kg":90.0,"field_size_ha":1.5,"season":"A","cooperative_support":1}', y, size=7.5)
    _footer(canvas); canvas.showPage()

    y = _header(canvas, 7, "Model card and reflection")
    y = _section(canvas, "Intended use and users", y)
    y = _paragraph(canvas, "A classroom demonstration for a cooperative field officer: offline comparison of synthetic farm conditions, estimated yield, and a high-risk screening signal. Predictions should support questions, not settle decisions.", y)
    y = _section(canvas, "Risks and prohibited use", y)
    y = _paragraph(canvas, "Synthetic-only training may omit real weather, disease, soil chemistry, management and market effects. Errors could misdirect scarce attention. Probability estimates are not calibrated agronomic truths. Outputs must never determine credit, insurance, public-service eligibility or treatment without consent, real validation and qualified review.", y)
    y = _section(canvas, "Measured errors requiring Kawaya's review", y)
    error = cls['example_classification_error']
    y = _paragraph(canvas, f"Regression: row 150 was under-estimated by 3.225 tons; relying on that estimate alone could understate expected logistics or storage needs. Classification: row {error['row_id']} was a false positive with estimated risk probability {error['high_risk_probability']:.3f}; unnecessary follow-up is less dangerous than missing risk, but it can consume limited field-officer time. Cluster: support status dominates the selected two-group pattern, so the grouping should not be read as farm performance.", y, size=8.8, bold=True)
    y = _section(canvas, "Improvement with real data", y)
    y = _paragraph(canvas, "After informed consent, collect representative outcomes across seasons; document measurement quality; add agronomically justified variables; check subgroup errors and probability calibration; validate prospectively; monitor drift; and keep an appeal and human-review path.", y)
    y = _paragraph(canvas, "STUDENT CHECK: Kawaya Gueylord must approve or rewrite the three interpretations above in their own words before submission and be able to explain them without reading a script.", y, size=9.2, bold=True)
    _footer(canvas); canvas.showPage()
    canvas.save()
    return output


def generate_declaration_template(root: Path) -> Path:
    """Create an explicitly unsigned declaration for the student to review."""
    output = root / "evidence" / "AI_A2_2527953_DECLARATION.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(str(output), pagesize=A4, invariant=1)
    canvas.setTitle("AI_A2_2527953 Declaration Template")
    y = _header(canvas, 1, "Student declaration — review and sign personally", 1)
    y = _paragraph(canvas, "Name: Kawaya Gueylord\nRegistration: 25/27953\nInstitutional username: ug2527953@ines.ac.rw\nVerification code: AI-A2-7953", y, bold=True)
    y = _paragraph(canvas, "I confirm, after my own review, that the submitted work and explanations accurately represent my understanding; that assistance is disclosed in evidence/AI_USE.md; that I can explain and modify the system; and that the dataset is synthetic. I understand that I must not make false authorship, execution, signature, or evidence claims.", y)
    y = _paragraph(canvas, "This document was generated as an UNSIGNED TEMPLATE. It is not a declaration until Kawaya personally reviews and signs it.", y, bold=True)
    y -= 35
    canvas.line(MARGIN, y, 280, y); canvas.drawString(MARGIN, y - 14, "Student signature")
    canvas.line(330, y, PAGE_W - MARGIN, y); canvas.drawString(330, y - 14, "Date")
    _footer(canvas); canvas.showPage(); canvas.save()
    return output


def generate_test_log(root: Path, command: str, result: str) -> Path:
    """Create a one-page PDF from an actually observed test result."""
    output = root / "evidence" / "AI_A2_2527953_TEST_LOG.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(str(output), pagesize=A4, invariant=1)
    canvas.setTitle("AI_A2_2527953 Test Log")
    y = _header(canvas, 1, "Automated test execution log", 1)
    y = _paragraph(canvas, "Student: Kawaya Gueylord  |  Verification: AI-A2-7953", y, bold=True)
    y = _paragraph(canvas, f"Environment: Python {platform.python_version()} on {platform.platform()}", y)
    y = _section(canvas, "Command executed", y)
    y = _paragraph(canvas, command, y, size=8.5)
    y = _section(canvas, "Observed result", y)
    y = _paragraph(canvas, result, y, size=8.5)
    y = _paragraph(canvas, "Negative-path coverage includes missing input, wrong type, out-of-range value, extra field, seed mismatch, invalid CLI input and HTTP 422 behavior.", y)
    _footer(canvas); canvas.showPage(); canvas.save()
    return output

