import os
import uuid
from datetime import datetime

os.environ.setdefault("MPLBACKEND", "Agg")

from flask import Flask, flash, redirect, render_template, request, send_file, session, url_for
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from werkzeug.utils import secure_filename

from predict import predict_pneumonia


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "xray-pneumonia-dev-secret")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_history():
    return session.setdefault("history", [])


def save_history_item(item):
    history = get_history()
    history.insert(0, item)
    session["history"] = history[:20]
    session.modified = True


def build_stats(history):
    total = len(history)
    normal = sum(1 for item in history if item["result"] == "NORMAL")
    pneumonia = sum(1 for item in history if item["result"] == "PNEUMONIA")
    return {
        "total": total,
        "normal": normal,
        "pneumonia": pneumonia,
        "normal_percent": round((normal / total) * 100, 1) if total else 0,
        "pneumonia_percent": round((pneumonia / total) * 100, 1) if total else 0,
    }


def create_pdf_report(analysis):
    report_path = os.path.join(REPORT_FOLDER, f"rapport_{analysis['id']}.pdf")
    image_path = os.path.join(BASE_DIR, analysis["image_path"])

    pdf = canvas.Canvas(report_path, pagesize=A4)
    width, height = A4

    pdf.setFillColor(colors.HexColor("#0f766e"))
    pdf.rect(0, height - 3.2 * cm, width, 3.2 * cm, stroke=0, fill=1)
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(2 * cm, height - 1.7 * cm, "Rapport de Diagnostic par Rayons X")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(2 * cm, height - 2.35 * cm, "Analyse assistee par intelligence artificielle")

    pdf.setFillColor(colors.HexColor("#111827"))
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(2 * cm, height - 4.3 * cm, "Informations de l'analyse")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(2 * cm, height - 5.1 * cm, f"Date: {analysis['date']}")
    pdf.drawString(2 * cm, height - 5.8 * cm, f"Diagnostic: {analysis['result']}")
    pdf.drawString(2 * cm, height - 6.5 * cm, f"Confiance: {analysis['confidence']}%")

    if os.path.exists(image_path):
        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(2 * cm, height - 7.7 * cm, "Radiographie analysee")
        pdf.drawImage(
            image_path,
            2 * cm,
            4 * cm,
            width=12 * cm,
            height=12 * cm,
            preserveAspectRatio=True,
            mask="auto",
        )

    pdf.setFont("Helvetica", 8)
    pdf.setFillColor(colors.HexColor("#6b7280"))
    pdf.drawString(
        2 * cm,
        2 * cm,
        "Ce rapport est genere automatiquement et ne remplace pas un avis medical professionnel.",
    )
    pdf.save()
    return report_path


@app.route("/")
def index():
    history = get_history()
    return render_template("index.html", history=history[:5], stats=build_stats(history))


@app.route("/analyze", methods=["POST"])
def analyze():
    if "xray_image" not in request.files:
        flash("Veuillez selectionner une image avant l'analyse.", "warning")
        return redirect(url_for("index"))

    file = request.files["xray_image"]

    if file.filename == "":
        flash("Aucun fichier selectionne.", "warning")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Format non autorise. Utilisez une image JPG, JPEG ou PNG.", "danger")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(upload_path)

    try:
        result, confidence = predict_pneumonia(
            upload_path,
            model_path=os.path.join(BASE_DIR, "best_DenseNet_pneumonia.keras"),
            model_type="DenseNet",
            output_path=os.path.join(REPORT_FOLDER, f"prediction_{unique_name}.png"),
        )
    except Exception as exc:
        flash(f"Erreur pendant l'analyse IA: {exc}", "danger")
        return redirect(url_for("index"))

    if result is None:
        flash("Impossible d'analyser cette image.", "danger")
        return redirect(url_for("index"))

    analysis = {
        "id": uuid.uuid4().hex,
        "filename": filename,
        "stored_filename": unique_name,
        "image_path": os.path.join("static", "uploads", unique_name),
        "image_url": url_for("static", filename=f"uploads/{unique_name}"),
        "result": result,
        "confidence": round(float(confidence) * 100, 2),
        "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
    }
    save_history_item(analysis)

    return render_template("result.html", analysis=analysis, history=get_history()[:5])


@app.route("/dashboard")
def dashboard():
    history = get_history()
    return render_template("dashboard.html", history=history, stats=build_stats(history))


@app.route("/report/<analysis_id>")
def report(analysis_id):
    analysis = next((item for item in get_history() if item["id"] == analysis_id), None)
    if not analysis:
        flash("Rapport introuvable pour cette session.", "warning")
        return redirect(url_for("dashboard"))

    report_path = create_pdf_report(analysis)
    return send_file(report_path, as_attachment=True, download_name=f"rapport_{analysis_id}.pdf")


@app.errorhandler(413)
def file_too_large(_error):
    flash("Image trop volumineuse. Taille maximale: 8 Mo.", "danger")
    return redirect(url_for("index"))


@app.errorhandler(500)
def internal_error(_error):
    flash("Une erreur interne est survenue. Veuillez reessayer.", "danger")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
