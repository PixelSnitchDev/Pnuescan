from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file, abort
import os
import sqlite3
import math
import uuid
from datetime import datetime
from functools import wraps

import numpy as np
import cv2
import requests
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from werkzeug.utils import secure_filename

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

app = Flask(__name__)
app.secret_key = "change-this-secret-key"

# -----------------------------
# Paths / config
# -----------------------------
UPLOAD_FOLDER = "static/uploads"
HEATMAP_FOLDER = "static/heatmaps"
REPORT_FOLDER = "reports"
DB_PATH = "pneumonia_scans.db"
MODEL_PATH = "pneumonia_model_v2.pth"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ["NORMAL", "PNEUMONIA"]

USERS = {
    "admin@pneuscan.com": "admin123"
}

# -----------------------------
# Preprocessing from Colab
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# Database helpers
# -----------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT UNIQUE,
                patient_name TEXT,
                age INTEGER,
                gender TEXT,
                submitted_by TEXT,
                status TEXT,
                confidence REAL,
                scan_time TEXT,
                image_path TEXT,
                heatmap_path TEXT,
                report_path TEXT
            )
        """)
        conn.commit()

def generate_case_id():
    return f"CASE-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

def insert_scan(case_id, patient_name, age, gender, submitted_by, status, confidence, scan_time, image_path, heatmap_path):
    with get_db() as conn:
        cur = conn.execute("""
            INSERT INTO scans (
                case_id, patient_name, age, gender, submitted_by,
                status, confidence, scan_time, image_path, heatmap_path, report_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            case_id, patient_name, age, gender, submitted_by,
            status, confidence, scan_time, image_path, heatmap_path, None
        ))
        conn.commit()
        return cur.lastrowid

def update_report_path(scan_id, report_path):
    with get_db() as conn:
        conn.execute("UPDATE scans SET report_path = ? WHERE id = ?", (report_path, scan_id))
        conn.commit()

def fetch_scan(scan_id):
    with get_db() as conn:
        return conn.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()

def fetch_all_scans():
    with get_db() as conn:
        return conn.execute("SELECT * FROM scans ORDER BY id DESC").fetchall()

def get_stats():
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
        normal = conn.execute("SELECT COUNT(*) FROM scans WHERE status = 'NORMAL'").fetchone()[0]
        pneumonia = conn.execute("SELECT COUNT(*) FROM scans WHERE status = 'PNEUMONIA'").fetchone()[0]
        recent = conn.execute("SELECT * FROM scans ORDER BY id DESC LIMIT 5").fetchall()
    return total, normal, pneumonia, recent

# -----------------------------
# Auth helpers
# -----------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

# -----------------------------
# Model loading
# -----------------------------
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model

model = load_model(MODEL_PATH)

# -----------------------------
# Image preprocessing / prediction
# -----------------------------
def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0)
    return image

def predict_image(image_path):
    x = preprocess_image(image_path).to(device)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item() * 100

    label = class_names[pred_idx]
    return label, round(confidence, 2), pred_idx

# -----------------------------
# Grad-CAM heatmap
# -----------------------------
def generate_gradcam(image_path, target_class_idx):
    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations["value"] = out

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    target_layer = model.layer4[-1].conv2

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    x = preprocess_image(image_path).to(device)
    model.zero_grad()

    outputs = model(x)
    score = outputs[:, target_class_idx]
    score.backward()

    fh.remove()
    bh.remove()

    acts = activations["value"].detach()[0]
    grads = gradients["value"].detach()[0]

    weights = torch.mean(grads, dim=(1, 2))
    cam = torch.zeros(acts.shape[1:], device=acts.device)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam.cpu().numpy()

    cam = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    original = Image.open(image_path).convert("RGB").resize((224, 224))
    original = np.array(original)

    overlay = np.uint8(0.6 * heatmap + 0.4 * original)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_name = f"{base_name}_heatmap.jpg"
    out_path = os.path.join(HEATMAP_FOLDER, out_name)

    Image.fromarray(overlay).save(out_path)

    return out_path

# -----------------------------
# PDF report
# -----------------------------
def generate_pdf_report(scan_row):
    report_name = f"report_{scan_row['id']}.pdf"
    report_path = os.path.join(REPORT_FOLDER, report_name)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="CenterTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1f2d3d")
    ))

    doc = SimpleDocTemplate(
        report_path,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )

    story = []
    story.append(Paragraph("Pneuscan Diagnostic Report", styles["CenterTitle"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"<b>Case ID:</b> {scan_row['case_id']}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Patient Name:</b> {scan_row['patient_name']}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Age:</b> {scan_row['age']}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Gender:</b> {scan_row['gender']}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Uploaded By:</b> {scan_row['submitted_by']}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Date:</b> {scan_row['scan_time']}", styles["BodyText"]))
    story.append(Spacer(1, 12))

    data = [
        ["Prediction", scan_row["status"]],
        ["Confidence", f"{scan_row['confidence']}%"]
    ]
    table = Table(data, colWidths=[5*cm, 9*cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf7f6")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(table)
    story.append(Spacer(1, 14))

    image_fs_path = scan_row["image_path"]
    heatmap_fs_path = scan_row["heatmap_path"]

    if image_fs_path and os.path.exists(image_fs_path):
        story.append(Paragraph("<b>Uploaded X-ray</b>", styles["Heading3"]))
        story.append(RLImage(image_fs_path, width=8*cm, height=8*cm))
        story.append(Spacer(1, 12))

    if heatmap_fs_path and os.path.exists(heatmap_fs_path):
        story.append(Paragraph("<b>Heatmap</b>", styles["Heading3"]))
        story.append(RLImage(heatmap_fs_path, width=8*cm, height=8*cm))
        story.append(Spacer(1, 12))

    story.append(Paragraph(
        "This report is intended for decision support and educational use. It should be reviewed by a qualified clinician.",
        styles["Italic"]
    ))

    doc.build(story)
    return report_path

# -----------------------------
# Hospital lookup
# -----------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

@app.route("/nearest-hospitals", methods=["POST"])
@login_required
def nearest_hospitals():
    data = request.get_json()

    if not data or "lat" not in data or "lon" not in data:
        return jsonify({"error": "Missing location"}), 400

    user_lat = float(data["lat"])
    user_lon = float(data["lon"])

    overpass_query = f"""
    [out:json][timeout:25];
    (
      node["amenity"="hospital"](around:10000,{user_lat},{user_lon});
      way["amenity"="hospital"](around:10000,{user_lat},{user_lon});
      relation["amenity"="hospital"](around:10000,{user_lat},{user_lon});
    );
    out center 20;
    """

    try:
        response = requests.get(
            "https://overpass-api.de/api/interpreter",
            params={"data": overpass_query},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return jsonify({"error": f"Hospital lookup failed: {str(e)}"}), 500

    hospitals = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name", "Hospital")

        if "lat" in el and "lon" in el:
            h_lat = el["lat"]
            h_lon = el["lon"]
        else:
            center = el.get("center", {})
            h_lat = center.get("lat")
            h_lon = center.get("lon")

        if h_lat is None or h_lon is None:
            continue

        dist = haversine(user_lat, user_lon, h_lat, h_lon)
        hospitals.append({
            "name": name,
            "lat": h_lat,
            "lon": h_lon,
            "distance_km": round(dist, 2)
        })

    hospitals = sorted(hospitals, key=lambda x: x["distance_km"])[:5]

    return jsonify({
        "user": {"lat": user_lat, "lon": user_lon},
        "hospitals": hospitals
    })

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    if session.get("logged_in"):
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if USERS.get(email) == password:
            session["logged_in"] = True
            session["username"] = email
            return redirect(url_for("dashboard"))
        error = "Invalid email or password"

    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    total, normal, pneumonia, recent = get_stats()
    return render_template(
        "dashboard.html",
        total=total,
        normal=normal,
        pneumonia=pneumonia,
        recent=recent,
        username=session.get("username")
    )

@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        patient_name = request.form.get("patient_name", "").strip()
        age = request.form.get("age", "").strip()
        gender = request.form.get("gender", "").strip()

        if "image" not in request.files:
            return render_template("upload.html", error="No file part")

        file = request.files["image"]

        if file.filename == "":
            return render_template("upload.html", error="No selected file")

        if not allowed_file(file.filename):
            return render_template("upload.html", error="Invalid file type")

        if not patient_name:
            return render_template("upload.html", error="Patient name is required")

        try:
            age_int = int(age)
        except:
            return render_template("upload.html", error="Age must be a number")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            result, confidence, pred_idx = predict_image(filepath)
            heatmap_fs_path = generate_gradcam(filepath, pred_idx)
            heatmap_url = url_for("static", filename=f"heatmaps/{os.path.basename(heatmap_fs_path)}")
            image_url = url_for("static", filename=f"uploads/{filename}")

            scan_time = datetime.now().strftime("%b %d, %Y, %I:%M %p")
            case_id = generate_case_id()

            scan_id = insert_scan(
                case_id=case_id,
                patient_name=patient_name,
                age=age_int,
                gender=gender,
                submitted_by=session.get("username", "Unknown"),
                status=result,
                confidence=confidence,
                scan_time=scan_time,
                image_path=filepath,
                heatmap_path=heatmap_fs_path
            )

            scan_row = fetch_scan(scan_id)
            report_path = generate_pdf_report(scan_row)
            update_report_path(scan_id, report_path)

            report_url = url_for("download_report", scan_id=scan_id)

            return render_template(
                "results.html",
                result=result,
                confidence=confidence,
                image_url=image_url,
                heatmap_url=heatmap_url,
                report_url=report_url,
                case_id=case_id,
                patient_name=patient_name,
                age=age_int,
                gender=gender,
                scan_time=scan_time
            )

        except Exception as e:
            return render_template("upload.html", error=f"Prediction failed: {str(e)}")

    return render_template("upload.html")

@app.route("/history")
@login_required
def history():
    scans = fetch_all_scans()
    return render_template("history.html", history=scans)

@app.route("/download-report/<int:scan_id>")
@login_required
def download_report(scan_id):
    scan = fetch_scan(scan_id)
    if not scan:
        abort(404)

    report_path = scan["report_path"]
    if not report_path or not os.path.exists(report_path):
        abort(404)

    return send_file(report_path, as_attachment=True, download_name=os.path.basename(report_path))

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    result, confidence, pred_idx = predict_image(filepath)
    heatmap_fs_path = generate_gradcam(filepath, pred_idx)
    heatmap_url = url_for("static", filename=f"heatmaps/{os.path.basename(heatmap_fs_path)}")

    return jsonify({
        "prediction": result,
        "confidence": confidence,
        "image_path": filepath,
        "heatmap_url": heatmap_url
    })

@app.route("/results")
@login_required
def results():
    return render_template("results.html")

# -----------------------------
# Init
# -----------------------------
init_db()

if __name__ == "__main__":
    app.run(debug=True)