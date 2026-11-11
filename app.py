
import streamlit as st
import torch
import torch.nn as nn
import json
import cv2
import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image
from torchvision import transforms, models
from datetime import datetime
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as PDFImage,
    HRFlowable
)


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "plant_disease_model.pth"
CLASS_PATH = BASE_DIR / "class_names.json"

CONFIDENCE_THRESHOLD = 60.0


# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>
    .main {
        background-color: #f7faf7;
    }

    .block-container {
        padding-top: 1.5rem;
    }

    .main-title {
        color: #166534;
        font-size: 38px;
        font-weight: 800;
    }

    .subtitle {
        color: #64748b;
        font-size: 16px;
    }

    .section-title {
        color: #166534;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }

    .footer {
        text-align: center;
        color: #64748b;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. SESSION STATE
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ==========================================
# 4. DISEASE INFORMATION
# ==========================================

DISEASE_INFO = {

    "Pepper__bell___Bacterial_spot": {
        "name": "Pepper Bacterial Spot",
        "cause": "Bacterial infection",
        "symptoms": [
            "Small dark or water-soaked spots on leaves.",
            "Yellowing around affected areas.",
            "Severe infection may cause leaf drop."
        ],
        "treatment": [
            "Remove heavily infected leaves where practical.",
            "Avoid overhead watering.",
            "Keep sufficient space between plants.",
            "Use only locally approved bacterial disease management products after expert advice."
        ],
        "prevention": [
            "Use disease-free seeds and seedlings.",
            "Avoid handling wet plants.",
            "Clean gardening tools regularly.",
            "Maintain good field sanitation."
        ]
    },

    "Pepper__bell___healthy": {
        "name": "Healthy Pepper Leaf",
        "cause": "No disease predicted",
        "symptoms": [
            "No supported disease was identified by the model."
        ],
        "treatment": [
            "No disease treatment is indicated by this prediction.",
            "Continue normal watering and plant care."
        ],
        "prevention": [
            "Monitor leaves regularly.",
            "Maintain proper sunlight and drainage.",
            "Inspect plants for early symptoms."
        ]
    },

    "Potato___Early_blight": {
        "name": "Potato Early Blight",
        "cause": "Fungal disease",
        "symptoms": [
            "Brown circular leaf spots.",
            "Concentric ring patterns may appear.",
            "Older leaves are often affected first."
        ],
        "treatment": [
            "Remove severely affected leaves when practical.",
            "Avoid wetting foliage during irrigation.",
            "Maintain good spacing and airflow.",
            "Ask a local agricultural expert about an approved fungicide if necessary."
        ],
        "prevention": [
            "Rotate crops with non-host crops.",
            "Remove infected plant debris.",
            "Use healthy seed potatoes.",
            "Avoid excessive leaf wetness."
        ]
    },

    "Potato___Late_blight": {
        "name": "Potato Late Blight",
        "cause": "Oomycete pathogen",
        "symptoms": [
            "Dark, irregular lesions on leaves.",
            "Water-soaked areas may develop.",
            "In humid conditions, pale growth may appear underneath leaves."
        ],
        "treatment": [
            "Isolate or remove heavily infected plant material where practical.",
            "Avoid overhead irrigation.",
            "Check nearby plants for spreading symptoms.",
            "Seek prompt agricultural advice because late blight can spread rapidly.",
            "Use only locally approved late-blight management products according to their label."
        ],
        "prevention": [
            "Use healthy certified seed potatoes.",
            "Maintain field sanitation.",
            "Avoid prolonged leaf wetness.",
            "Monitor crops during cool and humid weather."
        ]
    },

    "Potato___healthy": {
        "name": "Healthy Potato Leaf",
        "cause": "No disease predicted",
        "symptoms": [
            "No supported disease was identified by the model."
        ],
        "treatment": [
            "No disease treatment is indicated by this prediction.",
            "Continue regular plant maintenance."
        ],
        "prevention": [
            "Use healthy planting material.",
            "Maintain proper soil drainage.",
            "Monitor plants regularly."
        ]
    },

    "Tomato_Bacterial_spot": {
        "name": "Tomato Bacterial Spot",
        "cause": "Bacterial infection",
        "symptoms": [
            "Small dark spots on leaves.",
            "Yellow areas may surround spots.",
            "Spots can also occur on stems and fruits."
        ],
        "treatment": [
            "Remove severely affected leaves where practical.",
            "Avoid working with plants when foliage is wet.",
            "Improve spacing and airflow.",
            "Consult an agricultural expert for approved management options."
        ],
        "prevention": [
            "Use disease-free seeds and seedlings.",
            "Sanitize tools.",
            "Avoid overhead watering.",
            "Remove infected crop debris."
        ]
    },

    "Tomato_Early_blight": {
        "name": "Tomato Early Blight",
        "cause": "Fungal disease",
        "symptoms": [
            "Brown circular spots on leaves.",
            "Target-like concentric rings.",
            "Yellowing around affected areas."
        ],
        "treatment": [
            "Remove severely affected lower leaves.",
            "Avoid overhead watering.",
            "Maintain good airflow around plants.",
            "Seek advice about locally approved fungicide options if infection progresses."
        ],
        "prevention": [
            "Rotate crops.",
            "Remove infected plant debris.",
            "Use healthy seedlings.",
            "Avoid excessive moisture on leaves."
        ]
    },

    "Tomato_Late_blight": {
        "name": "Tomato Late Blight",
        "cause": "Oomycete pathogen",
        "symptoms": [
            "Dark irregular leaf lesions.",
            "Water-soaked patches.",
            "Rapid spread in favorable humid conditions."
        ],
        "treatment": [
            "Remove heavily infected plant material where practical.",
            "Avoid overhead irrigation.",
            "Monitor nearby tomato and potato plants.",
            "Consult an agricultural expert promptly.",
            "Use only approved late-blight management products according to their label."
        ],
        "prevention": [
            "Use healthy seedlings.",
            "Maintain adequate spacing.",
            "Avoid prolonged leaf wetness.",
            "Remove infected plant debris."
        ]
    },

    "Tomato_Leaf_Mold": {
        "name": "Tomato Leaf Mold",
        "cause": "Fungal disease",
        "symptoms": [
            "Yellow patches on upper leaf surfaces.",
            "Olive or grayish mold-like growth underneath.",
            "Affected leaves may curl and dry."
        ],
        "treatment": [
            "Remove severely infected leaves.",
            "Improve ventilation.",
            "Reduce humidity around foliage.",
            "Consult an expert for appropriate disease management."
        ],
        "prevention": [
            "Avoid overcrowding plants.",
            "Provide adequate airflow.",
            "Water near the soil rather than over leaves.",
            "Keep greenhouse humidity under control."
        ]
    },

    "Tomato_Septoria_leaf_spot": {
        "name": "Tomato Septoria Leaf Spot",
        "cause": "Fungal disease",
        "symptoms": [
            "Small circular spots on leaves.",
            "Gray or tan centers with dark borders.",
            "Tiny dark structures may appear inside spots."
        ],
        "treatment": [
            "Remove severely affected lower leaves.",
            "Avoid overhead irrigation.",
            "Keep leaves dry where possible.",
            "Ask an agricultural expert about approved fungicide options."
        ],
        "prevention": [
            "Remove infected crop debris.",
            "Rotate crops.",
            "Maintain proper spacing.",
            "Disinfect gardening tools."
        ]
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "name": "Tomato Spider Mites",
        "cause": "Two-spotted spider mite infestation",
        "symptoms": [
            "Tiny pale dots or stippling on leaves.",
            "Yellowing or bronzing.",
            "Fine webbing may be visible during heavy infestation."
        ],
        "treatment": [
            "Inspect leaf undersides for mites.",
            "Remove heavily infested leaves where practical.",
            "Avoid unnecessary broad-spectrum pesticide use.",
            "Seek expert advice about suitable mite management options."
        ],
        "prevention": [
            "Monitor plants regularly.",
            "Reduce excessive dust around plants.",
            "Protect beneficial insects.",
            "Avoid unnecessary pesticide applications."
        ]
    },

    "Tomato__Target_Spot": {
        "name": "Tomato Target Spot",
        "cause": "Fungal disease",
        "symptoms": [
            "Brown circular leaf lesions.",
            "Concentric patterns may develop.",
            "Leaf yellowing and premature leaf drop."
        ],
        "treatment": [
            "Remove severely affected leaves.",
            "Improve plant ventilation.",
            "Avoid prolonged leaf wetness.",
            "Consult an agricultural expert for approved management options."
        ],
        "prevention": [
            "Rotate crops.",
            "Remove infected plant debris.",
            "Use healthy seedlings.",
            "Maintain adequate spacing."
        ]
    },

    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "cause": "Viral disease, commonly spread by whiteflies",
        "symptoms": [
            "Yellowing of leaves.",
            "Upward leaf curling.",
            "Stunted plant growth."
        ],
        "treatment": [
            "There is no curative treatment for an infected plant.",
            "Remove severely infected plants where appropriate to reduce spread.",
            "Manage whitefly populations using integrated pest management.",
            "Seek local agricultural advice."
        ],
        "prevention": [
            "Use resistant varieties where available.",
            "Use healthy seedlings.",
            "Monitor and manage whiteflies.",
            "Remove weeds that may host the virus."
        ]
    },

    "Tomato__Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "cause": "Viral disease",
        "symptoms": [
            "Mosaic-like light and dark green patterns.",
            "Leaf distortion.",
            "Reduced plant growth."
        ],
        "treatment": [
            "There is no curative treatment for an infected plant.",
            "Remove severely infected plants where appropriate.",
            "Avoid handling plants before cleaning hands and tools.",
            "Seek expert confirmation."
        ],
        "prevention": [
            "Use certified disease-free seeds.",
            "Disinfect tools.",
            "Wash hands before handling plants.",
            "Remove infected plant debris."
        ]
    },

    "Tomato_healthy": {
        "name": "Healthy Tomato Leaf",
        "cause": "No disease predicted",
        "symptoms": [
            "No supported disease was identified by the model."
        ],
        "treatment": [
            "No disease treatment is indicated by this prediction.",
            "Continue normal plant care."
        ],
        "prevention": [
            "Monitor leaves regularly.",
            "Maintain proper watering.",
            "Provide adequate sunlight and airflow."
        ]
    }
}


# ==========================================
# 5. LOAD MODEL
# ==========================================

@st.cache_data
def load_classes():

    with open(CLASS_PATH, "r") as file:
        return json.load(file)


@st.cache_resource
def load_model():

    class_names = load_classes()

    model = models.mobilenet_v2(weights=None)

    model.classifier[1] = nn.Linear(
        model.last_channel,
        len(class_names)
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


# ==========================================
# 6. IMAGE TRANSFORM
# ==========================================

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 7. SEVERITY FUNCTION
# ==========================================

def get_severity(percentage):

    if percentage <= 10:
        return "Low"

    elif percentage <= 30:
        return "Medium"

    return "High"


# ==========================================
# 8. DAMAGE ANALYSIS
# ==========================================

def analyze_leaf_damage(pil_image, predicted_class):

    rgb_image = np.array(
        pil_image.convert("RGB")
    )

    height, width = rgb_image.shape[:2]

    scale = min(
        1.0,
        900 / max(height, width)
    )

    if scale < 1.0:

        rgb_image = cv2.resize(
            rgb_image,
            (
                int(width * scale),
                int(height * scale)
            )
        )

    image = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2BGR
    )

    height, width = image.shape[:2]

    # Leaf segmentation
    mask = np.zeros(
        (height, width),
        np.uint8
    )

    bg_model = np.zeros(
        (1, 65),
        np.float64
    )

    fg_model = np.zeros(
        (1, 65),
        np.float64
    )

    rectangle = (
        int(width * 0.02),
        int(height * 0.02),
        int(width * 0.96),
        int(height * 0.96)
    )

    try:

        cv2.grabCut(
            image,
            mask,
            rectangle,
            bg_model,
            fg_model,
            3,
            cv2.GC_INIT_WITH_RECT
        )

        leaf_mask = np.where(
            (mask == cv2.GC_FGD) |
            (mask == cv2.GC_PR_FGD),
            255,
            0
        ).astype(np.uint8)

    except cv2.error:

        leaf_mask = np.zeros(
            (height, width),
            np.uint8
        )

    if np.count_nonzero(leaf_mask) < (
        height * width * 0.02
    ):

        hsv_temp = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        )

        leaf_mask = np.where(
            (hsv_temp[:, :, 1] > 30) &
            (hsv_temp[:, :, 2] > 25),
            255,
            0
        ).astype(np.uint8)

    leaf_mask = cv2.morphologyEx(
        leaf_mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8)
    )

    leaf_area = np.count_nonzero(leaf_mask)

    if leaf_area == 0:
        raise ValueError("Leaf region could not be identified.")

    # HSV analysis
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    brown = (
        (h >= 5) &
        (h <= 30) &
        (s > 45) &
        (v < 190)
    )

    yellow = (
        (h >= 15) &
        (h <= 38) &
        (s > 60) &
        (v > 80) &
        (v < 240)
    )

    dark = (
        (v < 65) &
        (s > 20)
    )

    # Overall visible damage
    visible_mask = (
        brown | yellow | dark
    ).astype(np.uint8) * 255

    visible_mask = cv2.bitwise_and(
        visible_mask,
        leaf_mask
    )

    visible_mask = cv2.morphologyEx(
        visible_mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8)
    )

    visible_area = np.count_nonzero(
        visible_mask
    )

    visible_percentage = (
        visible_area / leaf_area
    ) * 100

    visible_percentage = min(
        visible_percentage,
        100.0
    )

    visible_severity = get_severity(
        visible_percentage
    )

    # Disease-specific masks
    disease = predicted_class.lower()

    disease_percentage = None
    disease_severity = None

    disease_mask = np.zeros(
        (height, width),
        np.uint8
    )

    if "healthy" not in disease:

        if "late_blight" in disease:
            specific = brown | dark

        elif "early_blight" in disease:
            specific = brown

        elif "bacterial_spot" in disease:
            specific = brown | dark

        elif "septoria" in disease:
            specific = brown | dark

        elif "target_spot" in disease:
            specific = brown

        elif "leaf_mold" in disease:
            specific = yellow | brown

        elif "spider_mites" in disease:

            pale = (
                (s >= 20) &
                (s <= 110) &
                (v > 100)
            )

            specific = yellow | pale

        elif "yellowleaf" in disease:
            specific = yellow

        elif "mosaic" in disease:
            specific = yellow

        else:
            specific = brown | yellow | dark

        disease_mask = (
            specific.astype(np.uint8) * 255
        )

        disease_mask = cv2.bitwise_and(
            disease_mask,
            leaf_mask
        )

        disease_mask = cv2.morphologyEx(
            disease_mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8)
        )

        disease_area = np.count_nonzero(
            disease_mask
        )

        disease_percentage = (
            disease_area / leaf_area
        ) * 100

        disease_percentage = min(
            disease_percentage,
            100.0
        )

        disease_severity = get_severity(
            disease_percentage
        )

    # Create overlays
    visible_overlay = rgb_image.copy()
    visible_overlay[visible_mask > 0] = (255, 0, 0)

    disease_overlay = rgb_image.copy()
    disease_overlay[disease_mask > 0] = (255, 0, 0)

    return (
        visible_percentage,
        visible_severity,
        disease_percentage,
        disease_severity,
        visible_overlay,
        disease_overlay
    )


# ==========================================
# 9. TREATMENT RECOMMENDATIONS
# ==========================================

def get_recommendations(predicted_class):

    return DISEASE_INFO.get(
        predicted_class,
        {
            "name": predicted_class.replace("_", " "),
            "cause": "Unknown or unsupported category",
            "symptoms": [
                "Please seek expert confirmation."
            ],
            "treatment": [
                "Do not apply chemicals based only on this prediction.",
                "Consult a local agricultural expert."
            ],
            "prevention": [
                "Monitor plant health regularly.",
                "Maintain good field hygiene."
            ]
        }
    )


def display_recommendations(info):

    st.subheader("🦠 Disease Information")

    st.write("**Disease:**", info["name"])
    st.write("**Cause:**", info["cause"])

    with st.expander("🔍 Common Symptoms", expanded=True):

        for item in info["symptoms"]:
            st.write(f"• {item}")

    with st.expander("💊 Treatment and Care", expanded=True):

        for item in info["treatment"]:
            st.write(f"• {item}")

    with st.expander("🛡️ Prevention Tips"):

        for item in info["prevention"]:
            st.write(f"• {item}")


# ==========================================
# 10. PDF REPORT GENERATION
# ==========================================

def generate_pdf_report(result):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "PlantTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#166534"),
        alignment=TA_CENTER,
        fontSize=23,
        spaceAfter=10
    )

    heading_style = ParagraphStyle(
        "PlantHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#166534"),
        spaceBefore=12,
        spaceAfter=7
    )

    body_style = styles["BodyText"]

    story = []

    story.append(
        Paragraph("PlantCare AI", title_style)
    )

    story.append(
        Paragraph(
            "Plant Health Analysis Report",
            ParagraphStyle(
                "Subtitle",
                parent=styles["Normal"],
                alignment=TA_CENTER,
                textColor=colors.grey
            )
        )
    )

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            f"Report Date: {result['date']}",
            body_style
        )
    )

    story.append(
        Paragraph(
            f"Predicted Plant Category: "
            f"{result['disease'].replace('_', ' ')}",
            body_style
        )
    )

    story.append(
        Paragraph(
            f"Prediction Confidence: "
            f"{result['confidence']:.2f}%",
            body_style
        )
    )

    story.append(Spacer(1, 10))

    # Uploaded image
    story.append(
        Paragraph("Uploaded Plant Image", heading_style)
    )

    image_buffer = BytesIO(
        result["image_bytes"]
    )

    report_image = PDFImage(
        image_buffer,
        width=3.4 * inch,
        height=3.0 * inch,
        kind="proportional"
    )

    story.append(report_image)

    story.append(
        Paragraph("Damage Analysis", heading_style)
    )

    damage_data = [
        ["Measurement", "Result"],
        [
            "Overall Visible Damage",
            f"{result['visible_percentage']:.2f}%"
        ],
        [
            "Visible Severity",
            result["visible_severity"]
        ],
        [
            "Disease-Specific Damage",
            (
                "Not applicable"
                if result["disease_percentage"] is None
                else f"{result['disease_percentage']:.2f}%"
            )
        ],
        [
            "Disease-Specific Severity",
            (
                "Not applicable"
                if result["disease_severity"] is None
                else result["disease_severity"]
            )
        ]
    ]

    table = Table(
        damage_data,
        colWidths=[230, 200]
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#166534")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 8),
            (
                "BACKGROUND",
                (0, 1),
                (-1, -1),
                colors.HexColor("#f0fdf4")
            )
        ])
    )

    story.append(table)

    info = result["info"]

    story.append(
        Paragraph("Disease Information", heading_style)
    )

    story.append(
        Paragraph(
            f"Disease: {info['name']}",
            body_style
        )
    )

    story.append(
        Paragraph(
            f"Cause: {info['cause']}",
            body_style
        )
    )

    story.append(
        Paragraph("Common Symptoms", heading_style)
    )

    for item in info["symptoms"]:
        story.append(
            Paragraph(f"• {item}", body_style)
        )

    story.append(
        Paragraph("Treatment and Care", heading_style)
    )

    for item in info["treatment"]:
        story.append(
            Paragraph(f"• {item}", body_style)
        )

    story.append(
        Paragraph("Prevention", heading_style)
    )

    for item in info["prevention"]:
        story.append(
            Paragraph(f"• {item}", body_style)
        )

    story.append(Spacer(1, 15))

    story.append(
        HRFlowable(
            width="100%",
            color=colors.HexColor("#166534")
        )
    )

    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            "Disclaimer: This report is generated by an "
            "experimental AI classification and color-based "
            "damage estimation system. Results are not a "
            "confirmed agricultural diagnosis. Damage percentages "
            "are estimates, not validated disease severity "
            "measurements. Consult an agricultural expert before "
            "applying treatments.",
            styles["Italic"]
        )
    )

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ==========================================
# 11. SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🌿 PlantCare AI")

    st.write("Plant Health Assistant")

    st.divider()

    st.markdown("### Navigation")

    st.write("📊 Dashboard")
    st.write("🔬 Disease Detection")
    st.write("💊 Treatment Guide")
    st.write("📄 PDF Health Report")

    st.divider()

    st.caption(
        "Powered by MobileNetV2 and OpenCV"
    )


# ==========================================
# 12. MAIN HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🌿 PlantCare AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Smart Plant Disease Detection and Health Monitoring'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

dashboard_tab, detection_tab, about_tab = st.tabs([
    "📊 Dashboard",
    "🔬 Disease Detection",
    "ℹ️ About"
])


# ==========================================
# 13. DASHBOARD
# ==========================================

with dashboard_tab:

    st.header("📊 Plant Health Dashboard")

    history = st.session_state.history

    total_scans = len(history)

    healthy_count = sum(
        1 for item in history
        if "healthy" in item["disease"].lower()
    )

    disease_count = total_scans - healthy_count

    average_damage = (
        sum(item["visible"] for item in history) /
        total_scans
        if total_scans > 0 else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Scans", total_scans)
    col2.metric("Healthy Predictions", healthy_count)
    col3.metric("Disease Predictions", disease_count)

    col4.metric(
        "Average Visible Damage",
        f"{average_damage:.2f}%"
    )

    st.divider()

    if history:

        st.subheader("📈 Scan Overview")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:

            st.write("Healthy vs Disease")

            st.bar_chart(
                pd.DataFrame({
                    "Scans": {
                        "Healthy": healthy_count,
                        "Disease": disease_count
                    }
                })
            )

        with chart_col2:

            st.write("Severity Distribution")

            severity_counts = {
                "Low": 0,
                "Medium": 0,
                "High": 0
            }

            for item in history:
                severity_counts[item["severity"]] += 1

            st.bar_chart(
                pd.DataFrame({
                    "Scans": severity_counts
                })
            )

        st.divider()

        st.subheader("🕒 Recent Scan History")

        history_df = pd.DataFrame([
            {
                "Date": item["date"],
                "Disease": item["disease"].replace("_", " "),
                "Confidence": f"{item['confidence']:.2f}%",
                "Damage": f"{item['visible']:.2f}%",
                "Severity": item["severity"]
            }
            for item in reversed(history[-10:])
        ])

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        if st.button("Clear Dashboard History"):

            st.session_state.history = []

            st.rerun()

    else:

        st.info(
            "No scans available yet. "
            "Open Disease Detection and upload a plant image."
        )


# ==========================================
# 14. DISEASE DETECTION
# ==========================================

with detection_tab:

    st.header("🔬 AI Disease Detection")

    uploaded_file = st.file_uploader(
        "Upload your plant leaf image",
        type=["jpg", "jpeg", "png"],
        key="plant_upload"
    )

    if uploaded_file is not None:

        try:

            image_bytes = uploaded_file.getvalue()

            image = Image.open(
                BytesIO(image_bytes)
            ).convert("RGB")

            st.image(
                image,
                caption="Uploaded Plant Leaf",
                width=350
            )

            if st.button(
                "🔍 Analyze Plant",
                type="primary"
            ):

                with st.spinner(
                    "AI is analyzing your plant..."
                ):

                    model = load_model()
                    class_names = load_classes()

                    input_tensor = transform(
                        image
                    ).unsqueeze(0)

                    with torch.no_grad():

                        outputs = model(
                            input_tensor
                        )

                        probabilities = torch.softmax(
                            outputs,
                            dim=1
                        )

                    top_probs, top_indices = torch.topk(
                        probabilities,
                        k=min(3, len(class_names)),
                        dim=1
                    )

                    predicted_index = (
                        top_indices[0][0].item()
                    )

                    predicted_class = (
                        class_names[predicted_index]
                    )

                    confidence = (
                        top_probs[0][0].item() * 100
                    )

                    (
                        visible_percentage,
                        visible_severity,
                        disease_percentage,
                        disease_severity,
                        visible_overlay,
                        disease_overlay
                    ) = analyze_leaf_damage(
                        image,
                        predicted_class
                    )

                    info = get_recommendations(
                        predicted_class
                    )

                    result_date = datetime.now().strftime(
                        "%d-%m-%Y %I:%M:%S %p"
                    )

                    result = {
                        "date": result_date,
                        "disease": predicted_class,
                        "confidence": confidence,
                        "visible": visible_percentage,
                        "visible_percentage": visible_percentage,
                        "visible_severity": visible_severity,
                        "disease_percentage": disease_percentage,
                        "disease_severity": disease_severity,
                        "severity": visible_severity,
                        "info": info,
                        "image_bytes": image_bytes,
                        "visible_overlay": visible_overlay,
                        "disease_overlay": disease_overlay,
                        "top_predictions": [
                            (
                                class_names[top_indices[0][i].item()],
                                top_probs[0][i].item() * 100
                            )
                            for i in range(len(top_indices[0]))
                        ]
                    }

                    st.session_state.last_result = result

                    st.session_state.history.append({
                        "date": result_date,
                        "disease": predicted_class,
                        "confidence": confidence,
                        "visible": visible_percentage,
                        "severity": visible_severity
                    })

            # ==================================
            # SHOW LATEST RESULT
            # ==================================

            result = st.session_state.last_result

            if result is not None:

                st.divider()

                st.header("📊 Prediction Results")

                st.write(
                    "**Predicted Category:**",
                    result["disease"]
                )

                st.metric(
                    "Prediction Confidence",
                    f"{result['confidence']:.2f}%"
                )

                st.progress(
                    result["confidence"] / 100
                )

                if result["confidence"] < CONFIDENCE_THRESHOLD:

                    st.warning(
                        "Low-confidence prediction. "
                        "Please upload another clear image "
                        "or seek expert confirmation."
                    )

                elif "healthy" in result["disease"].lower():

                    st.success(
                        "🌱 Model prediction: Healthy leaf"
                    )

                else:

                    st.error(
                        "⚠️ Model prediction: Disease detected"
                    )

                # Damage analysis
                st.divider()

                st.header("🍃 Leaf Damage Analysis")

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Overall Visible Damage",
                        f"{result['visible_percentage']:.2f}%"
                    )

                    st.progress(
                        result["visible_percentage"] / 100
                    )

                with col2:

                    st.metric(
                        "Visible Severity",
                        result["visible_severity"]
                    )

                    if result["visible_severity"] == "Low":
                        st.success("Low visible damage")

                    elif result["visible_severity"] == "Medium":
                        st.warning("Medium visible damage")

                    else:
                        st.error("High visible damage")

                st.image(
                    result["visible_overlay"],
                    caption="Red = estimated visible discoloration",
                    width=500
                )

                st.divider()

                st.subheader(
                    "Disease-Specific Damage"
                )

                if result["disease_percentage"] is None:

                    st.info(
                        "Healthy prediction: Disease-specific "
                        "damage is not applicable."
                    )

                else:

                    col3, col4 = st.columns(2)

                    with col3:

                        st.metric(
                            "Disease-Specific Damage",
                            f"{result['disease_percentage']:.2f}%"
                        )

                        st.progress(
                            result["disease_percentage"] / 100
                        )

                    with col4:

                        st.metric(
                            "Disease-Specific Severity",
                            result["disease_severity"]
                        )

                        if result["disease_severity"] == "Low":
                            st.success("Low severity")

                        elif result["disease_severity"] == "Medium":
                            st.warning("Medium severity")

                        else:
                            st.error("High severity")

                    st.image(
                        result["disease_overlay"],
                        caption="Red = estimated disease-specific color regions",
                        width=500
                    )

                st.caption(
                    "Damage percentages are experimental color-based "
                    "estimates, not validated disease-area measurements."
                )

                # Top predictions
                st.divider()

                st.subheader("Top 3 Predictions")

                for rank, (name, score) in enumerate(
                    result["top_predictions"],
                    start=1
                ):

                    st.write(
                        f"{rank}. {name} — {score:.2f}%"
                    )

                    st.progress(score / 100)

                # ==================================
                # TREATMENT RECOMMENDATIONS
                # ==================================

                st.divider()

                st.header("💊 Treatment Recommendations")

                display_recommendations(
                    result["info"]
                )

                # ==================================
                # PDF REPORT
                # ==================================

                st.divider()

                st.header("📄 Download Plant Health Report")

                pdf_bytes = generate_pdf_report(
                    result
                )

                st.download_button(
                    label="⬇️ Download PDF Health Report",
                    data=pdf_bytes,
                    file_name="PlantCare_AI_Health_Report.pdf",
                    mime="application/pdf",
                    type="primary"
                )

        except Exception as e:

            st.error(
                f"Unable to analyze image: {e}"
            )


# ==========================================
# 15. ABOUT PAGE
# ==========================================

with about_tab:

    st.header("ℹ️ About PlantCare AI")

    st.write(
        "PlantCare AI is a plant health monitoring "
        "application developed using deep learning "
        "and computer vision."
    )

    st.subheader("Technologies Used")

    technologies = [
        "Python",
        "Streamlit",
        "PyTorch",
        "MobileNetV2",
        "OpenCV",
        "NumPy",
        "Pandas",
        "ReportLab"
    ]

    for technology in technologies:
        st.write(f"• {technology}")

    st.subheader("Main Features")

    features = [
        "Plant disease classification",
        "Prediction confidence",
        "Overall visible damage estimation",
        "Disease-specific symptom estimation",
        "Low, Medium and High severity",
        "Treatment and prevention guidance",
        "PDF plant health report",
        "Dashboard and scan history"
    ]

    for feature in features:
        st.write(f"✅ {feature}")

    st.warning(
        "This application is an educational project. "
        "AI predictions and color-based damage estimates "
        "are not substitutes for professional agricultural diagnosis."
    )


# ==========================================
# 16. FOOTER
# ==========================================

st.divider()

st.markdown(
    '<div class="footer">'
    '🌿 PlantCare AI | Plant Disease Detection Using Deep Learning'
    '</div>',
    unsafe_allow_html=True
)
