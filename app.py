"""
Equine Lameness Location Predictor — Streamlit App
MU Veterinary Health Center, University of Missouri
"""

import os, pickle, joblib, warnings, re
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings('ignore')

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'pipeline.pkl')

st.set_page_config(
    page_title='Equine Lameness Predictor',
    page_icon='🐴',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown("""
<style>
  .block-container { padding-top: 1.2rem; }
  .stButton>button {
    background: #1B6CA8; color: white; border-radius: 8px;
    font-weight: 600; font-size: 15px; padding: 0.55rem 1.6rem;
    border: none; width: 100%;
  }
  .stButton>button:hover { background: #155a8a; }
  .example-btn button { background: #17A8A8 !important; font-size:13px !important; }
  .result-box {
    border-radius: 12px; padding: 22px; text-align: center;
    font-size: 1.7rem; font-weight: 700; color: white; margin: 10px 0;
  }
  .doc-box {
    background: #F0F4F8; border-left: 4px solid #1B6CA8;
    border-radius: 6px; padding: 14px 18px; margin-bottom: 10px;
  }
  .example-card {
    background: #F8FAFC; border: 1px solid #CBD5E1;
    border-radius: 10px; padding: 14px; margin-bottom: 8px;
  }
  .tag {
    display:inline-block; padding:2px 10px; border-radius:12px;
    font-size:0.78rem; font-weight:600; color:white; margin-right:4px;
  }
</style>
""", unsafe_allow_html=True)

# ─── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)

try:
    pipe = load_pipeline()
except FileNotFoundError:
    st.error('Model file not found. Run `python train_and_save.py` first.')
    st.stop()

stage1 = pipe['stage1']; lgbm2 = pipe['lgbm2']; xgb2 = pipe['xgb2']
et2 = pipe['et2']; meta = pipe['meta']; le2 = pipe['le2']
tfidf = pipe['tfidf']; label_encoders = pipe['label_encoders']
feature_cols = pipe['feature_cols']; sensor_feature_cols = pipe['sensor_feature_cols']
ANAT_FLAGS = pipe['anat_flags']; n_classes = pipe['n_classes']
CAT_COLS = pipe['cat_cols']; DATE_COLS = pipe['date_cols']

CLASS_COLORS  = {'distal':'#1B6CA8','middle':'#17A8A8','proximal':'#F39C12','nonlimb':'#E74C3C'}
CLASS_LABELS  = {'distal':'Distal (Foot/Fetlock)','middle':'Middle (Carpus/Hock/Stifle)',
                 'proximal':'Proximal (Shoulder/Hip)','nonlimb':'Non-Limb (Back/Spine)'}
CLASS_DESC    = {
    'distal':   'Lameness originates in the foot, pastern, or fetlock. '
                'Consider: hoof abscess, navicular syndrome, coffin joint arthritis, fetlock OA, laminitis.',
    'middle':   'Lameness originates in the carpus (knee), tarsus (hock), or stifle. '
                'Consider: carpal/tarsal OA, osteochondrosis (OCD), stifle meniscal injury, bog spavin.',
    'proximal': 'Lameness originates in the shoulder, elbow, hip, or upper limb. '
                'Consider: shoulder joint disease, coxofemoral arthritis, proximal suspensory desmitis, upward fixation of patella.',
    'nonlimb':  'Problem originates in back, sacroiliac, or cervical/thoracic/lumbar spine. '
                'Consider: kissing spines, sacroiliac strain, cervical facet arthropathy, epaxial muscle pain. '
                'Warrants axial skeleton workup — do NOT direct diagnostic blocks to limbs.',
}

# ─── Example Cases ───────────────────────────────────────────────────────────
EXAMPLES = {
    'Distal Lameness (Right Forelimb)': {
        'sensor': {
            'fore_diff_max_mean': 9.2, 'fore_diff_max_sd': 1.4,
            'fore_diff_min_mean': -7.1, 'fore_diff_min_sd': 1.1,
            'hind_diff_max_mean': 0.8, 'hind_diff_max_sd': 0.3,
            'hind_diff_min_mean': -0.6, 'hind_diff_min_sd': 0.2,
            'fore_vector_sum': 0.48, 'hind_vector_sum': 0.07,
            'fore_ratio_mean': 1.22, 'hind_ratio_mean': 1.01,
            'fore_trot_strides': 34, 'hind_trot_strides': 34,
            'fore_stride_rate': 149.0, 'hind_stride_rate': 149.0,
        },
        'med': {
            'USE_OF_HORSE': 'Sport/Performance', 'SHOD': 'Yes',
            'PREVIOUS_SURGERY': 'No', 'Any_Recent_Changes_In_Diet': 'No',
            'Grain/Concentrate': 'Yes', 'Hay': 'Yes', 'Pasture': 'No',
            'CARDIOVASCULAR': 'Normal', 'GENERAL_APPEARANCE': 'Normal',
            'INTEGUMENTARY': 'Normal', 'NERVOUS': 'Normal',
            'DIAGNOSTIC_NERVE/JOINT_BLOCKS': 'Positive',
            'EXERCISE_FLEXION/PROVOCATIVE_TESTS': 'Positive',
            'IMAGING': 'Radiographs Only',
            'LAMENESS_PALPATION/MANIPULATION': 'Grade 3',
            'DATE_OF_LAST_DENTAL_CARE': 180,
            'DATE_OF_LAST_HOOF_CARE': 42,
        },
        'history': (
            "6-year-old Warmblood sport horse presenting with a 3-week history of right forelimb "
            "lameness. Owner reports the horse is reluctant to work on hard ground and is short-striding "
            "at trot. No previous history of fetlock or hoof issues. Shod every 6 weeks. "
            "Mild digital pulse noted in right forelimb. Positive response to hoof testers over "
            "the navicular region and heel area."
        ),
        'reason': (
            "Right forelimb lameness grade 3/5 on hard surface, grade 2/5 on soft. "
            "Positive palmar digital nerve block abolished lameness. "
            "Positive fetlock flexion test. Radiographs: mild navicular bone remodeling. "
            "Coffin joint injection planned."
        ),
        'color': '#1B6CA8', 'expected': 'distal',
    },
    'Middle Lameness (Left Hindlimb — Hock)': {
        'sensor': {
            'fore_diff_max_mean': 1.2, 'fore_diff_max_sd': 0.4,
            'fore_diff_min_mean': -1.0, 'fore_diff_min_sd': 0.3,
            'hind_diff_max_mean': 7.8, 'hind_diff_max_sd': 1.6,
            'hind_diff_min_mean': -6.4, 'hind_diff_min_sd': 1.2,
            'fore_vector_sum': 0.10, 'hind_vector_sum': 0.41,
            'fore_ratio_mean': 1.02, 'hind_ratio_mean': 1.19,
            'fore_trot_strides': 30, 'hind_trot_strides': 30,
            'fore_stride_rate': 146.0, 'hind_stride_rate': 146.0,
        },
        'med': {
            'USE_OF_HORSE': 'Sport/Performance', 'SHOD': 'Yes',
            'PREVIOUS_SURGERY': 'No', 'Any_Recent_Changes_In_Diet': 'No',
            'Grain/Concentrate': 'Yes', 'Hay': 'Yes', 'Pasture': 'Yes',
            'CARDIOVASCULAR': 'Normal', 'GENERAL_APPEARANCE': 'Normal',
            'INTEGUMENTARY': 'Normal', 'NERVOUS': 'Normal',
            'DIAGNOSTIC_NERVE/JOINT_BLOCKS': 'Positive',
            'EXERCISE_FLEXION/PROVOCATIVE_TESTS': 'Positive',
            'IMAGING': 'Both',
            'LAMENESS_PALPATION/MANIPULATION': 'Grade 2',
            'DATE_OF_LAST_DENTAL_CARE': 200,
            'DATE_OF_LAST_HOOF_CARE': 55,
        },
        'history': (
            "10-year-old Quarter Horse show jumper with 6-week history of left hindlimb stiffness. "
            "Horse is ring sour and has difficulty with lead changes. Owner notes reduced impulsion "
            "and reluctance to collect. Bilateral hock stiffness on palpation. Positive distal limb "
            "flexion (spavin test) left hindlimb, grade 2 worsening to grade 3 after flex. "
            "Tarsus/hock region warm on palpation."
        ),
        'reason': (
            "Left hindlimb lameness grade 2/5. Strongly positive spavin (tarsal flexion) test. "
            "Distal tarsus intra-articular block significantly improved. "
            "Radiographs: distal intertarsal and tarsometatarsal joint narrowing consistent with bone spavin. "
            "Ultrasound: medial collateral ligament mild thickening. Hock injection performed."
        ),
        'color': '#17A8A8', 'expected': 'middle',
    },
    'Proximal Lameness (Shoulder)': {
        'sensor': {
            'fore_diff_max_mean': 5.1, 'fore_diff_max_sd': 2.1,
            'fore_diff_min_mean': -4.4, 'fore_diff_min_sd': 1.8,
            'hind_diff_max_mean': 1.5, 'hind_diff_max_sd': 0.6,
            'hind_diff_min_mean': -1.2, 'hind_diff_min_sd': 0.5,
            'fore_vector_sum': 0.28, 'hind_vector_sum': 0.11,
            'fore_ratio_mean': 1.12, 'hind_ratio_mean': 1.03,
            'fore_trot_strides': 28, 'hind_trot_strides': 28,
            'fore_stride_rate': 143.0, 'hind_stride_rate': 143.0,
        },
        'med': {
            'USE_OF_HORSE': 'Pleasure/Trail', 'SHOD': 'Yes',
            'PREVIOUS_SURGERY': 'No', 'Any_Recent_Changes_In_Diet': 'No',
            'Grain/Concentrate': 'No', 'Hay': 'Yes', 'Pasture': 'Yes',
            'CARDIOVASCULAR': 'Normal', 'GENERAL_APPEARANCE': 'Normal',
            'INTEGUMENTARY': 'Normal', 'NERVOUS': 'Normal',
            'DIAGNOSTIC_NERVE/JOINT_BLOCKS': 'Negative',
            'EXERCISE_FLEXION/PROVOCATIVE_TESTS': 'Positive',
            'IMAGING': 'Radiographs Only',
            'LAMENESS_PALPATION/MANIPULATION': 'Grade 2',
            'DATE_OF_LAST_DENTAL_CARE': 365,
            'DATE_OF_LAST_HOOF_CARE': 60,
        },
        'history': (
            "12-year-old Thoroughbred mare with insidious onset left forelimb lameness over 2 months. "
            "Short-strided at walk and trot. Distal limb nerve blocks did not improve lameness. "
            "Carpal blocks did not improve lameness. Swinging leg lameness with reduced cranial phase. "
            "Muscle atrophy over left shoulder region. Painful on deep palpation of bicipital bursa."
        ),
        'reason': (
            "Left forelimb lameness grade 2/5 not improved by distal or carpal blocks. "
            "Shoulder flexion and abduction painful. Bicipital bursa infiltration with local anesthetic "
            "significantly improved. Radiographs: shoulder joint mild degenerative changes. "
            "Ultrasound: bicipital tendon thickening and bursal effusion. "
            "Shoulder joint injection scheduled."
        ),
        'color': '#F39C12', 'expected': 'proximal',
    },
    'Non-Limb (Back / Sacroiliac)': {
        'sensor': {
            'fore_diff_max_mean': 2.3, 'fore_diff_max_sd': 1.8,
            'fore_diff_min_mean': -1.9, 'fore_diff_min_sd': 1.5,
            'hind_diff_max_mean': 3.1, 'hind_diff_max_sd': 2.0,
            'hind_diff_min_mean': -2.7, 'hind_diff_min_sd': 1.7,
            'fore_vector_sum': 0.15, 'hind_vector_sum': 0.19,
            'fore_ratio_mean': 1.05, 'hind_ratio_mean': 1.08,
            'fore_trot_strides': 26, 'hind_trot_strides': 26,
            'fore_stride_rate': 141.0, 'hind_stride_rate': 141.0,
        },
        'med': {
            'USE_OF_HORSE': 'Sport/Performance', 'SHOD': 'Yes',
            'PREVIOUS_SURGERY': 'No', 'Any_Recent_Changes_In_Diet': 'No',
            'Grain/Concentrate': 'Yes', 'Hay': 'Yes', 'Pasture': 'No',
            'CARDIOVASCULAR': 'Normal', 'GENERAL_APPEARANCE': 'Normal',
            'INTEGUMENTARY': 'Normal', 'NERVOUS': 'Normal',
            'DIAGNOSTIC_NERVE/JOINT_BLOCKS': 'Negative',
            'EXERCISE_FLEXION/PROVOCATIVE_TESTS': 'Negative',
            'IMAGING': 'Both',
            'LAMENESS_PALPATION/MANIPULATION': 'Grade 1',
            'DATE_OF_LAST_DENTAL_CARE': 120,
            'DATE_OF_LAST_HOOF_CARE': 50,
        },
        'history': (
            "8-year-old dressage Warmblood with 4-month history of poor performance and back pain. "
            "Rider reports loss of hindlimb engagement, resistance to collection, and bucking under saddle. "
            "All four limb nerve blocks negative. Horse shows epaxial muscle tension and pain on lumbar "
            "and sacral spine palpation. Sacroiliac region sensitive to pressure. "
            "No improvement with distal limb or hock joint blocks."
        ),
        'reason': (
            "Subtle bilateral hindlimb gait irregularity, worse under saddle. "
            "All distal and proximal limb nerve/joint blocks negative. "
            "Marked pain response on sacroiliac joint palpation and lumbar vertebral pressure. "
            "Nuclear scintigraphy: increased uptake thoracolumbar junction and bilateral sacroiliac joints. "
            "Ultrasound guided sacroiliac injection performed. Back and spine pathology confirmed."
        ),
        'color': '#E74C3C', 'expected': 'nonlimb',
    },
}

# ─── Inference helper ─────────────────────────────────────────────────────────
def predict(sensor_vals, med_vals, history_text, reason_text):
    combined = (history_text or '') + ' ' + (reason_text or '')
    tfidf_vec  = tfidf.transform([combined]).toarray()[0]
    tfidf_feat = {f'tfidf_{w}': tfidf_vec[i] for i, w in enumerate(tfidf.get_feature_names_out())}
    flag_feat  = {k: int(bool(re.search(p, combined.lower()))) for k, p in ANAT_FLAGS.items()}

    med_vec = {}
    for col in CAT_COLS:
        le = label_encoders.get(col)
        raw = str(med_vals.get(col, 'Unknown'))
        if le:
            try:    med_vec[col] = int(le.transform([raw])[0])
            except: med_vec[col] = int(le.transform(['Unknown'])[0]) if 'Unknown' in le.classes_ else 0
        else:
            med_vec[col] = 0
    for col in DATE_COLS:
        try:    med_vec[col] = float(med_vals.get(col, 0))
        except: med_vec[col] = 0.0

    sensor_feat = {c: float(sensor_vals.get(c, 0.0)) for c in sensor_feature_cols}
    agg_feat = {}
    for col in sensor_feature_cols:
        v = float(sensor_vals.get(col, 0.0))
        for stat, val in [('mean', v), ('std', 0.0), ('skew', 0.0),
                           ('iqr', 0.0), ('min', v), ('max', v)]:
            agg_feat[f'{col}_agg_{stat}'] = val

    all_feats = {**sensor_feat, **med_vec, **tfidf_feat, **flag_feat, **agg_feat}
    X = np.array([all_feats.get(c, 0.0) for c in feature_cols], dtype=float).reshape(1, -1)
    X = np.nan_to_num(X, nan=0.0)

    s1_prob   = stage1.predict_proba(X)[0]
    s1_pred   = int(stage1.predict(X)[0])
    nl_prob   = float(s1_prob[1])

    p2 = np.hstack([lgbm2.predict_proba(X), xgb2.predict_proba(X), et2.predict_proba(X)])
    limb_probs = meta.predict_proba(p2)[0]
    limb_pred  = meta.predict(p2)[0]
    pred_class = 'nonlimb' if s1_pred == 1 else le2.inverse_transform([limb_pred])[0]

    classes = list(le2.classes_)
    proba   = {cls: float(limb_probs[i]) * (1 - nl_prob) for i, cls in enumerate(classes)}
    proba['nonlimb'] = nl_prob
    return pred_class, proba, nl_prob, flag_feat


# ─── SESSION STATE — example loader ──────────────────────────────────────────
def load_example(name):
    ex = EXAMPLES[name]
    st.session_state['ex_sensor']  = ex['sensor']
    st.session_state['ex_med']     = ex['med']
    st.session_state['ex_history'] = ex['history']
    st.session_state['ex_reason']  = ex['reason']
    st.session_state['active_example'] = name

for key in ['ex_sensor','ex_med','ex_history','ex_reason','active_example']:
    if key not in st.session_state:
        st.session_state[key] = {} if 'ex_' in key else None

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🐴 Equine Lameness")
    st.markdown("**MU Veterinary Health Center**  \nUniversity of Missouri")
    st.markdown("---")

    tab_sel = st.radio("Navigation", ["Predictor", "Documentation", "About"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### Load Example Case")
    st.markdown("*Click to auto-fill all fields*")
    for ex_name, ex_data in EXAMPLES.items():
        color = ex_data['color']
        label = CLASS_LABELS[ex_data['expected']]
        active = st.session_state.get('active_example') == ex_name
        badge = "✓ " if active else ""
        if st.button(f"{badge}📋 {ex_name}", key=f"btn_{ex_name}", use_container_width=True):
            load_example(ex_name)
            st.rerun()

    st.markdown("---")
    st.markdown("### Model Performance")
    st.markdown(
        "| Metric | Value |\n|---|---|\n"
        "| Accuracy | **94.4%** |\n"
        "| Distal F1 | **0.97** |\n"
        "| Middle F1 | **0.94** |\n"
        "| Proximal F1 | **0.95** |\n"
        "| Nonlimb Recall | **85%** |\n"
        "| Training size | 8,454 strides |\n"
        "| Horses | 514 |"
    )


# ──────────────────────────────────────────────────────────────────────────────
# TAB: DOCUMENTATION
# ──────────────────────────────────────────────────────────────────────────────
if tab_sel == "Documentation":
    st.title("📖 Documentation & User Guide")
    st.markdown("---")

    st.header("Overview")
    st.markdown("""
This tool predicts the **anatomical location of equine lameness** from a combination of:
- **Inertial sensor (IMU) gait data** — objective asymmetry metrics recorded during a standardized trot-up
- **Structured clinical records** — examination findings, management details, diagnostic results
- **Veterinary narrative text** — free-text history and reason for visit

The model covers four lameness zones:

| Class | Region | Prevalence |
|---|---|---|
| **Distal** | Foot, pastern, fetlock | 42.9% |
| **Middle** | Carpus (knee), tarsus (hock), stifle | 41.5% |
| **Proximal** | Shoulder, elbow, hip, upper hindlimb | 12.4% |
| **Non-Limb** | Back, sacroiliac, cervical/thoracic/lumbar spine | 3.2% |
""")

    st.markdown("---")
    st.header("Step-by-Step Guide")

    with st.expander("Step 1 — Enter Sensor Data", expanded=True):
        st.markdown("""
The sensor inputs come from your **body-mounted IMU gait analysis report** (e.g., Lameness Locator, Equinosis Q).

**Forelimb & Hindlimb fields:**
| Field | Description | Typical Range |
|---|---|---|
| `Diff Max Mean` | Mean peak upward displacement asymmetry (mm) | -20 to +20 mm |
| `Diff Max SD` | Stride-to-stride variability of peak asymmetry | 0 – 5 mm |
| `Diff Min Mean` | Mean trough (downward) displacement asymmetry | -20 to +20 mm |
| `Diff Min SD` | Variability of trough asymmetry | 0 – 5 mm |
| `Vector Sum` | Combined 3D asymmetry magnitude | 0 – 1.0 |
| `Ratio Mean` | Limb loading ratio (1.0 = symmetric) | 0.8 – 1.4 |
| `Trot Strides` | Number of strides recorded in the session | 20 – 80 |
| `Stride Rate` | Cadence in strides per minute | 130 – 165 |

> **Tip:** For lame forelimb, expect high `fore_diff_max_mean` (positive = right limb lame, negative = left). For lame hindlimb, expect high `hind_diff_max_mean`.

> **If sensor data is unavailable**, leave all fields at 0 — the model will rely more heavily on the clinical and text features.
""")

    with st.expander("Step 2 — Fill Clinical Information"):
        st.markdown("""
These structured fields come from the **EHR / clinical record**. They are optional but improve accuracy.

**Management fields:**
- `Use of Horse`: Sport/performance horses tend to have different lameness distributions than pleasure horses
- `Shod`: Shoeing status affects certain hoof-region diagnoses
- `Previous Surgery`: Prior surgical history influences likelihood of recurrence

**Examination fields:**
- `Lameness Grade`: Use the AAEP 0–5 scale
  - **0** = No perceptible lameness
  - **1** = Difficult to observe, not consistently apparent
  - **2** = Difficult to observe at walk, consistently apparent at trot
  - **3** = Consistently observable at trot
  - **4** = Obvious lameness, marked head nod/hip hike
  - **5** = Minimal weight bearing at rest
- `Diagnostic Nerve/Joint Blocks`: Whether blocks were performed and result
- `Exercise/Flexion Tests`: Whether flexion tests were positive
- `Imaging`: What imaging modality was used

**Date fields:**
- Days since last dental care and hoof care — longer intervals correlate with some lameness types
""")

    with st.expander("Step 3 — Paste Veterinary Notes"):
        st.markdown("""
The **free-text fields** are processed with TF-IDF and anatomical keyword detection.

**Best practices for text input:**
- Copy and paste directly from the case record — no need to edit
- Mention **specific anatomical structures** where possible (e.g., "fetlock", "stifle", "navicular", "sacroiliac")
- Include **nerve block results** — these are strongly predictive
- Include **imaging findings** if available
- Both the `History` and `Reason` fields are used; fill whichever you have

**High-signal keywords the model uses:**
| Anatomy | Keywords |
|---|---|
| Distal | fetlock, navicular, coffin, pastern, hoof, pedal bone, coronet |
| Middle | hock, tarsus, stifle, carpus, knee, spavin, OCD, meniscus |
| Proximal | shoulder, hip, elbow, pelvis, bicipital, suspensory proximal |
| Non-Limb | back, spine, sacroiliac, SI, lumbar, thoracic, cervical, kissing spines |
""")

    with st.expander("Step 4 — Interpret Results"):
        st.markdown("""
After clicking **Predict**, you will see:

**Result Banner** — predicted class with confidence percentage.

**Probability Bars** — confidence across all four classes. A well-calibrated prediction will show one dominant class; diffuse bars suggest ambiguity.

**Clinical Interpretation** — plain-English description of the predicted region and what diagnoses to consider.

**Pipeline Details** (expandable) — shows the Stage 1 (non-limb vs. limb) probability and Stage 2 (limb region) probabilities separately.

**Keyword Detection** (expandable) — lists which anatomical terms were found in the notes.

---

### Understanding Confidence Scores

| Confidence | Interpretation |
|---|---|
| > 80% | High confidence — model strongly favors one region |
| 50–80% | Moderate confidence — consider the top 2 classes |
| < 50% | Low confidence — ambiguous presentation, consider clinical context |

> **Important:** This tool is a clinical **decision support** aid, not a replacement for veterinary examination. Always interpret results in the context of the full clinical picture.
""")

    st.markdown("---")
    st.header("Example Cases")

    for ex_name, ex in EXAMPLES.items():
        color = ex['color']
        label = CLASS_LABELS[ex['expected']]
        with st.expander(f"Example: {ex_name}"):
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown(f"**Expected Prediction:** <span style='background:{color};color:white;padding:3px 10px;border-radius:10px;font-weight:600'>{label}</span>", unsafe_allow_html=True)
                st.markdown(f"**History:**\n\n> {ex['history']}")
                st.markdown(f"**Reason for Visit:**\n\n> {ex['reason']}")
            with c2:
                st.markdown("**Key Sensor Values:**")
                for k in ['fore_diff_max_mean','hind_diff_max_mean','fore_vector_sum','hind_vector_sum']:
                    v = ex['sensor'].get(k, 0)
                    st.markdown(f"- `{k}`: **{v}**")
                st.markdown("**Key Clinical Findings:**")
                st.markdown(f"- Blocks: {ex['med'].get('DIAGNOSTIC_NERVE/JOINT_BLOCKS','—')}")
                st.markdown(f"- Lameness Grade: {ex['med'].get('LAMENESS_PALPATION/MANIPULATION','—')}")
                st.markdown(f"- Imaging: {ex['med'].get('IMAGING','—')}")
            st.markdown("---")
            if st.button(f"Load This Example into Predictor", key=f"doc_btn_{ex_name}"):
                load_example(ex_name)
                st.session_state['tab_switch'] = 'Predictor'
                st.rerun()

    st.markdown("---")
    st.header("Frequently Asked Questions")

    with st.expander("Can I use this without sensor data?"):
        st.markdown("""
Yes. Leave all sensor fields at 0. The model will rely entirely on the clinical record and veterinary notes.
Accuracy will be somewhat lower without sensor data, but clinical text features (especially nerve block results
and anatomical keywords) still provide strong predictive signal.
""")

    with st.expander("What if I only have one of the text fields?"):
        st.markdown("""
Both `History` and `Reason` are combined into a single text input internally.
Fill whichever field you have — even a single sentence with the relevant anatomy
(e.g., "positive fetlock flexion") will activate the anatomical keyword features.
""")

    with st.expander("How was the model trained?"):
        st.markdown("""
The model was trained on **10,568 stride observations** from **514 horses** at the MU Veterinary Health Center.
It uses a **two-stage stacking ensemble**:
1. **Stage 1**: LightGBM binary classifier separates non-limb from limb cases
2. **Stage 2**: LightGBM + XGBoost + Extra Trees → Logistic Regression meta-learner classifies distal/middle/proximal

Features: 16 sensor metrics + 17 medical fields + 20 anatomical flags + 60 TF-IDF terms + 96 case-level aggregations = **209 total features**.
""")

    with st.expander("What does 'Non-Limb' mean clinically?"):
        st.markdown("""
**Non-Limb** means the primary source of the gait abnormality is **axial** — the back, sacroiliac joints,
or neck/spine — rather than any of the four limbs. This is clinically critical because:

- Limb nerve blocks **will not improve** a non-limb lameness
- Treatment is directed at the spine/SI (mesotherapy, chiropractic, injection, NSAID therapy)
- Missing this diagnosis leads to costly and ineffective limb workup

The model has **85% recall** for non-limb cases — it catches most of these presentations.
A positive non-limb prediction should prompt sacroiliac and spinal palpation and imaging.
""")


# ──────────────────────────────────────────────────────────────────────────────
# TAB: ABOUT
# ──────────────────────────────────────────────────────────────────────────────
elif tab_sel == "About":
    st.title("About This Tool")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
### Equine Lameness Location Predictor

This tool was developed at the **MU Veterinary Health Center, University of Missouri**
to assist veterinarians in localizing equine lameness using machine learning.

**Research Summary:**
Through systematic experimentation across eight model iterations, overall accuracy improved
from a 49.3% baseline to **94.4%** on a held-out test set of 2,114 stride observations.
The full study is documented in our conference report.

**Key Innovations:**
- Two-stage hierarchical classifier for non-limb detection
- Case-level gait aggregation features (16 sensors × 6 statistics)
- Multi-modal fusion: sensor + EHR + free text
- Stacking ensemble (LightGBM + XGBoost + Extra Trees → Logistic Regression)
- Bayesian hyperparameter optimization (Optuna, 40 trials × 5-fold CV)

**Intended Use:**
Clinical decision support for equine practitioners performing lameness workups.
Results should be interpreted alongside physical examination findings.

**Disclaimer:**
This tool is for research and clinical decision support only.
It is not FDA-approved and does not replace veterinary clinical judgment.
""")
    with col2:
        st.markdown("### Model Card")
        st.markdown("""
| | |
|---|---|
| **Version** | 1.0 |
| **Algorithm** | Two-stage stacking |
| **Accuracy** | 94.4% |
| **Dataset** | MU VHC (514 horses) |
| **Features** | 209 |
| **Classes** | 4 |
| **CV** | 5-fold stratified |
| **Framework** | LightGBM, XGBoost |
""")

    st.markdown("---")
    st.markdown("""
### Experiment History

| Experiment | Configuration | Accuracy | Delta |
|---|---|---|---|
| 1 | Baseline LightGBM (sensor only) | 49.3% | — |
| 2 | + class_weight=balanced | 54.7% | +5.4% |
| 3 | + Medical record features | 58.3% | +3.6% |
| 4 | + TF-IDF text + anatomical flags | 68.5% | +10.2% |
| 5 | Two-stage architecture | 78.3% | +9.8% |
| 6 | + Case-level gait aggregations | 85.2% | +6.9% |
| 7 | + Optuna Bayesian HPO | 85.2% | +0.0% |
| 8 | + Stacking ensemble | **94.4%** | **+9.2%** |

### Contact
For questions or collaboration inquiries, contact the MU Veterinary Health Center.
""")


# ──────────────────────────────────────────────────────────────────────────────
# TAB: PREDICTOR
# ──────────────────────────────────────────────────────────────────────────────
else:
    st.title("🐴 Equine Lameness Location Predictor")
    st.markdown(
        "Predict the anatomical location of equine lameness from inertial sensor gait data "
        "and clinical records. Use the **sidebar** to load an example case or navigate to **Documentation** for a full guide."
    )

    if st.session_state.get('active_example'):
        st.success(f"Example loaded: **{st.session_state['active_example']}** — fields have been pre-filled below.")

    st.markdown("---")

    # ── Sensor Data ───────────────────────────────────────────────────────────
    st.subheader("1. Inertial Sensor Data")
    st.caption("Values from the IMU gait analysis report (Lameness Locator or equivalent). Leave at 0 if unavailable.")

    def sv(key, default=0.0):
        return st.session_state.get('ex_sensor', {}).get(key, default)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Forelimb**")
        fore_diff_max_mean = st.number_input("Diff Max Mean (mm)",   key="f1", value=sv('fore_diff_max_mean'), step=0.1, format="%.2f")
        fore_diff_max_sd   = st.number_input("Diff Max SD (mm)",     key="f2", value=sv('fore_diff_max_sd'),   step=0.1, format="%.2f")
        fore_diff_min_mean = st.number_input("Diff Min Mean (mm)",   key="f3", value=sv('fore_diff_min_mean'), step=0.1, format="%.2f")
        fore_diff_min_sd   = st.number_input("Diff Min SD (mm)",     key="f4", value=sv('fore_diff_min_sd'),   step=0.1, format="%.2f")
        fore_vector_sum    = st.number_input("Vector Sum",           key="f5", value=sv('fore_vector_sum'),    step=0.01, format="%.3f")
        fore_ratio_mean    = st.number_input("Ratio Mean",           key="f6", value=sv('fore_ratio_mean', 1.0), step=0.01, format="%.3f")
        fore_trot_strides  = st.number_input("Trot Strides",         key="f7", value=int(sv('fore_trot_strides')), step=1, min_value=0)
        fore_stride_rate   = st.number_input("Stride Rate (str/min)",key="f8", value=sv('fore_stride_rate'),   step=0.1, format="%.1f")

    with col2:
        st.markdown("**Hindlimb**")
        hind_diff_max_mean = st.number_input("Diff Max Mean (mm)",   key="h1", value=sv('hind_diff_max_mean'), step=0.1, format="%.2f")
        hind_diff_max_sd   = st.number_input("Diff Max SD (mm)",     key="h2", value=sv('hind_diff_max_sd'),   step=0.1, format="%.2f")
        hind_diff_min_mean = st.number_input("Diff Min Mean (mm)",   key="h3", value=sv('hind_diff_min_mean'), step=0.1, format="%.2f")
        hind_diff_min_sd   = st.number_input("Diff Min SD (mm)",     key="h4", value=sv('hind_diff_min_sd'),   step=0.1, format="%.2f")
        hind_vector_sum    = st.number_input("Vector Sum",           key="h5", value=sv('hind_vector_sum'),    step=0.01, format="%.3f")
        hind_ratio_mean    = st.number_input("Ratio Mean",           key="h6", value=sv('hind_ratio_mean', 1.0), step=0.01, format="%.3f")
        hind_trot_strides  = st.number_input("Trot Strides",         key="h7", value=int(sv('hind_trot_strides')), step=1, min_value=0)
        hind_stride_rate   = st.number_input("Stride Rate (str/min)",key="h8", value=sv('hind_stride_rate'),   step=0.1, format="%.1f")

    sensor_vals = {
        'fore_diff_max_mean': fore_diff_max_mean, 'fore_diff_max_sd': fore_diff_max_sd,
        'fore_diff_min_mean': fore_diff_min_mean, 'fore_diff_min_sd': fore_diff_min_sd,
        'hind_diff_max_mean': hind_diff_max_mean, 'hind_diff_max_sd': hind_diff_max_sd,
        'hind_diff_min_mean': hind_diff_min_mean, 'hind_diff_min_sd': hind_diff_min_sd,
        'fore_vector_sum': fore_vector_sum, 'hind_vector_sum': hind_vector_sum,
        'fore_ratio_mean': fore_ratio_mean, 'hind_ratio_mean': hind_ratio_mean,
        'fore_trot_strides': fore_trot_strides, 'hind_trot_strides': hind_trot_strides,
        'fore_stride_rate': fore_stride_rate, 'hind_stride_rate': hind_stride_rate,
    }

    st.markdown("---")

    # ── Clinical Information ──────────────────────────────────────────────────
    st.subheader("2. Clinical Information")
    st.caption("Structured EHR fields. All optional — fill what is available.")

    def mv(key, default='Unknown'):
        return st.session_state.get('ex_med', {}).get(key, default)

    use_opts    = ['Sport/Performance','Pleasure/Trail','Working/Ranch','Breeding','Other','Unknown']
    shod_opts   = ['Yes','No','Partial','Unknown']
    surg_opts   = ['No','Yes','Unknown']
    yn_opts     = ['Yes','No','Unknown']
    exam_opts   = ['Normal','Abnormal','Not Examined','Unknown']
    gen_opts    = ['Normal','Mildly Abnormal','Moderately Abnormal','Severely Abnormal','Unknown']
    blk_opts    = ['None Performed','Positive','Negative','Partial Block','Unknown']
    flex_opts   = ['None Performed','Positive','Negative','Equivocal','Unknown']
    img_opts    = ['None Performed','Radiographs Only','Ultrasound Only','Both','MRI','CT','Unknown']
    lame_opts   = ['Grade 0','Grade 1','Grade 2','Grade 3','Grade 4','Grade 5','Unknown']

    def sel_idx(opts, val):
        try:    return opts.index(str(val))
        except: return len(opts) - 1

    c3, c4, c5 = st.columns(3)
    with c3:
        st.markdown("**Horse & Management**")
        use_of_horse = st.selectbox("Use of Horse",          use_opts,  index=sel_idx(use_opts,  mv('USE_OF_HORSE')))
        shod         = st.selectbox("Shod",                  shod_opts, index=sel_idx(shod_opts, mv('SHOD')))
        prev_surgery = st.selectbox("Previous Surgery",      surg_opts, index=sel_idx(surg_opts, mv('PREVIOUS_SURGERY')))
        diet_changes = st.selectbox("Recent Diet Changes",   yn_opts,   index=sel_idx(yn_opts,   mv('Any_Recent_Changes_In_Diet')))
        grain        = st.selectbox("Grain/Concentrate",     yn_opts,   index=sel_idx(yn_opts,   mv('Grain/Concentrate')))
        hay          = st.selectbox("Hay",                   yn_opts,   index=sel_idx(yn_opts,   mv('Hay')))
        pasture      = st.selectbox("Pasture Access",        yn_opts,   index=sel_idx(yn_opts,   mv('Pasture')))
    with c4:
        st.markdown("**Physical Examination**")
        cardiovascular = st.selectbox("Cardiovascular",    exam_opts, index=sel_idx(exam_opts, mv('CARDIOVASCULAR')))
        general_appear = st.selectbox("General Appearance", gen_opts, index=sel_idx(gen_opts,  mv('GENERAL_APPEARANCE')))
        integumentary  = st.selectbox("Integumentary",     exam_opts, index=sel_idx(exam_opts, mv('INTEGUMENTARY')))
        nervous        = st.selectbox("Nervous System",    exam_opts, index=sel_idx(exam_opts, mv('NERVOUS')))
        dental_days    = st.number_input("Days Since Last Dental Care", value=int(mv('DATE_OF_LAST_DENTAL_CARE', 180)), min_value=0, max_value=3650)
        hoof_days      = st.number_input("Days Since Last Hoof Care",   value=int(mv('DATE_OF_LAST_HOOF_CARE', 45)),   min_value=0, max_value=3650)
    with c5:
        st.markdown("**Diagnostic Findings**")
        diag_blocks  = st.selectbox("Nerve/Joint Blocks",      blk_opts,  index=sel_idx(blk_opts,  mv('DIAGNOSTIC_NERVE/JOINT_BLOCKS')))
        exercise_flex= st.selectbox("Exercise/Flexion Tests",  flex_opts, index=sel_idx(flex_opts, mv('EXERCISE_FLEXION/PROVOCATIVE_TESTS')))
        imaging      = st.selectbox("Imaging Performed",       img_opts,  index=sel_idx(img_opts,  mv('IMAGING')))
        lame_palp    = st.selectbox("Lameness Grade (AAEP 0–5)", lame_opts, index=sel_idx(lame_opts, mv('LAMENESS_PALPATION/MANIPULATION')))

    med_vals = {
        'USE_OF_HORSE': use_of_horse, 'SHOD': shod,
        'PREVIOUS_SURGERY': prev_surgery, 'Any_Recent_Changes_In_Diet': diet_changes,
        'Grain/Concentrate': grain, 'Hay': hay, 'Pasture': pasture,
        'CARDIOVASCULAR': cardiovascular, 'GENERAL_APPEARANCE': general_appear,
        'INTEGUMENTARY': integumentary, 'NERVOUS': nervous,
        'DIAGNOSTIC_NERVE/JOINT_BLOCKS': diag_blocks,
        'EXERCISE_FLEXION/PROVOCATIVE_TESTS': exercise_flex,
        'IMAGING': imaging, 'LAMENESS_PALPATION/MANIPULATION': lame_palp,
        'DATE_OF_LAST_DENTAL_CARE': dental_days, 'DATE_OF_LAST_HOOF_CARE': hoof_days,
    }

    st.markdown("---")

    # ── Veterinary Notes ──────────────────────────────────────────────────────
    st.subheader("3. Veterinary Narrative Notes")
    st.caption("Paste from the case record. Anatomical keywords (fetlock, stifle, navicular, sacroiliac…) strongly influence the prediction.")

    ex_hist = st.session_state.get('ex_history', '')
    ex_reas = st.session_state.get('ex_reason',  '')

    c6, c7 = st.columns(2)
    with c6:
        history_text = st.text_area("History of Current Problem", value=ex_hist, height=150,
            placeholder="e.g., 6-year-old Warmblood with 3-week right forelimb lameness, short-striding at trot, "
                        "positive response to hoof testers over navicular region...")
    with c7:
        reason_text = st.text_area("Reason for Visit / Chief Complaint", value=ex_reas, height=150,
            placeholder="e.g., Right forelimb grade 3/5 lameness. Positive palmar digital nerve block. "
                        "Fetlock flexion positive. Radiographs: navicular remodeling...")

    st.markdown("---")

    # ── Predict Button ────────────────────────────────────────────────────────
    pcol, _ = st.columns([2, 3])
    with pcol:
        do_predict = st.button("🔍  Predict Lameness Location", use_container_width=True)

    if do_predict:
        with st.spinner("Running two-stage prediction pipeline..."):
            pred_class, proba, nl_prob, flag_feat = predict(sensor_vals, med_vals, history_text, reason_text)

        st.markdown("---")
        st.subheader("Prediction Results")

        color = CLASS_COLORS[pred_class]
        label = CLASS_LABELS[pred_class]
        conf  = proba.get(pred_class, 0)

        st.markdown(
            f'<div class="result-box" style="background:{color};">'
            f'Predicted Location: {label}'
            f'<br><span style="font-size:1.05rem;font-weight:400;opacity:0.92;">'
            f'Confidence: {conf*100:.1f}%</span></div>',
            unsafe_allow_html=True
        )

        st.info(f"**Clinical Interpretation:** {CLASS_DESC[pred_class]}")

        # Probability bars
        st.markdown("**Confidence Across All Classes**")
        for cls, prob in sorted(proba.items(), key=lambda x: -x[1]):
            bar_color = CLASS_COLORS[cls]
            pct = prob * 100
            marker = " ◀" if cls == pred_class else ""
            st.markdown(
                f"<div style='margin-bottom:9px;'>"
                f"<div style='display:flex;justify-content:space-between;margin-bottom:2px;'>"
                f"<span><b>{CLASS_LABELS[cls]}</b>{marker}</span>"
                f"<span><b>{pct:.1f}%</b></span></div>"
                f"<div style='background:#e0e0e0;border-radius:6px;height:16px;'>"
                f"<div style='background:{bar_color};width:{min(pct,100):.1f}%;height:16px;"
                f"border-radius:6px;transition:width 0.4s;'></div></div></div>",
                unsafe_allow_html=True
            )

        st.markdown("---")
        d1, d2 = st.columns(2)
        with d1:
            with st.expander("Pipeline Details"):
                st.markdown(f"**Stage 1 (Non-Limb Screen)**")
                st.markdown(f"- Non-limb probability: `{nl_prob*100:.1f}%`")
                st.markdown(f"- Decision: {'Non-limb ⚠️' if nl_prob > 0.5 else 'Limb ✓'}")
                st.markdown(f"**Stage 2 (Limb Region)**")
                for cls in ['distal','middle','proximal']:
                    if cls in proba:
                        st.markdown(f"- {CLASS_LABELS[cls]}: `{proba[cls]*100:.1f}%`")
        with d2:
            active_flags = [k.replace('flag_','').replace('_',' ').title()
                            for k, v in flag_feat.items() if v == 1]
            with st.expander(f"Anatomical Keywords Detected ({len(active_flags)})"):
                if active_flags:
                    cols = st.columns(2)
                    for i, f in enumerate(active_flags):
                        cols[i % 2].markdown(f"✓ {f}")
                else:
                    st.markdown("No specific anatomical keywords detected.  \n"
                                "Adding terms like *fetlock*, *stifle*, *navicular*, *sacroiliac* "
                                "can improve prediction accuracy.")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#888;font-size:0.82rem;'>"
        "Equine Lameness Location Predictor · MU Veterinary Health Center · University of Missouri<br>"
        "For research and clinical decision support use only. Not a substitute for veterinary examination."
        "</div>",
        unsafe_allow_html=True
    )
