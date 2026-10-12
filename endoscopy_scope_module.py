"""
endoscopy_scope_module.py
────────────────────────────────────────────────────────────────────────────
Endoscopy (Upper & Lower) — a 360°-navigable 3D OR room (via or_room_scene.py,
the same shared builder used by intubation_module.py) showing the scope's
path through the GI tract, plus a real-photo gallery leaders can populate
with their own endoscopy images, viewed through the same pan/zoom mechanism
already built and shipping in pathology_module.py.

This is a SEPARATE department from the existing endoscopy_module.py (the
ArUco-marker camera drills) — that file is untouched. This one is a viewing/
orientation experience: rotate freely around the scope's path through the
tract, step through landmarks, and look at real mucosa photos.

HONEST SCOPE NOTE — read this before telling students what this is:
  - The 3D "OR room + GI tract path" is real, freely navigable 3D (genuine
    orbit/zoom/pan via or_room_scene.py's OrbitControls) — not a static image.
    The tract itself is a schematic tube built from primitive shapes for
    orientation and landmark sequencing, not a licensed anatomical model —
    same boundary as foley_module.py and intubation_module.py.
  - "Real human mucosa" photos are NOT bundled with this file (no binary
    image data ships in the repo). What IS confirmed and wired in:
    HyperKvasir (Borgli et al. 2020, Simula Research Laboratory) is a
    real, published GI-endoscopy image dataset released under
    **CC BY 4.0** — free to use here with attribution, confirmed by
    reading the LICENSE file in its GitHub repo (github.com/simula/
    hyper-kvasir) directly. Six of its labeled classes line up with our
    landmarks: normal-z-line → Z-line, normal-pylorus → Pylorus,
    retroflex-stomach → Gastric body, ileum → Ileocecal valve,
    normal-cecum → Cecum, retroflex-rectum → Rectum. (Duodenal bulb and
    Sigmoid colon have no matching HyperKvasir class — those two stay
    placeholder until a leader supplies a photo.) Browse/preview images at
    https://osf.io/mh9sj (the dataset's own page) or
    https://datasets.simula.no/hyper-kvasir — copy a direct image link and
    paste it into the leader panel below ("Fetch from a URL"); the app
    downloads and stores it server-side. The earlier GastroVision dataset
    was checked too and dropped — its GitHub/OSF pages never stated a
    license, so nothing from it is used. Until a leader adds a photo for a
    given landmark, students see a clearly labeled placeholder pattern,
    never something presented as real tissue.

Purely additive: new file only, plus 2 lines in hospital_portal.py and
1 line in shared_session.py (same pattern as every other new department).

Local storage only (same as pathology_module.py — no Supabase table needed):
photos are saved as base64-encoded JSON files under endoscopy_photos/.
"""

import base64
import glob
import io
import json
import os
import time
import zipfile

import streamlit as st
import streamlit.components.v1 as components

from shared_session import dept_header, record_session
from or_room_scene import build_or_room_html

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PHOTOS_DIR = os.path.join(_SCRIPT_DIR, "endoscopy_photos")
os.makedirs(PHOTOS_DIR, exist_ok=True)

LEADER_PASSPHRASE = "mls-leader-2026"  # same passphrase used across the app

UPPER_LANDMARKS = [
    {"key": "zline", "title": "Z-line (GE junction)", "note": "Where esophageal squamous mucosa meets gastric columnar mucosa — the key landmark for Barrett's surveillance."},
    {"key": "stomach_body", "title": "Gastric body", "note": "Rugal folds, assess for gastritis, ulcers, polyps."},
    {"key": "pylorus", "title": "Pylorus", "note": "The gatekeeper to the duodenum — note patency and any deformity."},
    {"key": "duodenal_bulb", "title": "Duodenal bulb", "note": "Common site for peptic ulcers — inspect all walls."},
]
LOWER_LANDMARKS = [
    {"key": "rectum", "title": "Rectum (retroflexed view)", "note": "Retroflex to inspect the rectal valves and anorectal junction — easy to miss lesions here on the way in."},
    {"key": "sigmoid", "title": "Sigmoid colon", "note": "Common site for diverticula — note any inflammation."},
    {"key": "ileocecal_valve", "title": "Ileocecal valve", "note": "Marks the junction with the terminal ileum — confirms a complete exam to the cecum."},
    {"key": "cecum", "title": "Cecum", "note": "The appendiceal orifice landmark confirms the scope reached the end of the colon."},
]

UPPER_FINDINGS = [
    {"key": "barretts", "title": "Barrett's Esophagus", "scope_type": "upper",
     "note": "Columnar metaplasia replacing the distal esophageal squamous lining — salmon-colored mucosa extending above the GE junction. Premalignant; needs surveillance biopsies."},
    {"key": "esophagitis_a", "title": "Esophagitis — Grade A (LA classification)", "scope_type": "upper",
     "note": "One or more mucosal breaks ≤5mm, not extending between the tops of two mucosal folds — the mildest grade of reflux esophagitis."},
    {"key": "esophagitis_b_d", "title": "Esophagitis — Grade B–D (LA classification)", "scope_type": "upper",
     "note": "Grade B: breaks >5mm not bridging folds. Grades C–D: breaks that bridge folds, up to circumferential — higher-grade reflux esophagitis."},
]
LOWER_FINDINGS = [
    {"key": "polyp", "title": "Colonic Polyp", "scope_type": "lower",
     "note": "A raised or flat mucosal lesion — removed and sent for histology, since adenomatous polyps are precursors to colorectal cancer."},
]
ALL_FINDINGS = UPPER_FINDINGS + LOWER_FINDINGS
_FINDING_BY_KEY = {f["key"]: f for f in ALL_FINDINGS}

QUIZ = {
    "upper": [
        {"q": "Which landmark marks where esophageal and gastric mucosa meet?",
         "options": ["Pylorus", "Z-line", "Duodenal bulb"], "answer": 1,
         "explain": "The Z-line (squamocolumnar junction) is the key landmark for Barrett's esophagus surveillance."},
        {"q": "What confirms the scope has reached the duodenum?",
         "options": ["Seeing rugal folds", "Passing through the pylorus", "Seeing the z-line"], "answer": 1,
         "explain": "The pylorus is the gateway from stomach to duodenum."},
    ],
    "lower": [
        {"q": "What confirms a complete colonoscopy to the cecum?",
         "options": ["Seeing the sigmoid colon", "Identifying the ileocecal valve / appendiceal orifice", "Retroflexing in the rectum"], "answer": 1,
         "explain": "Reaching and identifying the ileocecal valve/appendiceal orifice is the standard marker of cecal intubation."},
        {"q": "Why retroflex in the rectum on withdrawal?",
         "options": ["It's not necessary", "To inspect areas behind rectal valves easy to miss on the way in", "To measure scope length"], "answer": 1,
         "explain": "Retroflexion lets you see the rectal mucosa just proximal to the anorectal junction, which faces away from the forward view on insertion."},
    ],
}


def _photo_path(photo_id: str) -> str:
    return os.path.join(PHOTOS_DIR, f"{photo_id}.json")


def save_photo(row: dict):
    tmp = _photo_path(row["photo_id"]) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(row, f)
    os.replace(tmp, _photo_path(row["photo_id"]))


def list_photos(scope_type: str = None) -> list:
    out = []
    for p in sorted(glob.glob(os.path.join(PHOTOS_DIR, "*.json"))):
        try:
            with open(p) as f:
                row = json.load(f)
                if scope_type is None or row.get("scope_type") == scope_type:
                    out.append(row)
        except (json.JSONDecodeError, OSError):
            pass
    return out


def delete_photo(photo_id: str) -> bool:
    p = _photo_path(photo_id)
    if os.path.exists(p):
        os.remove(p)
        return True
    return False


# Confirmed HyperKvasir (CC BY 4.0) class → our landmark mapping. Only the
# six that actually line up are listed — Duodenal bulb and Sigmoid colon
# have no matching HyperKvasir class and are left out on purpose rather
# than guessed.
HYPERKVASIR_CLASS_HINT = {
    "zline":          "normal-z-line",
    "stomach_body":   "retroflex-stomach",
    "pylorus":        "normal-pylorus",
    "rectum":         "retroflex-rectum",
    "ileocecal_valve": "ileum",
    "cecum":          "normal-cecum",
}


_ALL_LANDMARK_SCOPE = {l["key"]: "upper" for l in UPPER_LANDMARKS}
_ALL_LANDMARK_SCOPE.update({l["key"]: "lower" for l in LOWER_LANDMARKS})
_ALL_LANDMARK_TITLE = {l["key"]: l["title"] for l in UPPER_LANDMARKS + LOWER_LANDMARKS}

# Finding keys use a distinct namespace from landmark keys (e.g. "barretts" vs
# "zline") so they can share the same storage/frame-stepper machinery without
# ever colliding. A finding photo's `seq` ordering is how the "normal, then
# the abnormality, then back to normal" video-like feel is achieved — it's a
# property of how a leader numbers the files on import, not separate code.
_ALL_FINDING_SCOPE = {f["key"]: f["scope_type"] for f in ALL_FINDINGS}
_ALL_FINDING_TITLE = {f["key"]: f["title"] for f in ALL_FINDINGS}
# One combined map used by bulk ZIP import so a single ZIP can carry both
# landmark photos and finding-series photos together.
_ALL_KEYS_SCOPE = {**_ALL_LANDMARK_SCOPE, **_ALL_FINDING_SCOPE}
_ALL_KEYS_TITLE = {**_ALL_LANDMARK_TITLE, **_ALL_FINDING_TITLE}


import re as _re


def import_zip_bulk(zip_bytes: bytes) -> tuple:
    """Open a leader-supplied ZIP and save every entry whose filename (minus
    extension, lowercased) matches a landmark key (e.g. zline.jpg,
    pylorus.jpg, cecum.jpg) OR a finding key with an optional sequence
    suffix (e.g. barretts.jpg for a single photo, or barretts_01.jpg,
    barretts_02.jpg, barretts_03.jpg for a normal→abnormal→normal series —
    the leader orders frames by numbering the files; lower numbers play
    first). Returns (matched, skipped) lists of filenames."""
    matched, skipped = [], []
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile:
        return matched, ["(not a valid ZIP file)"]
    for info in zf.infolist():
        if info.is_dir():
            continue
        fname = os.path.basename(info.filename)
        stem, ext = os.path.splitext(fname)
        stem = stem.strip().lower()
        if ext.lower() not in (".jpg", ".jpeg", ".png"):
            skipped.append(fname)
            continue

        key, seq = stem, 0
        m = _re.match(r"^(.*)_(\d+)$", stem)
        if m and m.group(1) in _ALL_KEYS_SCOPE:
            key, seq = m.group(1), int(m.group(2))
        elif stem not in _ALL_KEYS_SCOPE:
            skipped.append(fname)
            continue

        if key not in _ALL_KEYS_SCOPE:
            skipped.append(fname)
            continue
        data = zf.read(info)
        if len(data) > 6 * 1024 * 1024:
            skipped.append(f"{fname} (over 6MB)")
            continue
        b64 = base64.b64encode(data).decode()
        photo_id = f"endo_{int(time.time()*1000)}_{key}_{seq:02d}"
        save_photo({
            "photo_id": photo_id, "scope_type": _ALL_KEYS_SCOPE[key], "landmark_key": key,
            "seq": seq,
            "name": _ALL_KEYS_TITLE[key],
            "diagnosis_note": "HyperKvasir dataset (Simula Research Laboratory), CC BY 4.0 — Borgli et al. 2020",
            "image_b64": b64,
        })
        matched.append(fname)
        time.sleep(0.001)
    return matched, skipped


def fetch_and_save_from_url(url: str, scope_type: str, landmark_key: str,
                             name: str, source_note: str = "") -> tuple:
    """Download one image from a URL the leader supplies (e.g. a HyperKvasir
    image link browsed on OSF) and store it the same way an uploaded file is
    stored. Returns (ok: bool, message: str). Network-dependent — only works
    where the app itself has outbound internet (e.g. Streamlit Cloud); this
    sandbox's own network is restricted, so this path is exercised by the
    deployed app, not here.
    """
    try:
        import requests
    except ImportError:
        return False, "The `requests` package isn't installed — add it to requirements.txt."
    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        return False, "That doesn't look like a URL."
    try:
        resp = requests.get(url, timeout=15, stream=True)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if not ctype.startswith("image/"):
            return False, f"URL did not return an image (Content-Type: {ctype or 'unknown'})."
        img_bytes = resp.content
        if len(img_bytes) > 6 * 1024 * 1024:
            return False, "Image too large (over 6MB)."
    except Exception as e:
        return False, f"Could not fetch that URL: {e}"
    b64 = base64.b64encode(img_bytes).decode()
    photo_id = f"endo_{int(time.time()*1000)}"
    save_photo({
        "photo_id": photo_id, "scope_type": scope_type, "landmark_key": landmark_key,
        "name": name.strip() or landmark_key,
        "diagnosis_note": (source_note or f"Source: {url}").strip(),
        "image_b64": b64,
    })
    return True, "Saved."


def _placeholder_svg_b64(label: str) -> str:
    import random
    random.seed(hash(label) % 1000)
    blobs = []
    for _ in range(60):
        cx, cy = random.uniform(0, 800), random.uniform(0, 600)
        r = random.uniform(20, 55)
        blobs.append(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{r:.0f}" '
                      f'fill="#d98a7a" fill-opacity="0.35" stroke="#a84a3a" stroke-width="1"/>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600">'
           f'<rect width="800" height="600" fill="#e8b4a8"/>{"".join(blobs)}'
           f'<text x="400" y="560" text-anchor="middle" font-size="20" fill="#7a2f1a" '
           f'font-family="sans-serif">PLACEHOLDER — not a real photo — {label}</text></svg>')
    return base64.b64encode(svg.encode()).decode()


def _build_photo_viewer_html(image_b64: str, is_svg: bool, label: str) -> str:
    mime = "image/svg+xml" if is_svg else "image/jpeg"
    img_src = f"data:{mime};base64,{image_b64}"
    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0d1117;overflow:hidden}}
#stage{{position:relative;width:100%;height:100%;overflow:hidden;cursor:grab;background:#000}}
#stage.dragging{{cursor:grabbing}}
#img{{position:absolute;left:50%;top:50%;transform-origin:center center;user-select:none;pointer-events:none;}}
#hud{{position:absolute;top:10px;left:10px;background:rgba(0,0,0,.75);color:#e6edf3;border-radius:8px;
     padding:7px 12px;font-size:12px;font-family:sans-serif;}}
#hint{{position:absolute;bottom:10px;left:50%;transform:translateX(-50%);font-size:11px;
      color:#9091a4;font-family:sans-serif;}}
</style></head>
<body>
<div id="stage">
  <img id="img" src="{img_src}" draggable="false">
  <div id="hud">{label}</div>
  <div id="hint">Drag to pan &middot; scroll/pinch to zoom</div>
</div>
<script>
const stage = document.getElementById('stage');
const img = document.getElementById('img');
let scale = 1, panX = 0, panY = 0, dragging = false, lastX = 0, lastY = 0;
function apply() {{ img.style.transform = 'translate(-50%,-50%) translate(' + panX + 'px,' + panY + 'px) scale(' + scale + ')'; }}
stage.addEventListener('mousedown', e => {{ dragging = true; lastX = e.clientX; lastY = e.clientY; stage.classList.add('dragging'); }});
window.addEventListener('mouseup', () => {{ dragging = false; stage.classList.remove('dragging'); }});
window.addEventListener('mousemove', e => {{
  if (!dragging) return;
  panX += (e.clientX - lastX) / scale; panY += (e.clientY - lastY) / scale;
  lastX = e.clientX; lastY = e.clientY; apply();
}});
stage.addEventListener('wheel', e => {{
  e.preventDefault();
  scale = Math.max(0.5, Math.min(10, scale * (e.deltaY < 0 ? 1.1 : 0.9)));
  apply();
}}, {{ passive: false }});
let lastDist = null;
stage.addEventListener('touchstart', e => {{
  if (e.touches.length === 1) {{ dragging = true; lastX = e.touches[0].clientX; lastY = e.touches[0].clientY; }}
  if (e.touches.length === 2) {{ lastDist = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY); }}
}});
stage.addEventListener('touchmove', e => {{
  e.preventDefault();
  if (e.touches.length === 1 && dragging) {{
    panX += (e.touches[0].clientX - lastX) / scale; panY += (e.touches[0].clientY - lastY) / scale;
    lastX = e.touches[0].clientX; lastY = e.touches[0].clientY; apply();
  }} else if (e.touches.length === 2 && lastDist) {{
    const dist = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY);
    scale = Math.max(0.5, Math.min(10, scale * (dist / lastDist))); lastDist = dist; apply();
  }}
}}, {{ passive: false }});
stage.addEventListener('touchend', () => {{ dragging = false; lastDist = null; }});
</script>
</body></html>"""


def _build_tract_js(scope_type: str, landmark_idx: int) -> str:
    """Schematic GI tract tube inside the shared OR room, with the scope
    advancing to the current landmark. Primitive shapes only."""
    is_upper = scope_type == "upper"
    n = 4
    t = landmark_idx / max(1, n - 1)
    return f"""
const patientDetail = new THREE.Group();

const torso = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.25, 1.3),
  new THREE.MeshStandardMaterial({{color:0xe6c3a8}}));
torso.position.set(0, -0.25, -0.1);
patientDetail.add(torso);

const pts = [];
for (let i=0;i<=24;i++){{
  const u = i/24;
  const x = Math.sin(u*Math.PI*1.5)*0.15;
  const y = -0.1 - u*0.1;
  const z = {str(0.6 if is_upper else -0.6)} {"- u*1.1" if is_upper else "+ u*1.1"};
  pts.push(new THREE.Vector3(x, y, z));
}}
const curve = new THREE.CatmullRomCurve3(pts);
const tract = new THREE.Mesh(new THREE.TubeGeometry(curve, 50, 0.1, 12, false),
  new THREE.MeshStandardMaterial({{color:{"0xd98a7a" if is_upper else "0xc97a5a"}, transparent:true, opacity:0.55}}));
patientDetail.add(tract);

const scopePos = curve.getPoint({t});
const scopeTip = new THREE.Mesh(new THREE.SphereGeometry(0.06, 16, 16),
  new THREE.MeshStandardMaterial({{color:0xffd400, emissive:0x554400}}));
scopeTip.position.copy(scopePos);
patientDetail.add(scopeTip);

const scopeBody = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.5, 12),
  new THREE.MeshStandardMaterial({{color:0x333333}}));
scopeBody.position.copy(curve.getPoint(Math.max(0, {t} - 0.08)));
scopeBody.rotation.x = Math.PI/2;
patientDetail.add(scopeBody);
"""


def _render_photo_frame_series(photos: list, frame_key: str, fallback_label: str):
    """Shared frame-stepper viewer: plays a sorted list of real photos like a
    short clip (◀ Frame / Frame ▶), used for both landmark walkthroughs and
    finding case series. Falls back to a labeled placeholder if empty."""
    if photos:
        fidx = st.session_state.get(frame_key, 0)
        fidx = max(0, min(fidx, len(photos) - 1))
        fcols = st.columns([1, 4, 1])
        with fcols[0]:
            if st.button("◀ Frame", disabled=fidx == 0, key=f"{frame_key}_prev"):
                st.session_state[frame_key] = max(0, fidx - 1)
                st.rerun()
        with fcols[2]:
            if st.button("Frame ▶", disabled=fidx == len(photos) - 1, key=f"{frame_key}_next"):
                st.session_state[frame_key] = min(len(photos) - 1, fidx + 1)
                st.rerun()
        with fcols[1]:
            st.progress((fidx + 1) / len(photos), text=f"Frame {fidx + 1} of {len(photos)} — real photo series")
        row = photos[fidx]
        if row.get("diagnosis_note"):
            st.caption(row["diagnosis_note"])
        html = _build_photo_viewer_html(row["image_b64"], False, row.get("name", fallback_label))
        components.html(html, height=460, scrolling=False)
    else:
        st.info("No real photo uploaded yet — showing a labeled placeholder.")
        b64 = _placeholder_svg_b64(fallback_label)
        html = _build_photo_viewer_html(b64, True, "Placeholder — not real tissue")
        components.html(html, height=460, scrolling=False)


def _render_findings(student_id: str):
    st.caption(
        "Each case plays as a short real-photo series: normal tissue first, then the "
        "abnormal finding, then back to normal — the way it looks scrolling through an "
        "actual procedure. Order is set by the leader when photos are uploaded (frame 1, "
        "frame 2, …)."
    )
    gi_type = st.radio("GI tract", ["Upper GI", "Lower GI"], horizontal=True, key="endo_finding_gi")
    findings = UPPER_FINDINGS if gi_type == "Upper GI" else LOWER_FINDINGS
    if not findings:
        st.info("No case findings added for this tract yet.")
        return

    titles = [f["title"] for f in findings]
    sel_title = st.selectbox("Case", titles, key="endo_finding_sel")
    finding = next(f for f in findings if f["title"] == sel_title)

    photos = sorted(
        [p for p in list_photos() if p.get("landmark_key") == finding["key"]],
        key=lambda p: (p.get("seq", 0), p.get("photo_id", "")),
    )
    _render_photo_frame_series(photos, f"endo_finding_frame_{finding['key']}", finding["title"])
    st.markdown(f"**{finding['title']}** — {finding['note']}")
    st.markdown("---")

    with st.expander("🔬 Leader: bulk-import a ZIP for this case (whole series at once)"):
        pw_bulk = st.text_input("Leader passphrase", type="password", key=f"endo_finding_pw_{finding['key']}")
        if pw_bulk == LEADER_PASSPHRASE:
            st.caption(
                f"Name files `{finding['key']}_01.jpg`, `{finding['key']}_02.jpg`, "
                f"`{finding['key']}_03.jpg` … in the order you want them to play "
                f"(e.g. 01–02 normal, 03–04 the abnormality, 05 back to normal). "
                f"Pack the whole case into one ZIP — the series updates in one step."
            )
            zfile = st.file_uploader("ZIP of photos for this case", type=["zip"], key=f"endo_finding_zip_{finding['key']}")
            if st.button("📦 Import ZIP", key=f"endo_finding_zip_import_{finding['key']}"):
                if not zfile:
                    st.error("Choose a ZIP file first.")
                else:
                    matched, skipped = import_zip_bulk(zfile.read())
                    if matched:
                        st.success(f"✅ Saved: {', '.join(matched)}")
                    if skipped:
                        st.warning(f"Skipped (name didn't match a case/landmark key, or not an image): {', '.join(skipped)}")
                    if matched:
                        st.rerun()
        elif pw_bulk:
            st.error("Wrong passphrase.")


def render(student_id: str):
    dept_header("Endoscopy (Upper & Lower)", "🔭", student_id)
    st.info("🔒 **Teaching simulation — follow local protocol.** Full 3D room: orbit, zoom "
            "and pan to follow the scope's path. Real photos only appear once your course "
            "leader uploads them — see the note below.")
    with st.expander("⚠️ What this is and isn't"):
        st.markdown(
            "- **Is:** a real, freely navigable 3D OR room following the scope's path through "
            "the tract, landmark-by-landmark, with a quiz, plus a real-photo viewer your "
            "leader can populate.\n"
            "- **Isn't:** a substitute for hands-on scope practice. The tract shown in 3D is "
            "schematic, not a licensed anatomical model. No stock or research-dataset mucosa "
            "photos are bundled — I checked a real published dataset (GastroVision) and "
            "couldn't confirm its license permits redistribution here, so nothing from it "
            "is included. Photos only appear if your leader uploads images they hold rights to."
        )

    section = st.radio("Section", ["🧭 Anatomy Walkthrough", "🔬 Case Findings"],
                        horizontal=True, key="endo_section")

    if section.startswith("🔬"):
        _render_findings(student_id)
        return

    scope_type = st.radio("Exam type", ["Upper GI (EGD)", "Lower GI (Colonoscopy)"],
                           horizontal=True, key="endo_scope_type")
    key = "upper" if scope_type.startswith("Upper") else "lower"
    landmarks = UPPER_LANDMARKS if key == "upper" else LOWER_LANDMARKS

    idx = st.session_state.get(f"endo_{key}_idx", 0)
    cols = st.columns([1, 4, 1])
    with cols[0]:
        if st.button("⬅ Prev", disabled=idx == 0, key=f"endo_prev_{key}"):
            st.session_state[f"endo_{key}_idx"] = max(0, idx - 1)
            st.rerun()
    with cols[2]:
        if st.button("Next ➡", disabled=idx == len(landmarks) - 1, key=f"endo_next_{key}"):
            st.session_state[f"endo_{key}_idx"] = min(len(landmarks) - 1, idx + 1)
            st.rerun()
    lm = landmarks[idx]
    with cols[1]:
        st.progress((idx + 1) / len(landmarks), text=f"Landmark {idx+1} of {len(landmarks)}: {lm['title']}")

    view_mode = st.radio("View", ["🏥 3D Room (orbit/zoom/pan)", "📷 Real photo (leader-uploaded)"],
                          horizontal=True, key=f"endo_view_{key}")

    if view_mode.startswith("🏥"):
        html = build_or_room_html(_build_tract_js(key, idx), f"{scope_type} — {lm['title']}")
        components.html(html, height=500, scrolling=False)
    else:
        photos = sorted(
            [p for p in list_photos(key) if p.get("landmark_key") == lm["key"]],
            key=lambda p: (p.get("seq", 0), p.get("photo_id", "")),
        )
        if photos:
            frame_key = f"endo_frame_{key}_{lm['key']}"
            fidx = st.session_state.get(frame_key, 0)
            fidx = max(0, min(fidx, len(photos) - 1))
            fcols = st.columns([1, 4, 1])
            with fcols[0]:
                if st.button("◀ Frame", disabled=fidx == 0, key=f"{frame_key}_prev"):
                    st.session_state[frame_key] = max(0, fidx - 1)
                    st.rerun()
            with fcols[2]:
                if st.button("Frame ▶", disabled=fidx == len(photos) - 1, key=f"{frame_key}_next"):
                    st.session_state[frame_key] = min(len(photos) - 1, fidx + 1)
                    st.rerun()
            with fcols[1]:
                st.progress((fidx + 1) / len(photos), text=f"Frame {fidx + 1} of {len(photos)} — real photo series")
            row = photos[fidx]
            if row.get("diagnosis_note"):
                st.caption(row["diagnosis_note"])
            html = _build_photo_viewer_html(row["image_b64"], False, row.get("name", lm["title"]))
            components.html(html, height=460, scrolling=False)
        else:
            st.info("No real photo uploaded yet for this landmark — showing a labeled placeholder.")
            b64 = _placeholder_svg_b64(lm["title"])
            html = _build_photo_viewer_html(b64, True, "Placeholder — not real tissue")
            components.html(html, height=460, scrolling=False)

    st.markdown(f"**{lm['title']}** — {lm['note']}")
    st.markdown("---")

    with st.expander("📦 Leader: bulk-import a ZIP (whole series at once)"):
        pw_bulk = st.text_input("Leader passphrase", type="password", key="endo_leader_pw_bulk")
        if pw_bulk == LEADER_PASSPHRASE:
            have = {p.get("landmark_key") for p in list_photos()}
            all_lm = UPPER_LANDMARKS + LOWER_LANDMARKS
            st.caption(
                "Keys needed (filename, any image ext): " + ", ".join(
                    f"✅ {l['key']}" if l["key"] in have else f"⬜ {l['key']}" for l in all_lm
                )
            )
            st.markdown(
                "Name each file after its landmark key — `zline.jpg`, `stomach_body.jpg`, "
                "`pylorus.jpg`, `rectum.jpg`, `ileocecal_valve.jpg`, `cecum.jpg` are matched "
                "automatically to HyperKvasir's classes (CC BY 4.0); `duodenal_bulb` / `sigmoid` "
                "need your own image since HyperKvasir has no matching class. Pack them all into "
                "one ZIP and the whole series updates in one step."
            )
            zfile = st.file_uploader("ZIP of photos", type=["zip"], key="endo_zip")
            if st.button("📦 Import ZIP", key="endo_zip_import"):
                if not zfile:
                    st.error("Choose a ZIP file first.")
                else:
                    matched, skipped = import_zip_bulk(zfile.read())
                    if matched:
                        st.success(f"✅ Saved: {', '.join(matched)}")
                    if skipped:
                        st.warning(f"Skipped (name didn't match a landmark key, or not an image): {', '.join(skipped)}")
                    if matched:
                        st.rerun()
        elif pw_bulk:
            st.error("Wrong passphrase.")

    with st.expander("🩺 Leader: add a real photo for this landmark"):
        pw = st.text_input("Leader passphrase", type="password", key=f"endo_leader_pw_{key}")
        if pw == LEADER_PASSPHRASE:
            hint = HYPERKVASIR_CLASS_HINT.get(lm["key"])
            if hint:
                st.caption(
                    f"💡 HyperKvasir (CC BY 4.0) class for this landmark: **{hint}**. "
                    f"Browse it at https://osf.io/mh9sj or https://datasets.simula.no/hyper-kvasir, "
                    f"copy a direct image link, and paste it below."
                )
            else:
                st.caption("No matching HyperKvasir class for this landmark — upload your own image instead.")

            src_mode = st.radio("Source", ["Upload a file", "Fetch from a URL"],
                                 horizontal=True, key=f"endo_src_{key}")
            name = st.text_input("Label", value=lm["title"], key=f"endo_upl_name_{key}")

            if src_mode == "Upload a file":
                uploaded = st.file_uploader("Photo (JPG/PNG) — must be an image you hold rights to",
                                             type=["jpg", "jpeg", "png"], key=f"endo_upl_{key}")
                diagnosis_note = st.text_input("Teaching note (optional)", key=f"endo_upl_note_{key}")
                if st.button("💾 Save photo", key=f"endo_upl_save_{key}"):
                    if not uploaded:
                        st.error("Choose an image file first.")
                    else:
                        img_bytes = uploaded.read()
                        if len(img_bytes) > 6 * 1024 * 1024:
                            st.error("File too large (over 6MB).")
                        else:
                            b64 = base64.b64encode(img_bytes).decode()
                            photo_id = f"endo_{int(time.time()*1000)}"
                            save_photo({
                                "photo_id": photo_id, "scope_type": key, "landmark_key": lm["key"],
                                "name": name.strip(), "diagnosis_note": diagnosis_note.strip(),
                                "image_b64": b64,
                            })
                            st.success("✅ Saved — switch to the Real photo view to see it.")
                            st.rerun()
            else:
                url = st.text_input("Direct image URL", key=f"endo_url_{key}",
                                     placeholder="https://.../normal-z-line/xxxxx.jpg")
                source_note = st.text_input(
                    "Attribution / teaching note",
                    value=("HyperKvasir dataset (Simula Research Laboratory), CC BY 4.0 — Borgli et al. 2020"
                           if hint else ""),
                    key=f"endo_url_note_{key}")
                if st.button("⬇️ Fetch & save", key=f"endo_url_save_{key}"):
                    ok, msg = fetch_and_save_from_url(url, key, lm["key"], name, source_note)
                    if ok:
                        st.success("✅ Saved — switch to the Real photo view to see it.")
                        st.rerun()
                    else:
                        st.error(msg)
        elif pw:
            st.error("Wrong passphrase.")

    if idx == len(landmarks) - 1:
        st.markdown("### 🧪 Landmark Quiz")
        answers = []
        for i, q in enumerate(QUIZ[key]):
            answers.append(st.radio(q["q"], q["options"], key=f"endo_q_{key}_{i}", index=None))
        if st.button("Submit Quiz", type="primary", key=f"endo_submit_{key}"):
            score_count = 0
            for i, q in enumerate(QUIZ[key]):
                picked = answers[i]
                if picked is not None and q["options"].index(picked) == q["answer"]:
                    score_count += 1
                    st.success(f"✅ {q['q']} — {q['explain']}")
                else:
                    st.error(f"❌ {q['q']} — {q['explain']}")
            pct = round(100 * score_count / len(QUIZ[key]), 1)
            st.metric("Quiz score", f"{pct}%")
            record_session(student_id, "endoscopy_scope", pct, f"{key} GI tract")
