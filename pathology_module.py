"""
pathology_module.py
────────────────────────────────────────────────────────────────────────────
Pathology Centre — a real pan/zoom virtual microscope department for the
AI MLS Simulation Hospital, in the same style as the other departments
(ultrasound, surgery, endoscopy, etc.): guest-mode, no account, scores saved
via shared_session.record_session(), leader-authored content gated behind
the same "mls-leader-2026" passphrase already used for Reasoning Canvas.

This is the Simulation Hospital's own copy of the microscope experience
(ported from the Virtual Hospital's medsim_pathology.py) — storage here is
local JSON + image files under slides/, matching how this app already
stores cases/ and hospital_data/, since this codebase has no Supabase
database wired in.

HONEST SCOPE NOTE (same as the Virtual Hospital version): this is a pan/
zoom viewer over a single uploaded slide image (a photo or cropped scan
region), not a true whole-slide-imaging (WSI) gigapixel tile pyramid —
that needs a dedicated tile server, which is heavier infrastructure than
a single image file can provide. For real microscope photos or reasonably
cropped scan regions, pan/zoom/annotation works genuinely.

Does not modify any existing department file. Purely additive:
  - new file: pathology_module.py (this file)
  - new dir:  slides/ (created on first use, like hospital_data/ and cases/)
  - 2 lines added to hospital_portal.py (lobby entry + route)
  - 1 line added to shared_session.py (DEPT_KEYS + 1 badge rule)
"""

import base64
import glob
import json
import os
import time

import streamlit as st
import streamlit.components.v1 as components

from shared_session import dept_header, record_session

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SLIDES_DIR = os.path.join(_SCRIPT_DIR, "slides")
os.makedirs(SLIDES_DIR, exist_ok=True)

LEADER_PASSPHRASE = "mls-leader-2026"
MAGNIFICATIONS = [4, 10, 40, 100]


# ── Local storage (mirrors the cases/ + hospital_data/ pattern already used
#    elsewhere in this app — a .json sidecar per slide, image kept as base64
#    inside that same json so there's only one file to move/back up) ────────
def _slide_path(slide_id: str) -> str:
    return os.path.join(SLIDES_DIR, f"{slide_id}.json")


def save_slide(row: dict):
    tmp = _slide_path(row["slide_id"]) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(row, f)
    os.replace(tmp, _slide_path(row["slide_id"]))


def list_slides() -> list:
    out = []
    for p in sorted(glob.glob(os.path.join(SLIDES_DIR, "*.json"))):
        try:
            with open(p) as f:
                out.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    return out


def get_slide(slide_id: str):
    p = _slide_path(slide_id)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def delete_slide(slide_id: str) -> bool:
    p = _slide_path(slide_id)
    if os.path.exists(p):
        os.remove(p)
        return True
    return False


# ── Placeholder slide (shown until a leader uploads a real one) ─────────────
def _placeholder_slide_svg_b64() -> str:
    import random
    random.seed(7)
    cells = []
    for _ in range(220):
        cx, cy = random.uniform(0, 1000), random.uniform(0, 1000)
        r = random.uniform(8, 16)
        cells.append(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{r:.0f}" '
                      f'fill="#d88a9a" fill-opacity="0.55" stroke="#7a2f42" stroke-width="1"/>')
        cells.append(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{r*0.4:.0f}" '
                      f'fill="#5b1a2e" fill-opacity="0.8"/>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">'
           f'<rect width="1000" height="1000" fill="#f3d9de"/>{"".join(cells)}'
           f'<text x="500" y="980" text-anchor="middle" font-size="22" fill="#7a2f42" '
           f'font-family="sans-serif">PLACEHOLDER not a real pathology image</text></svg>')
    return base64.b64encode(svg.encode()).decode()


def _build_microscope_html(image_b64: str, is_svg: bool, annotations: list, slide_name: str) -> str:
    mime = "image/svg+xml" if is_svg else "image/jpeg"
    img_src = f"data:{mime};base64,{image_b64}"
    ann_json = json.dumps(annotations or [])

    template = """<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;overflow:hidden;font-family:'Segoe UI',sans-serif;color:#e6edf3}
#stage{position:relative;width:100%;height:100%;overflow:hidden;cursor:grab;background:#000}
#stage.dragging{cursor:grabbing}
#slide{position:absolute;left:50%;top:50%;transform-origin:center center;user-select:none;
       pointer-events:none;}
#hud{position:absolute;top:10px;left:10px;background:rgba(0,0,0,.75);border:1px solid rgba(255,255,255,.15);
     border-radius:8px;padding:7px 12px;font-size:12px;pointer-events:none}
#mags{position:absolute;top:10px;right:10px;display:flex;gap:6px}
.magbtn{background:rgba(20,20,30,.85);border:1px solid rgba(255,255,255,.2);color:#e6edf3;
        border-radius:6px;padding:6px 12px;font-size:12px;cursor:pointer}
.magbtn.on{background:#ffd400;border-color:#ffd400;color:#1D1D20}
.marker{position:absolute;width:18px;height:18px;border-radius:50%;background:#ffd400;
        border:2px solid #7a2f42;transform:translate(-50%,-50%);cursor:pointer;
        box-shadow:0 0 8px rgba(255,212,0,.8);}
#note{position:absolute;bottom:10px;left:10px;right:10px;background:rgba(0,0,0,.85);
      border:1px solid rgba(255,212,0,.4);border-radius:8px;padding:10px 14px;font-size:13px;
      display:none;}
#hint{position:absolute;bottom:10px;left:50%;transform:translateX(-50%);font-size:11px;
      color:#9091a4;pointer-events:none;}
</style></head>
<body>
<div id="stage">
  <div id="slide"><img id="img" src="__IMG_SRC__" draggable="false"></div>
  <div id="hud">__NAME__ &mdash; <span id="magLabel">4x</span></div>
  <div id="mags">
    <button class="magbtn on" data-m="4">4x</button>
    <button class="magbtn" data-m="10">10x</button>
    <button class="magbtn" data-m="40">40x</button>
    <button class="magbtn" data-m="100">100x</button>
  </div>
  <div id="note"></div>
  <div id="hint">Drag to pan &middot; scroll or pinch to zoom &middot; click markers for findings</div>
</div>
<script>
const ANNOTATIONS = __ANN_JSON__;
const stage = document.getElementById('stage');
const slide = document.getElementById('slide');
const img = document.getElementById('img');

let scale = 1, panX = 0, panY = 0;
let dragging = false, lastX = 0, lastY = 0;
const MAG_SCALE = {4:1, 10:2.5, 40:8, 100:18};

function apply() {
  slide.style.transform = 'translate(-50%,-50%) translate(' + panX + 'px,' + panY + 'px) scale(' + scale + ')';
}

img.onload = () => { placeMarkers(); apply(); };

function placeMarkers() {
  ANNOTATIONS.forEach((a) => {
    const dot = document.createElement('div');
    dot.className = 'marker';
    dot.style.left = (a.x * 100) + '%';
    dot.style.top = (a.y * 100) + '%';
    dot.title = a.label || '';
    dot.onclick = (e) => {
      e.stopPropagation();
      const note = document.getElementById('note');
      note.style.display = 'block';
      note.innerHTML = '<b>' + (a.label || 'Finding') + '</b><br>' + (a.description || '');
    };
    slide.appendChild(dot);
  });
}

stage.addEventListener('mousedown', e => { dragging = true; lastX = e.clientX; lastY = e.clientY; stage.classList.add('dragging'); });
window.addEventListener('mouseup', () => { dragging = false; stage.classList.remove('dragging'); });
window.addEventListener('mousemove', e => {
  if (!dragging) return;
  panX += (e.clientX - lastX) / scale; panY += (e.clientY - lastY) / scale;
  lastX = e.clientX; lastY = e.clientY;
  apply();
});
stage.addEventListener('wheel', e => {
  e.preventDefault();
  const delta = e.deltaY < 0 ? 1.1 : 0.9;
  scale = Math.max(0.5, Math.min(24, scale * delta));
  apply();
}, { passive: false });

let lastDist = null;
stage.addEventListener('touchstart', e => {
  if (e.touches.length === 1) { dragging = true; lastX = e.touches[0].clientX; lastY = e.touches[0].clientY; }
  if (e.touches.length === 2) { lastDist = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY); }
});
stage.addEventListener('touchmove', e => {
  e.preventDefault();
  if (e.touches.length === 1 && dragging) {
    panX += (e.touches[0].clientX - lastX) / scale; panY += (e.touches[0].clientY - lastY) / scale;
    lastX = e.touches[0].clientX; lastY = e.touches[0].clientY;
    apply();
  } else if (e.touches.length === 2 && lastDist) {
    const dist = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY);
    scale = Math.max(0.5, Math.min(24, scale * (dist / lastDist)));
    lastDist = dist;
    apply();
  }
}, { passive: false });
stage.addEventListener('touchend', () => { dragging = false; lastDist = null; });

document.querySelectorAll('.magbtn').forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll('.magbtn').forEach(b => b.classList.remove('on'));
    btn.classList.add('on');
    const m = parseInt(btn.dataset.m);
    scale = MAG_SCALE[m];
    panX = 0; panY = 0;
    document.getElementById('magLabel').textContent = m + 'x';
    apply();
  };
});
</script>
</body></html>"""
    return (template.replace("__IMG_SRC__", img_src)
                     .replace("__NAME__", slide_name)
                     .replace("__ANN_JSON__", ann_json))


def render_microscope(slide_row, height: int = 540):
    if not slide_row:
        b64 = _placeholder_slide_svg_b64()
        html = _build_microscope_html(b64, True, [
            {"x": 0.3, "y": 0.4, "label": "Placeholder marker",
             "description": "A leader hasn't uploaded a real slide yet — this is a demo pattern."},
        ], "Demo pattern (not a real slide)")
    else:
        html = _build_microscope_html(slide_row["image_b64"], False,
                                       slide_row.get("annotations") or [], slide_row["name"])
    components.html(html, height=height, scrolling=False)


def _leader_upload_form():
    st.markdown("#### 📤 Leader: upload a real slide")
    st.caption("Keep files reasonably sized (a few MB) — stored as a local file, not a tile "
               "server, so very large whole-slide scans won't work here.")
    uploaded = st.file_uploader("Slide image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="path_upl")
    name = st.text_input("Slide name", placeholder="e.g. Invasive Ductal Carcinoma — H&E")
    specialty = st.selectbox("Specialty", ["Oncology", "Hematology", "Dermatopathology",
                                            "Microbiology", "General Pathology", "Other"])
    diagnosis = st.text_input("Correct diagnosis (used to grade the quiz below)")
    description = st.text_area("Teaching notes")

    st.markdown("##### Annotations (optional)")
    n_ann = st.number_input("Number of annotations", min_value=0, max_value=10, value=0)
    annotations = []
    for i in range(int(n_ann)):
        c1, c2, c3 = st.columns([1, 1, 2])
        x = c1.slider(f"X% #{i+1}", 0, 100, 50, key=f"path_ann_x_{i}") / 100
        y = c2.slider(f"Y% #{i+1}", 0, 100, 50, key=f"path_ann_y_{i}") / 100
        label = c3.text_input(f"Label #{i+1}", key=f"path_ann_l_{i}")
        desc = st.text_input(f"Finding description #{i+1}", key=f"path_ann_d_{i}")
        annotations.append({"x": x, "y": y, "label": label, "description": desc})

    st.markdown("##### Quiz distractors (wrong diagnosis options shown alongside the correct one)")
    distractors_raw = st.text_input("Comma-separated wrong options",
                                     placeholder="e.g. Fibroadenoma, Normal breast tissue, Lipoma")

    if st.button("💾 Save Slide", type="primary"):
        if not uploaded or not name.strip() or not diagnosis.strip():
            st.error("An image file, a slide name, and the correct diagnosis are required.")
        else:
            img_bytes = uploaded.read()
            if len(img_bytes) > 6 * 1024 * 1024:
                st.error("File too large (over 6MB) — compress or crop the image and try again.")
                return
            b64 = base64.b64encode(img_bytes).decode()
            slide_id = f"slide_{int(time.time()*1000)}"
            save_slide({
                "slide_id": slide_id, "name": name.strip(), "specialty": specialty,
                "diagnosis": diagnosis.strip(), "description": description.strip(),
                "image_b64": b64, "annotations": annotations,
                "distractors": [d.strip() for d in distractors_raw.split(",") if d.strip()],
            })
            st.success("✅ Slide saved — it's now in the slide picker for all students.")


def render(student_id: str):
    dept_header("Pathology Centre", "🔬", student_id)
    st.caption("A real virtual microscope — pan and zoom across magnification levels, "
               "find the marked findings, then call the diagnosis.")

    with st.expander("🩺 Leader: author / manage slides"):
        pw = st.text_input("Leader passphrase", type="password", key="path_leader_pw")
        if pw == LEADER_PASSPHRASE:
            _leader_upload_form()
            st.markdown("---")
            for s in list_slides():
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"**{s['name']}** — {s.get('specialty','')} · {s.get('diagnosis','')}")
                if c2.button("🗑 Delete", key=f"del_slide_{s['slide_id']}"):
                    delete_slide(s["slide_id"])
                    st.rerun()
        elif pw:
            st.error("Wrong passphrase.")

    slides = list_slides()
    if not slides:
        st.info("No real slides uploaded yet — exploring the demo pattern below so you can try "
                "the pan/zoom/marker mechanics right now. Ask your course leader to upload real "
                "slides (see the box above) to unlock graded cases.")
        render_microscope(None)
        return

    labels = {s["slide_id"]: f"{s['name']} ({s.get('specialty','')})" for s in slides}
    picked_id = st.selectbox("Select a slide", list(labels.keys()), format_func=lambda k: labels[k])
    row = get_slide(picked_id)
    if not row:
        st.error("Could not load that slide.")
        return

    render_microscope(row)

    st.markdown("---")
    st.markdown("##### 🧪 Call the diagnosis")
    options = list({row["diagnosis"], *row.get("distractors", [])})
    import random
    random.Random(row["slide_id"]).shuffle(options)
    if not options:
        st.caption("This slide has no quiz options set — explore it for practice only.")
        return
    answer = st.radio("What is this?", options, key=f"path_quiz_{row['slide_id']}")
    if st.button("Submit diagnosis", key=f"path_submit_{row['slide_id']}"):
        correct = answer == row["diagnosis"]
        score = 100.0 if correct else max(0.0, 100.0 / max(1, len(options)))
        if correct:
            st.success(f"✅ Correct — {row['diagnosis']}")
        else:
            st.error(f"❌ Not quite — correct diagnosis was **{row['diagnosis']}**")
        if row.get("description"):
            st.info(row["description"])
        record_session(student_id, "pathology", score, f"slide:{row['name']} answer:{answer}")
