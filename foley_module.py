"""
foley_module.py
────────────────────────────────────────────────────────────────────────────
Foley Catheterization department for the AI MLS Simulation Hospital — a
gesture/mouse-controlled 3D procedure walkthrough (9 steps), schematic
cutaway, and safety quiz, scored through the same shared_session.
record_session() used by every other department here.

This is the Simulation Hospital's own copy of the trainer also delivered to
the Virtual Hospital as medsim_foley.py — same mechanics, wired to this
app's local scoring/badge system instead of Supabase, per the uploaded
Foley_Training_App_Spec.md.

HONEST SCOPE NOTE (same boundary as medsim_foley.py and medsim_cpr.py's
depth tier): this gives real anatomy orientation, correct procedural
sequence, and decision-point quizzing. It does NOT simulate touch. A webcam
sees hand position in open air — it cannot sense contact, resistance, or a
catheter entering a lumen. Real tactile feedback needs a physical phantom
with embedded sensors (see FoleyHardwareBridge stub below, matching the
spec's own §8) — not built or tested here, and never claimed as working.

Purely additive: new file only, plus 2 lines in hospital_portal.py and
1 line in shared_session.py (same pattern as pathology_module.py).
"""

import streamlit as st
import streamlit.components.v1 as components

from shared_session import dept_header, record_session

STEPS = [
    {"title": "1. Prepare", "note": "Gather sterile kit, correct catheter size, gloves. Explain the procedure, confirm consent/identity.",
     "why": "Preventing CAUTI (catheter-associated UTI) starts before you open the kit."},
    {"title": "2. Position", "note": "Supine, legs abducted (female) or supine (male). Drape for privacy and a sterile field.",
     "why": "Correct positioning gives a clear, stable line of sight to the meatus."},
    {"title": "3. Sterile setup", "note": "Open the kit using no-touch technique. Don sterile gloves.",
     "why": "A break in sterile technique here is the single biggest CAUTI risk factor."},
    {"title": "4. Clean the meatus", "note": "Antiseptic swabs, one pass each, outward from the meatus, non-dominant hand retracts/separates.",
     "why": "Reduces bacterial load being pushed into the urethra on insertion."},
    {"title": "5. Lubricate", "note": "Apply sterile lubricant (with anesthetic gel for male patients) to the catheter tip / urethra.",
     "why": "Minimizes mucosal trauma and patient discomfort."},
    {"title": "6. Insert gently", "note": "Advance steadily until urine return — never force against resistance.",
     "why": "Forcing against resistance is how urethral injury happens."},
    {"title": "7. Confirm urine return", "note": "Watch for flow into the tubing/bag before advancing further or inflating.",
     "why": "Urine return confirms the tip is in the bladder, not just the urethra."},
    {"title": "8. Inflate the balloon", "note": "Only after confirmed urine return, inflate with the manufacturer-specified volume of sterile water.",
     "why": "Inflating before bladder placement is a classic cause of urethral injury."},
    {"title": "9. Secure & document", "note": "Secure tubing to the leg, bag below bladder level, document time/size/output/tolerance.",
     "why": "Securing prevents traction injury; documentation drives CAUTI-prevention review."},
]

QUIZ = [
    {"q": "You feel firm resistance halfway through insertion. What do you do?",
     "options": ["Push a little harder, it usually passes", "Stop, do not force it, reassess/escalate", "Inflate the balloon early to anchor it"],
     "answer": 1,
     "explain": "Never force against resistance — that's how urethral trauma happens. Stop and reassess."},
    {"q": "When is it safe to inflate the retention balloon?",
     "options": ["As soon as the catheter is fully inserted", "Only after confirmed urine return", "Before insertion, to guide it"],
     "answer": 1,
     "explain": "Inflating before confirmed bladder placement risks inflating within the urethra."},
    {"q": "Why clean the meatus outward, away from it, rather than side to side?",
     "options": ["It's faster", "It avoids dragging bacteria back toward the meatus", "It doesn't matter clinically"],
     "answer": 1,
     "explain": "A single outward pass per swab avoids reintroducing bacteria into the field you just cleaned."},
]


def _build_foley_html(sex: str, step_idx: int) -> str:
    female = sex == "Female"
    step = STEPS[step_idx]
    template = """<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#1D1D20;overflow:hidden;font-family:'Segoe UI',sans-serif;color:#fbfbff}
#c{width:100%;height:100%;display:block}
#hud{position:absolute;top:10px;left:10px;background:rgba(0,0,0,.75);border:1px solid rgba(255,255,255,.15);
     border-radius:8px;padding:8px 14px;font-size:13px;max-width:320px}
#hint{position:absolute;bottom:10px;left:50%;transform:translateX(-50%);font-size:11px;color:#909094}
#toggle{position:absolute;bottom:10px;right:10px;background:rgba(20,20,30,.85);border:1px solid rgba(255,255,255,.2);
        color:#fbfbff;border-radius:6px;padding:6px 12px;font-size:12px;cursor:pointer}
video#gesturebox{position:absolute;top:10px;right:10px;width:160px;border-radius:8px;
            border:1px solid rgba(255,255,255,.2);display:none}
</style></head>
<body>
<canvas id="c"></canvas>
<div id="hud"><b>__STEP_TITLE__</b></div>
<video id="gesturebox" autoplay muted playsinline></video>
<button id="toggle">🖐 Enable gesture camera (optional)</button>
<div id="hint">Drag to rotate &middot; scroll/pinch to zoom &middot; or enable gestures above</div>
<script type="importmap">
{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({canvas, antialias:true});
renderer.setSize(window.innerWidth, window.innerHeight);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1D1D20);
const camera = new THREE.PerspectiveCamera(50, window.innerWidth/window.innerHeight, 0.1, 100);
camera.position.set(0, 0, 6);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const dl = new THREE.DirectionalLight(0xffffff, 0.8); dl.position.set(3,4,5); scene.add(dl);

// Schematic pelvic cutaway — primitive shapes only, for orientation.
// NOT a licensed anatomical model (see file docstring / instructions tab).
const group = new THREE.Group();
const bladder = new THREE.Mesh(new THREE.SphereGeometry(1.1, 32, 32),
  new THREE.MeshStandardMaterial({color:0xd98a9a, transparent:true, opacity:0.55}));
bladder.position.set(0, 1.4, 0);
group.add(bladder);

const urethraLen = __FEMALE__ ? 1.4 : 3.2;
const urethraCurve = __FEMALE__ ? 0.1 : 0.9;
const pts = [];
for (let i=0;i<=20;i++){
  const t = i/20;
  pts.push(new THREE.Vector3(Math.sin(t*Math.PI)*urethraCurve*0.3, 1.4 - t*urethraLen, 0.2));
}
const curve = new THREE.CatmullRomCurve3(pts);
const tube = new THREE.Mesh(new THREE.TubeGeometry(curve, 40, 0.14, 12, false),
  new THREE.MeshStandardMaterial({color:0xf0b7c4, transparent:true, opacity:0.7}));
group.add(tube);

const catheter = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 1, 16),
  new THREE.MeshStandardMaterial({color:0xffd400}));
const stepIdx = __STEP_IDX__;
const advance = Math.min(1, Math.max(0, (stepIdx - 4) / 3));
const catPos = curve.getPoint(1 - advance*0.85);
catheter.position.copy(catPos);
catheter.visible = stepIdx >= 5;
group.add(catheter);

if (stepIdx >= 7) {
  const balloon = new THREE.Mesh(new THREE.SphereGeometry(0.22, 16, 16),
    new THREE.MeshStandardMaterial({color:0xffffff}));
  balloon.position.copy(curve.getPoint(0.08));
  group.add(balloon);
}

scene.add(group);

function animate(){
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

document.getElementById('toggle').onclick = async () => {
  const video = document.getElementById('gesturebox');
  video.style.display = 'block';
  try {
    const stream = await navigator.mediaDevices.getUserMedia({video:true});
    video.srcObject = stream;
    const { HandLandmarker, FilesetResolver } = await import(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm");
    const files = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
    const landmarker = await HandLandmarker.createFromOptions(files, {
      baseOptions: {modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"},
      numHands: 1, runningMode: "VIDEO",
    });
    let lastDist = null;
    function loop(ts) {
      if (video.readyState >= 2) {
        const res = landmarker.detectForVideo(video, ts);
        if (res.landmarks && res.landmarks[0]) {
          const lm = res.landmarks[0];
          const thumb = lm[4], index = lm[8];
          const dist = Math.hypot(thumb.x-index.x, thumb.y-index.y);
          if (lastDist !== null) {
            const delta = dist - lastDist;
            camera.position.z = Math.max(2, Math.min(12, camera.position.z - delta*20));
          }
          lastDist = dist;
        }
      }
      requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
  } catch (e) {
    alert("Camera unavailable: " + e.message);
  }
};
</script>
</body></html>"""
    return (template.replace("__STEP_TITLE__", step["title"])
                     .replace("__FEMALE__", "true" if female else "false")
                     .replace("__STEP_IDX__", str(step_idx)))


def render(student_id: str):
    dept_header("Foley Catheterization", "🩺", student_id)
    st.info("🔒 **Teaching simulation — follow local protocol.** Gives sequence, anatomy "
            "orientation and decision-point quizzing. Does not simulate touch — see below.")
    with st.expander("⚠️ What this is and isn't"):
        st.markdown(
            "- **Is:** a gesture/mouse-controlled 3D walkthrough of the 9-step procedure, a "
            "schematic (not licensed-anatomy) cutaway showing catheter advance and balloon "
            "timing, and a safety-reasoning quiz.\n"
            "- **Isn't:** a substitute for hands-on phantom practice. A webcam cannot sense "
            "contact or resistance — real touch feedback needs a physical phantom with "
            "embedded sensors (see the hardware note in Instructions), which is scoped but "
            "not built or tested here."
        )

    sex = st.radio("Case", ["Female", "Male"], horizontal=True, key="foley_sex")
    step_idx = st.session_state.get("foley_step", 0)

    cols = st.columns([1, 4, 1])
    with cols[0]:
        if st.button("⬅ Prev", disabled=step_idx == 0, key="foley_prev"):
            st.session_state.foley_step = max(0, step_idx - 1)
            st.rerun()
    with cols[2]:
        if st.button("Next ➡", disabled=step_idx == len(STEPS) - 1, key="foley_next"):
            st.session_state.foley_step = min(len(STEPS) - 1, step_idx + 1)
            st.rerun()

    step = STEPS[step_idx]
    with cols[1]:
        st.progress((step_idx + 1) / len(STEPS), text=f"Step {step_idx+1} of {len(STEPS)}: {step['title']}")

    components.html(_build_foley_html(sex, step_idx), height=480, scrolling=False)

    st.markdown(f"**{step['title']}** — {step['note']}")
    st.caption(f"Why it matters: {step['why']}")
    st.markdown("---")

    if step_idx == len(STEPS) - 1:
        st.markdown("### 🧪 Safety Quiz")
        answers = []
        for i, q in enumerate(QUIZ):
            answers.append(st.radio(q["q"], q["options"], key=f"foley_q_{i}", index=None))
        if st.button("Submit Quiz", type="primary", key="foley_submit"):
            score_count = 0
            for i, q in enumerate(QUIZ):
                picked = answers[i]
                if picked is not None and q["options"].index(picked) == q["answer"]:
                    score_count += 1
                    st.success(f"✅ {q['q']} — {q['explain']}")
                else:
                    st.error(f"❌ {q['q']} — {q['explain']}")
            pct = round(100 * score_count / len(QUIZ), 1)
            st.metric("Quiz score", f"{pct}%")
            record_session(student_id, "foley", pct, f"case:{sex}")


# ── Optional hardware bridge — scoped per Foley_Training_App_Spec.md §8,
# NOT built/tested. Kept only as an honest landing point for a future
# ESP32-driven phantom (IR break-beam at meatus/bladder + flow sensor). ─────
class FoleyHardwareBridge:
    """STUB. Would receive WebSocket events from a physical phantom and
    advance the on-screen catheter/urine-return state to match real
    insertion. Never tested against real hardware."""
    pass
