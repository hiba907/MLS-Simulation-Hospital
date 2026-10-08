"""
intubation_module.py
────────────────────────────────────────────────────────────────────────────
Endotracheal Intubation department — a gesture/mouse-controlled 3D OR room
(via or_room_scene.py) showing laryngoscope + ET tube placement through the
airway, with a "Room view" / "Close-up" camera toggle inspired by the layout
of the uploaded design mockup (close-up pane + schematic cutaway side by
side), a 7-step guided sequence, and a safety quiz — scored through this
app's shared_session.record_session(), same as every other department.

HONEST SCOPE NOTE (same boundary as foley_module.py): the OR room shell and
camera rig are real, freely navigable 3D — genuine orbit/zoom/pan, not a
static image. The airway anatomy (mouth, vocal cords, trachea, esophagus)
is a schematic built from primitive shapes for orientation and correct
sequencing, not a licensed anatomical model. Swap in a real .glb (see
or_room_scene.build_or_room_html's extra_glb_url param) once you've picked
a licensed source — the loader hook is already wired, so that's a one-line
change here, not a rebuild.

Purely additive: new file only, plus 2 lines in hospital_portal.py and
1 line in shared_session.py (same pattern as pathology_module.py /
foley_module.py).
"""

import streamlit as st
import streamlit.components.v1 as components

from shared_session import dept_header, record_session
from or_room_scene import build_or_room_html

STEPS = [
    {"title": "1. Pre-oxygenate", "note": "100% O2 by mask for 3-5 min before attempting. Position: sniffing position, ear-to-sternal-notch alignment.",
     "why": "Buys apnea time once paralytic/sedation is given — the single biggest safety margin in the whole procedure."},
    {"title": "2. Prepare equipment", "note": "Check laryngoscope light, cuff integrity on the ET tube, suction on and within reach, confirm tube size.",
     "why": "A dead laryngoscope bulb or a leaking cuff discovered mid-procedure costs seconds you don't have."},
    {"title": "3. Induce / sedate", "note": "Give induction agent and paralytic per protocol; bag-mask ventilate if needed while onset occurs.",
     "why": "Vocal cords must be relaxed, or the tube will not pass safely."},
    {"title": "4. Laryngoscopy", "note": "Insert blade along the right side of the tongue, sweep it left, lift (don't pry) to expose the vocal cords.",
     "why": "Prying on the upper teeth as a fulcrum is how dental injury happens — lift along the blade's axis instead."},
    {"title": "5. Visualize cords & pass tube", "note": "Confirm vocal cord visualization, pass the ET tube through the cords under direct vision.",
     "why": "Passing blind, without seeing the cords, risks esophageal intubation."},
    {"title": "6. Inflate cuff & confirm placement", "note": "Inflate the cuff, confirm with capnography (EtCO2 waveform), bilateral breath sounds, chest rise.",
     "why": "Capnography is the single most reliable confirmation — esophageal placement gives no sustained EtCO2 waveform."},
    {"title": "7. Secure & document", "note": "Note tube depth at the teeth/lips, secure the tube, get a confirmatory chest X-ray, document.",
     "why": "An unsecured tube can migrate into a mainstem bronchus or out of the trachea entirely with head movement."},
]

QUIZ = [
    {"q": "After passing the tube, how do you MOST reliably confirm tracheal (not esophageal) placement?",
     "options": ["Listening for breath sounds alone", "Capnography showing a sustained EtCO2 waveform", "Watching the chest rise once"],
     "answer": 1,
     "explain": "Breath sounds and chest rise can be misleading; a sustained EtCO2 waveform is the gold-standard confirmation."},
    {"q": "You're having trouble visualizing the cords and are levering the blade against the upper teeth. What's wrong?",
     "options": ["Nothing, this is normal technique", "You should lift along the blade's axis, not pry on the teeth", "You should push the blade in further"],
     "answer": 1,
     "explain": "Using the teeth as a fulcrum is a common cause of dental injury — lift, don't pry."},
    {"q": "Why pre-oxygenate for 3–5 minutes before attempting?",
     "options": ["It's just routine paperwork", "It extends the safe apnea time once sedation/paralysis is given", "It makes the tube easier to see"],
     "answer": 1,
     "explain": "Pre-oxygenation extends the window before desaturation once the patient stops breathing on their own."},
]


def _build_patient_js(step_idx: int) -> str:
    """Schematic head/neck + airway cutaway, placed on the bed inside the
    shared OR room. Primitive shapes only — see module docstring."""
    cords_open = step_idx >= 3   # visualized from laryngoscopy step onward
    tube_in = step_idx >= 4      # tube visible from step 5 onward
    cuff_inflated = step_idx >= 5
    return f"""
const patientDetail = new THREE.Group();

// Torso (simple capsule-like box) lying on the bed
const torso = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.25, 1.3),
  new THREE.MeshStandardMaterial({{color:0xe6c3a8}}));
torso.position.set(0, -0.25, -0.1);
patientDetail.add(torso);

// Head, tilted back (sniffing position)
const head = new THREE.Mesh(new THREE.SphereGeometry(0.22, 24, 24),
  new THREE.MeshStandardMaterial({{color:0xe6c3a8}}));
head.position.set(0, -0.05, 0.55);
head.rotation.x = -0.35;
patientDetail.add(head);

// Airway tube (mouth -> pharynx -> trachea), curved
const pts = [];
for (let i=0;i<=16;i++){{
  const t = i/16;
  pts.push(new THREE.Vector3(0, -0.05 + t*0.05, 0.75 - t*0.9));
}}
const curve = new THREE.CatmullRomCurve3(pts);
const airway = new THREE.Mesh(new THREE.TubeGeometry(curve, 30, 0.07, 10, false),
  new THREE.MeshStandardMaterial({{color:0xcc6677, transparent:true, opacity:0.5}}));
patientDetail.add(airway);

// Esophagus (adjacent tube, behind the trachea) — labeled distractor for
// confirming correct vs incorrect placement
const eCurve = new THREE.CatmullRomCurve3(pts.map(p => new THREE.Vector3(p.x, p.y - 0.09, p.z)));
const esophagus = new THREE.Mesh(new THREE.TubeGeometry(eCurve, 30, 0.05, 10, false),
  new THREE.MeshStandardMaterial({{color:0x8a5a66, transparent:true, opacity:0.4}}));
patientDetail.add(esophagus);

// Vocal cords — two small bars that "open" from step 4 onward
const cordsOpen = {str(cords_open).lower()};
const cordGap = cordsOpen ? 0.05 : 0.01;
const cordMat = new THREE.MeshStandardMaterial({{color:0xffd4d4}});
const cordL = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.015, 0.02), cordMat);
cordL.position.set(-cordGap, -0.02, 0.25);
patientDetail.add(cordL);
const cordR = cordL.clone();
cordR.position.set(cordGap, -0.02, 0.25);
patientDetail.add(cordR);

// Laryngoscope — visible during steps 4-5
if ({str(step_idx in (3,4)).lower()}) {{
  const blade = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.03, 0.4),
    new THREE.MeshStandardMaterial({{color:0xcccccc, metalness:0.8, roughness:0.2}}));
  blade.position.set(0.05, 0.02, 0.45);
  blade.rotation.x = -0.3;
  patientDetail.add(blade);
  const handle = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.3, 12),
    new THREE.MeshStandardMaterial({{color:0x333333, metalness:0.6}}));
  handle.position.set(0.05, 0.2, 0.68);
  handle.rotation.x = 0.6;
  patientDetail.add(handle);
}}

// ET tube — advances from step 5 onward, cuff visible once inflated
if ({str(tube_in).lower()}) {{
  const tube = new THREE.Mesh(new THREE.TubeGeometry(curve, 30, 0.028, 10, false),
    new THREE.MeshStandardMaterial({{color:0xfff2cc, transparent:true, opacity:0.9}}));
  patientDetail.add(tube);
  if ({str(cuff_inflated).lower()}) {{
    const cuff = new THREE.Mesh(new THREE.SphereGeometry(0.06, 16, 16),
      new THREE.MeshStandardMaterial({{color:0xffffff, transparent:true, opacity:0.6}}));
    cuff.position.copy(curve.getPoint(0.75));
    patientDetail.add(cuff);
  }}
}}
"""


def render(student_id: str):
    dept_header("Endotracheal Intubation", "🫁", student_id)
    st.info("🔒 **Teaching simulation on a manikin — follow local protocol.** Full 3D OR room: "
            "orbit, zoom and pan to inspect from any angle. Does not simulate touch/resistance "
            "— see the note below.")
    with st.expander("⚠️ What this is and isn't"):
        st.markdown(
            "- **Is:** a real, freely navigable 3D OR room (rotate, zoom in/out, pan) with a "
            "laryngoscope/ET-tube sequence, a Room-view / Close-up camera toggle, and a "
            "safety-reasoning quiz.\n"
            "- **Isn't:** a substitute for airway-manikin practice. This airway is a schematic "
            "for orientation and sequencing, not a licensed anatomical model, and there's no "
            "simulated resistance or tactile feedback — that needs a physical manikin."
        )

    step_idx = st.session_state.get("intub_step", 0)

    cols = st.columns([1, 4, 1])
    with cols[0]:
        if st.button("⬅ Prev", disabled=step_idx == 0, key="intub_prev"):
            st.session_state.intub_step = max(0, step_idx - 1)
            st.rerun()
    with cols[2]:
        if st.button("Next ➡", disabled=step_idx == len(STEPS) - 1, key="intub_next"):
            st.session_state.intub_step = min(len(STEPS) - 1, step_idx + 1)
            st.rerun()

    step = STEPS[step_idx]
    with cols[1]:
        st.progress((step_idx + 1) / len(STEPS), text=f"Step {step_idx+1} of {len(STEPS)}: {step['title']}")

    view_mode = st.radio(
        "Anatomy view", ["🏥 Schematic OR room (orbit/zoom/pan)", "🫁 Real anatomy (CT-derived model)"],
        horizontal=True, key="intub_view_mode",
        help="The schematic room is fully free-roam 3D. The real model is a genuine CT-derived "
             "airway from University of Dundee School of Medicine — NonCommercial license, so it "
             "must not be shown inside any tuition/paid program (e.g. the ASHM Fellowship) without "
             "written permission from Dundee first.",
    )
    if view_mode.startswith("🫁"):
        st.caption("Real CT-derived airway anatomy — © University of Dundee School of Medicine, "
                   "CC BY-NC-SA 4.0. Educational, non-commercial use only. "
                   "[View original on Sketchfab](https://sketchfab.com/3d-models/pharynx-and-floor-of-mouth-b262c70bf9bd49c2a5428581b754f24b).")
        components.html(
            '<div style="position:relative;width:100%;height:500px;">'
            '<iframe title="Pharynx and floor of mouth" frameborder="0" allowfullscreen '
            'mozallowfullscreen="true" webkitallowfullscreen="true" allow="autoplay; fullscreen; xr-spatial-tracking" '
            'style="width:100%;height:100%;" '
            'src="https://sketchfab.com/models/b262c70bf9bd49c2a5428581b754f24b/embed">'
            '</iframe></div>',
            height=500, scrolling=False,
        )
    else:
        html = build_or_room_html(_build_patient_js(step_idx), step["title"])
        components.html(html, height=520, scrolling=False)

    st.markdown(f"**{step['title']}** — {step['note']}")
    st.caption(f"Why it matters: {step['why']}")
    st.markdown("---")

    if step_idx == len(STEPS) - 1:
        st.markdown("### 🧪 Safety Quiz")
        answers = []
        for i, q in enumerate(QUIZ):
            answers.append(st.radio(q["q"], q["options"], key=f"intub_q_{i}", index=None))
        if st.button("Submit Quiz", type="primary", key="intub_submit"):
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
            record_session(student_id, "intubation", pct, "7-step sequence")
