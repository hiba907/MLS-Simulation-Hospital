"""
or_room_scene.py
────────────────────────────────────────────────────────────────────────────
Shared Three.js "inside the OR" scene builder — an actual enclosing room
(floor, walls, overhead surgical light, bed, instrument tray) with full
orbit/zoom/pan camera control, used by both foley_module.py and
intubation_module.py so a student orbiting the room feels surrounded by it,
not looking at a single object floating in black space.

HONEST SCOPE NOTE: the room shell and camera rig are real, freely navigable
3D (genuine rotate/zoom/pan on every axis, min/max zoom clamps). The patient
airway / pelvic anatomy inside it is still a schematic built from primitive
shapes, not a licensed anatomical model or real video footage — same
boundary as foley_module.py. Swap in a real .glb (BodyParts3D / licensed
Sketchfab / paid pack) via `extra_glb_url` once one is chosen; the loader
hook is already wired below so that's a one-line change, not a rebuild.

Exposes one function: build_or_room_html(body_group_js, hud_title,
extra_glb_url=None) -> full HTML string for components.html().
`body_group_js` is a JS snippet that builds a THREE.Group named `patientDetail`
and returns it — the caller supplies the procedure-specific anatomy.
"""

ROOM_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;overflow:hidden;font-family:'Segoe UI',sans-serif;color:#e6edf3}
#c{width:100%;height:100%;display:block}
#hud{position:absolute;top:10px;left:10px;background:rgba(0,0,0,.75);border:1px solid rgba(255,255,255,.15);
     border-radius:8px;padding:8px 14px;font-size:13px;max-width:340px}
#hint{position:absolute;bottom:10px;left:50%;transform:translateX(-50%);font-size:11px;color:#9091a4;
      background:rgba(0,0,0,.55);padding:4px 10px;border-radius:6px}
#viewbtns{position:absolute;top:10px;right:10px;display:flex;gap:6px}
.vbtn{background:rgba(20,20,30,.85);border:1px solid rgba(255,255,255,.2);color:#e6edf3;
      border-radius:6px;padding:7px 12px;font-size:12px;cursor:pointer}
.vbtn.on{background:#3a6df0;border-color:#3a6df0}
#toggle{position:absolute;bottom:10px;right:10px;background:rgba(20,20,30,.85);border:1px solid rgba(255,255,255,.2);
        color:#e6edf3;border-radius:6px;padding:6px 12px;font-size:12px;cursor:pointer}
video#gesturebox{position:absolute;bottom:50px;right:10px;width:150px;border-radius:8px;
            border:1px solid rgba(255,255,255,.2);display:none}
</style></head>
<body>
<canvas id="c"></canvas>
<div id="hud"><b>__HUD_TITLE__</b></div>
<div id="viewbtns">
  <button class="vbtn on" data-v="room">🏥 Room view</button>
  <button class="vbtn" data-v="closeup">🔎 Close-up</button>
</div>
<video id="gesturebox" autoplay muted playsinline></video>
<button id="toggle">🖐 Enable gesture camera (optional)</button>
<div id="hint">Drag to orbit &middot; scroll/pinch to zoom &middot; right-drag to pan</div>
<script type="importmap">
{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({canvas, antialias:true});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x11161b);
scene.fog = new THREE.Fog(0x11161b, 10, 22);

const camera = new THREE.PerspectiveCamera(55, window.innerWidth/window.innerHeight, 0.1, 100);
const ROOM_VIEW = {pos:[5.5,4,6.5], target:[0,1,0]};
const CLOSEUP_VIEW = {pos:[0,1.9,2.2], target:[0,1.3,0]};
camera.position.set(...ROOM_VIEW.pos);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.minDistance = 1.2;      // "minimize" — zoom in close
controls.maxDistance = 12;       // "maximize" — pull back to see the whole room
controls.maxPolarAngle = Math.PI * 0.49; // keep camera from diving under the floor
controls.target.set(...ROOM_VIEW.target);
controls.update();

// ── Lighting: one overhead surgical light (the room's visual anchor) ───────
scene.add(new THREE.AmbientLight(0xffffff, 0.35));
const surgLight = new THREE.SpotLight(0xffffff, 3.2, 14, Math.PI/5, 0.4, 1.2);
surgLight.position.set(0, 4.6, 0.3);
surgLight.target.position.set(0, 1, 0);
surgLight.castShadow = true;
scene.add(surgLight, surgLight.target);
const fill = new THREE.DirectionalLight(0xbfd4ff, 0.5);
fill.position.set(-4, 5, -3);
scene.add(fill);

// ── Room shell: floor + two walls, so orbiting feels enclosed ──────────────
const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(14, 14),
  new THREE.MeshStandardMaterial({color:0x3b4650, roughness:0.85})
);
floor.rotation.x = -Math.PI/2;
floor.receiveShadow = true;
scene.add(floor);

const wallMat = new THREE.MeshStandardMaterial({color:0x6fa8a8, roughness:0.9});
const wallBack = new THREE.Mesh(new THREE.PlaneGeometry(14, 6), wallMat);
wallBack.position.set(0, 3, -7);
scene.add(wallBack);
const wallSide = new THREE.Mesh(new THREE.PlaneGeometry(14, 6), wallMat);
wallSide.rotation.y = Math.PI/2;
wallSide.position.set(-7, 3, 0);
scene.add(wallSide);

// ── OR bed ───────────────────────────────────────────────────────────────
const bedFrame = new THREE.Mesh(
  new THREE.BoxGeometry(2.1, 0.5, 0.9),
  new THREE.MeshStandardMaterial({color:0x8a8f96, metalness:0.6, roughness:0.4})
);
bedFrame.position.set(0, 0.5, 0);
bedFrame.castShadow = true;
scene.add(bedFrame);
const mattress = new THREE.Mesh(
  new THREE.BoxGeometry(1.9, 0.18, 0.75),
  new THREE.MeshStandardMaterial({color:0xe9ecef})
);
mattress.position.set(0, 0.85, 0);
scene.add(mattress);

// ── Instrument tray on a stand ──────────────────────────────────────────
const trayStand = new THREE.Mesh(
  new THREE.CylinderGeometry(0.04, 0.04, 1.0, 12),
  new THREE.MeshStandardMaterial({color:0xaaaaaa, metalness:0.7})
);
trayStand.position.set(1.6, 0.5, 1.1);
scene.add(trayStand);
const tray = new THREE.Mesh(
  new THREE.BoxGeometry(0.6, 0.04, 0.4),
  new THREE.MeshStandardMaterial({color:0xcfd8dc, metalness:0.4, roughness:0.3})
);
tray.position.set(1.6, 1.02, 1.1);
scene.add(tray);

// ── Procedure-specific patient anatomy (schematic — see module docstring) ──
__BODY_GROUP_JS__
patientDetail.position.set(0, 1.0, 0);
scene.add(patientDetail);

__EXTRA_GLB_LOADER__

let viewMode = 'room';
function setView(mode) {
  viewMode = mode;
  const v = mode === 'room' ? ROOM_VIEW : CLOSEUP_VIEW;
  const start = camera.position.clone();
  const startTarget = controls.target.clone();
  let t0 = null;
  function step(ts) {
    if (!t0) t0 = ts;
    const p = Math.min(1, (ts - t0) / 500);
    camera.position.lerpVectors(start, new THREE.Vector3(...v.pos), p);
    controls.target.lerpVectors(startTarget, new THREE.Vector3(...v.target), p);
    controls.update();
    if (p < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}
document.querySelectorAll('.vbtn').forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll('.vbtn').forEach(b => b.classList.remove('on'));
    btn.classList.add('on');
    setView(btn.dataset.v);
  };
});

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

// ── Optional MediaPipe gesture camera: pinch to zoom (same math used
// across the project) — entirely opt-in, never auto-starts. ───────────────
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
            const dir = camera.position.clone().sub(controls.target).normalize();
            const newDist = Math.max(controls.minDistance, Math.min(controls.maxDistance,
              camera.position.distanceTo(controls.target) - delta*20));
            camera.position.copy(controls.target).add(dir.multiplyScalar(newDist));
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


def build_or_room_html(body_group_js: str, hud_title: str, extra_glb_url: str = None) -> str:
    """body_group_js must define a JS variable `patientDetail` (a THREE.Group).
    extra_glb_url: optional URL to a real licensed .glb to load alongside/
    instead of the schematic group once one is chosen (not required)."""
    if extra_glb_url:
        loader_js = f"""
const glbLoader = new GLTFLoader();
glbLoader.load("{extra_glb_url}", (gltf) => {{
  gltf.scene.position.set(0, 1.0, 0);
  gltf.scene.scale.set(1, 1, 1);
  scene.add(gltf.scene);
}}, undefined, (err) => console.warn("GLB load failed:", err));
"""
    else:
        loader_js = "// no licensed .glb configured yet — showing schematic only"
    return (ROOM_TEMPLATE.replace("__BODY_GROUP_JS__", body_group_js)
                          .replace("__HUD_TITLE__", hud_title)
                          .replace("__EXTRA_GLB_LOADER__", loader_js))
