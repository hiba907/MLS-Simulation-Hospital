"""Smoke tests for the AI MLS Hospital modules.

Run:  python test_smoke.py
Exits 0 on success, non-zero on failure.

These cover pure-Python pieces only (volume generators, dept whitelist,
atomic write, badge rules). Streamlit UI is not exercised.
"""
import sys, os, traceback, tempfile, shutil
import numpy as np

# Make local modules importable when run from this folder
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def test(name):
    def deco(fn):
        def wrapper():
            try:
                fn()
                print(f"  ✅  {name}")
                return True
            except Exception as e:
                print(f"  ❌  {name}: {e}")
                traceback.print_exc()
                return False
        return wrapper
    return deco


# ── shared_session tests (with isolated temp data dir) ────────────────────────
TMP = tempfile.mkdtemp(prefix="mls_test_")
os.environ["MLS_TEST_DIR"] = TMP


def _has_streamlit():
    try:
        import streamlit  # noqa
        return True
    except ImportError:
        return False


@test("shared_session: register / save / load round-trip")
def t_register():
    if not _has_streamlit():
        print("    (skipped: streamlit not installed)")
        return
    import shared_session
    # Monkeypatch STUDENTS_DIR to isolated temp
    shared_session.STUDENTS_DIR = TMP
    # Clean any existing test data
    for f in os.listdir(TMP):
        os.remove(os.path.join(TMP, f))
    s = shared_session.register_student("Test User")
    assert s["id"].startswith("s_")
    assert s["name"] == "Test User"
    loaded = shared_session.load_student(s["id"])
    assert loaded["name"] == "Test User"
    assert loaded["total_sessions"] == 0


@test("shared_session: record_session rejects unknown dept")
def t_dept_whitelist():
    if not _has_streamlit():
        print("    (skipped: streamlit not installed)")
        return
    import shared_session
    s = shared_session.register_student("Whitelist User")
    try:
        shared_session.record_session(s["id"], "ct2", 80.0)
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "ct2" in str(e)
        assert "ct" in str(e)


@test("shared_session: record_session accepts valid dept")
def t_valid_dept():
    if not _has_streamlit():
        print("    (skipped: streamlit not installed)")
        return
    import shared_session
    s = shared_session.register_student("Valid User")
    out = shared_session.record_session(s["id"], "ct", 85.0, "test")
    assert out["total_sessions"] == 1
    assert out["best"]["ct"] == 85.0
    assert out["avg"]["ct"] == 85.0


@test("shared_session: Top Surgeon badge requires all depts attempted")
def t_top_surgeon():
    if not _has_streamlit():
        print("    (skipped: streamlit not installed)")
        return
    import shared_session
    s = shared_session.register_student("Badge User")
    # Only do CT and ultrasound well — no others attempted
    shared_session.record_session(s["id"], "ct", 90.0)
    shared_session.record_session(s["id"], "ultrasound", 90.0)
    loaded = shared_session.load_student(s["id"])
    # Should NOT have Top Surgeon yet (haven't tried surgery/fls/endoscopy)
    assert "🏆Top Surgeon" not in loaded["badges"], \
        f"Got badge prematurely: {loaded['badges']}"
    # Now do all three remaining at >= 80
    for d in ("surgery", "fls", "endoscopy"):
        shared_session.record_session(s["id"], d, 85.0)
    loaded = shared_session.load_student(s["id"])
    assert "🏆Top Surgeon" in loaded["badges"], \
        f"Should have badge now: {loaded['badges']}"


@test("shared_session: atomic write survives mid-write crash")
def t_atomic_write():
    if not _has_streamlit():
        print("    (skipped: streamlit not installed)")
        return
    import shared_session
    s = shared_session.register_student("Atomic User")
    # Verify the temp-file rename pattern is used (target has no .tmp leftover)
    files = os.listdir(TMP)
    leftover = [f for f in files if f.endswith(".tmp")]
    assert not leftover, f"Temp file leaked: {leftover}"
    # Simulate crash: write garbage to .tmp — should not corrupt main file
    target = os.path.join(TMP, f"{s['id']}.json")
    tmp = target + ".tmp"
    with open(tmp, "w") as f:
        f.write("{ this is not valid json")
    # Main file still loads fine
    loaded = shared_session.load_student(s["id"])
    assert loaded["name"] == "Atomic User"


# ── ct_hospital_module tests (volume generators + spacing helpers) ────────────
@test("ct_hospital_module: synthetic volumes are correct shape & dtype")
def t_vol_shapes():
    if not _has_streamlit():
        print("    (skipped: streamlit not installed)")
        return
    import ct_hospital_module as ct
    for name, (fn, _) in ct.CASES.items():
        v = fn()
        assert v.dtype == np.float32, f"{name} not float32: {v.dtype}"
        assert v.ndim == 3, f"{name} not 3D: {v.ndim}D"
        assert v.shape[1] == 144 and v.shape[2] == 144, f"{name} bad xy: {v.shape}"


@test("ct_hospital_module: brain volume has expected HU ranges")
def t_brain_hu():
    if not _has_streamlit():
        print("    (skipped: streamlit not installed)")
        return
    import ct_hospital_module as ct
    v = ct._make_brain()
    # Background is -1000 (air); cortex 40, white matter 32, skull 1200, CSF 8
    assert v.min() == -1000.0, f"min: {v.min()}"
    assert v.max() >= 1000.0, f"max should be skull (~1200), got {v.max()}"
    # White matter (~32) and gray matter (~40) must be present
    vals = set(np.round(np.unique(v), 0).tolist())
    assert 32 in vals, f"no white matter HU ~32, got {sorted(vals)[:10]}"
    assert 40 in vals, f"no gray matter HU ~40, got {sorted(vals)[:10]}"
    assert (v >= 30).sum() > 100, "no soft-tissue voxels"


@test("ct_hospital_module: kidney has all three layers")
def t_kidney_layers():
    if not _has_streamlit():
        print("    (skipped: streamlit not installed)")
        return
    import ct_hospital_module as ct
    v = ct._make_kidney()
    # New anatomical kidney: 35 (cortex), 25 (medulla), 5 (pelvis), -80 (fat), 10 (skin), 30 (soft tissue)
    vals = set(np.unique(v).tolist())
    for expected in (35.0, 25.0, 5.0, -80.0, 30.0):
        assert expected in vals, f"kidney missing HU {expected}, got {vals}"


@test("ct_hospital_module: synthetic generation is fast (<2s per case)")
def t_vol_speed():
    if not _has_streamlit():
        print("    (skipped: streamlit not installed)")
        return
    import time
    import ct_hospital_module as ct
    for name, (fn, _) in ct.CASES.items():
        t0 = time.time()
        fn()
        dt = time.time() - t0
        assert dt < 2.0, f"{name} took {dt:.2f}s (should be < 2s with vectorization)"


@test("ct_hospital_module: _extract_spacing handles missing tags")
def t_spacing_fallback():
    if not _has_streamlit():
        print("    (skipped: streamlit not installed)")
        return
    import ct_hospital_module as ct
    class FakeDS:
        pass
    ds = FakeDS()  # no PixelSpacing, no SliceThickness
    r, c, s = ct._extract_spacing(ds)
    assert r == 0.5 and c == 0.5 and s == 0.5, f"fallback failed: {(r,c,s)}"

    ds2 = FakeDS()
    ds2.PixelSpacing = [0.7, 0.7]
    ds2.SliceThickness = 1.5
    r, c, s = ct._extract_spacing(ds2)
    assert r == 0.7 and c == 0.7 and s == 1.5




# ── suture_module tests ──────────────────────────────────────────────────────
@test("suture_module: 8 suture patterns are defined")
def t_suture_count():
    import suture_module
    assert len(suture_module.SUTURE_TYPES) >= 8,         f"Need ≥8 patterns, got {len(suture_module.SUTURE_TYPES)}"
    for name, info in suture_module.SUTURE_TYPES.items():
        assert "description" in info
        assert "indications" in info
        assert "n_stitches" in info
        assert "n_bites_per_stitch" in info
        assert callable(info["bites"])
        targets = info["bites"](0.8, 1.0)
        # Continuous patterns also include "in-between" connecting bites
        assert len(targets) >= info["n_stitches"], f"{name}: got {len(targets)} targets, expected >= {info['n_stitches']}"
        # All targets in [0, 1] range
        for x, y in targets:
            assert 0.0 <= x <= 1.0, f"{name}: target x={x} out of [0,1]"
            assert 0.0 <= y <= 1.0, f"{name}: target y={y} out of [0,1]"


@test("suture_module: score_stitch gives full marks at target")
def t_suture_score_perfect():
    import suture_module
    score, dist, status = suture_module.score_stitch((0.5, 0.5), (0.5, 0.5))
    assert score == 100.0
    assert dist == 0.0
    assert status == "excellent"


@test("suture_module: score_stitch gives 0 beyond max_dist")
def t_suture_score_miss():
    import suture_module
    score, dist, status = suture_module.score_stitch((0.0, 0.0), (0.5, 0.5),
                                                     max_dist=0.18)
    assert score == 0.0
    assert status == "miss"


@test("suture_module: render_schematic produces a figure for every type")
def t_schematic_renders():
    import suture_module
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name in suture_module.SUTURE_TYPES:
        fig = suture_module.render_schematic(name)
        assert fig is not None, f"{name} returned no figure"
        plt.close(fig)


# ── reasoning_canvas tests (the new department) ─────────────────────────────
@test("reasoning_canvas: empty tree mutates correctly")
def t_rc_empty_tree():
    import reasoning_canvas as rc
    t = rc.empty_tree()
    assert t["nodes"] == []
    assert t["edges"] == []
    n1 = rc.add_node(t, "history", "Chest pain history")
    n2 = rc.add_node(t, "exam",    "CV exam")
    assert n1["id"] == "n_001"
    assert n2["id"] == "n_002"
    rc.add_edge(t, n1["id"], n2["id"])
    assert len(t["edges"]) == 1
    # Cycle detection
    rc.add_edge(t, n2["id"], n1["id"])
    try:
        rc.topological_order(t)
        assert False, "should have raised ValueError on cycle"
    except ValueError:
        pass


@test("reasoning_canvas: scoring on perfect tree = ~100")
def t_rc_perfect_score():
    import reasoning_canvas as rc
    case = rc.get_case("stemi_58m")
    perfect = {
        "nodes": [dict(n) for n in case["expert_tree"]["nodes"]],
        "edges": [dict(e) for e in case["expert_tree"]["edges"]],
    }
    result = rc.score_tree(perfect, case["expert_tree"],
                           case.get("critical_node_labels"))
    assert result["total"] >= 95, f"Perfect tree should score ~100, got {result['total']}"


@test("reasoning_canvas: scoring on bad tree = low")
def t_rc_bad_score():
    import reasoning_canvas as rc
    case = rc.get_case("stemi_58m")
    bad = rc.empty_tree()
    rc.add_node(bad, "imaging", "ECG", "Check the heart", t_added=60)
    rc.add_node(bad, "lab",     "CBC", "Routine labs", t_added=120)
    rc.add_node(bad, "disposition", "Discharge home with PPI",
                "Maybe GERD", t_added=600)
    result = rc.score_tree(bad, case["expert_tree"],
                           case.get("critical_node_labels"))
    # Bad tree should score meaningfully below perfect
    assert result["total"] < 70, f"Bad tree should score < 70, got {result['total']}"
    # Should have missed at least one of the critical must-have steps
    missed_lower = [m.lower() for m in result["diff"]["missed_labels"]]
    assert any("aspirin" in m or "cath" in m or "PCI" in m for m in missed_lower), \
        f"Should miss aspirin/cath-lab/PCI, got {result['diff']['missed_labels']}"


@test("reasoning_canvas: VitalsSim deteriorates on missed critical step")
def t_rc_vitals_deteriorate():
    import reasoning_canvas as rc
    sim = rc.VitalsSim(
        initial={"hr": 108, "sbp": 92, "dbp": 60, "spo2": 94, "rr": 22},
        critical_steps=[{"label": "ECG", "target_by": 120, "done_at": None}],
    )
    sbp_before = sim.snapshot()["sbp"]
    for _ in range(200):  # simulate 200s of doing nothing
        sim.tick(1.0)
    # After 200s with missed ECG, vitals should have worsened
    assert sim.snapshot()["sbp"] < sbp_before, \
        f"Should deteriorate, SBP went {sbp_before} -> {sim.snapshot()['sbp']}"
    assert sim.outcome() in ("crash", "deteriorated", "missed_critical_step"), \
        f"Should be a bad outcome, got {sim.outcome()}"


@test("shared_session: reasoning dept in DEPT_KEYS")
def t_ss_reasoning_dept():
    # Skip if streamlit not available
    try:
        import streamlit  # noqa
    except ImportError:
        print("    (skipped: streamlit not installed)")
        return
    import shared_session
    assert "reasoning" in shared_session.DEPT_KEYS, \
        f"reasoning should be in DEPT_KEYS, got {shared_session.DEPT_KEYS}"
    # record_session should accept it
    s = shared_session.register_student("Reasoning User")
    out = shared_session.record_session(s["id"], "reasoning", 85.0, "test")
    assert out["best"]["reasoning"] == 85.0
    # Reasoner badge should be earned
    assert "🧠Reasoner" in out["badges"], f"Should earn Reasoner badge: {out['badges']}"


# ── elembic_pipeline tests (the v2 NER+L module) ─────────────────────────────
@test("elembic_pipeline: section segmentation finds HPI")
def t_eb_segment():
    import elembic_pipeline as ep
    note = "CHIEF COMPLAINT: chest pain.\n\nHPI: 58M with crushing pain.\n\nPLAN: aspirin."
    sections = ep.segment_sections(note)
    section_names = [s["section"] for s in sections]
    assert "chief_complaint" in section_names
    assert "hpi" in section_names
    assert "plan" in section_names


@test("elembic_pipeline: temporal anchor parses 'day N' and 'before admission'")
def t_eb_anchor():
    import datetime
    import elembic_pipeline as ep
    admit = datetime.date(2026, 1, 15)
    assert ep.anchor_time("day 3", admit, admit) == 3 * 86400
    assert ep.anchor_time("before admission", admit, admit) < 0
    assert ep.anchor_time("ED arrival", admit, admit) == 0
    assert ep.anchor_time("at discharge", admit, admit) == 5 * 86400


# ── Run all ───────────────────────────────────────────────────────────────────
def main():
    print(f"Running smoke tests (data dir: {TMP})...")
    tests = [v for k, v in globals().items() if k.startswith("t_") and callable(v)]
    passed = sum(1 for t in tests if t())
    total = len(tests)
    print(f"\n{passed}/{total} tests passed")
    shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
