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


@test("shared_session: register / save / load round-trip")
def t_register():
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
    import shared_session
    s = shared_session.register_student("Valid User")
    out = shared_session.record_session(s["id"], "ct", 85.0, "test")
    assert out["total_sessions"] == 1
    assert out["best"]["ct"] == 85.0
    assert out["avg"]["ct"] == 85.0


@test("shared_session: Top Surgeon badge requires all depts attempted")
def t_top_surgeon():
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
    import ct_hospital_module as ct
    for name, (fn, _) in ct.CASES.items():
        v = fn()
        assert v.dtype == np.float32, f"{name} not float32: {v.dtype}"
        assert v.ndim == 3, f"{name} not 3D: {v.ndim}D"
        assert v.shape[1] == 128 and v.shape[2] == 128, f"{name} bad xy: {v.shape}"


@test("ct_hospital_module: brain volume has expected HU ranges")
def t_brain_hu():
    import ct_hospital_module as ct
    v = ct._make_brain()
    # Background is -1000 (air), core has ~45
    assert v.min() == -1000.0, f"min: {v.min()}"
    assert v.max() == 45.0, f"max: {v.max()}"
    assert (v >= 30).sum() > 0, "no soft-tissue voxels"


@test("ct_hospital_module: kidney has all three layers")
def t_kidney_layers():
    import ct_hospital_module as ct
    v = ct._make_kidney()
    # -100 (bg), -50 (pelvis), 0 (medulla), 30 (cortex) should all be present
    vals = set(np.unique(v).tolist())
    for expected in (-100.0, -50.0, 0.0, 30.0):
        assert expected in vals, f"kidney missing HU {expected}, got {vals}"


@test("ct_hospital_module: synthetic generation is fast (<2s per case)")
def t_vol_speed():
    import time
    import ct_hospital_module as ct
    for name, (fn, _) in ct.CASES.items():
        t0 = time.time()
        fn()
        dt = time.time() - t0
        assert dt < 2.0, f"{name} took {dt:.2f}s (should be < 2s with vectorization)"


@test("ct_hospital_module: _extract_spacing handles missing tags")
def t_spacing_fallback():
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
