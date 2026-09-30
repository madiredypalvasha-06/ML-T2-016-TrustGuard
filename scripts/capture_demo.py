#!/usr/bin/env python3
"""Drive the TrustGuard dashboard in a real browser: screenshots + demo video.

Screenshots land in submission/screenshots/ and the walkthrough video in
submission/. Both are captured against a running Streamlit server.

Usage:
    python3 scripts/capture_demo.py --url http://localhost:8502
    python3 scripts/capture_demo.py --url http://localhost:8502 --video
"""
import argparse
import re
import pathlib
import sys
import time

from playwright.sync_api import sync_playwright

BASE = pathlib.Path(__file__).resolve().parent.parent
SHOT_DIR = BASE / "submission" / "screenshots"
VIDEO_DIR = BASE / "submission"

VIEWPORT = {"width": 1600, "height": 1000}

# Streamlit paints its own chrome; wait for the app to report "running".
RUNNING = '[data-testid="stStatusWidget"]'


def wait_for_app(page, timeout: int = 180_000) -> None:
    """Block until Streamlit has finished the initial script run."""
    page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=timeout)
    deadline = time.time() + timeout / 1000
    while time.time() < deadline:
        running = page.locator('[data-testid="stStatusWidget"]').count()
        if running == 0:
            page.wait_for_timeout(1200)
            return
        page.wait_for_timeout(500)
    print("[capture] warning: app still busy at timeout; capturing anyway")


def settle(page, ms: int = 2500) -> None:
    """Wait for Streamlit to finish re-running, then let charts animate.

    Checking only for the *absence* of the busy widget races the framework:
    immediately after a click the app has not yet started re-running, so the
    widget is still absent and we would proceed against a stale DOM. We
    therefore give the run a moment to signal that it started, then wait for
    it to finish.
    """
    page.wait_for_timeout(500)
    appeared = False
    for _ in range(24):  # ~12 s ceiling for the run to start
        if page.locator('[data-testid="stStatusWidget"]').count():
            appeared = True
            break
        page.wait_for_timeout(500)
    if appeared:
        try:
            page.wait_for_function(
                '() => !document.querySelector(\'[data-testid="stStatusWidget"]\')',
                timeout=180_000,
            )
        except Exception:
            print("[capture] warning: script run did not finish; capturing anyway")
    page.wait_for_timeout(ms)
    page.wait_for_timeout(900)  # Plotly draw + CSS transition


def pick(page, label_regex: str, timeout: int = 30_000) -> bool:
    """Click a Streamlit radio option whose text matches label_regex.

    Streamlit prefixes each option's label with a hidden field name ("panel",
    "source", ...), so anchored patterns are matched against the *last* line of
    the label text as well as the whole string.
    """


    pattern = re.compile(label_regex, re.I)
    selectors = [
        '[data-testid="stSidebar"] [data-testid="stRadio"] label',
        '[data-testid="stMain"] [data-testid="stRadio"] label',
        '[data-testid="stRadio"] label',
        '[data-testid="stSidebar"] [role="radio"]',
    ]
    for sel in selectors:
        for el in page.locator(sel).all():
            try:
                raw = (el.inner_text() or "").strip()
                candidates = {raw, " ".join(raw.split())}
                # Drop a leading hidden field-name line if present.
                lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
                if len(lines) > 1:
                    candidates.add(lines[-1])
                if any(pattern.search(c) for c in candidates if c):
                    el.click()
                    return True
            except Exception:
                continue
    print(f"[capture] no control matching {label_regex!r}")
    return False


def pick_select(page, label_regex: str, option_regex: str | None = None) -> bool:
    """Select an option in a Streamlit selectbox.

    `label_regex` identifies *which* widget (e.g. its label "corruption");
    `option_regex` identifies *which value* to choose inside it (e.g.
    "gaussian_noise"). With no `option_regex` the widget is opened and closed
    unchanged.
    """
    label_pat = re.compile(label_regex, re.I)
    opt_pat = re.compile(option_regex, re.I) if option_regex else None

    for box in page.locator('[data-testid="stSelectbox"]').all():
        try:
            label = (
                box.locator('[data-testid="stWidgetLabel"]').inner_text() or ""
            ).strip()
            if label and not label_pat.search(label):
                continue
            box.locator('[role="combobox"]').first.click(timeout=20_000)
            page.wait_for_timeout(800)
            options = page.locator('[role="option"]').all()
            if opt_pat is None:
                page.keyboard.press("Escape")
                return True
            for opt in options:
                if opt_pat.search((opt.inner_text() or "").strip()):
                    opt.click()
                    page.wait_for_timeout(400)
                    return True
            page.keyboard.press("Escape")
            print(
                f"[capture] options in {label!r} did not match "
                f"{option_regex!r}: "
                f"{[(o.inner_text() or '').strip() for o in options]}"
            )
        except Exception as exc:
            print(f"[capture] selectbox {label_regex!r}: {type(exc).__name__}: {exc}")
            continue
    print(f"[capture] no selectbox labelled {label_regex!r}")
    return False


def set_slider(page, testid: str, value: float) -> None:
    """Move a Streamlit slider by its accessible label text."""
    try:
        sl = page.locator(f'[data-testid="{testid}"] [role="slider"]').first
        sl.click()
        for _ in range(80):
            cur = sl.get_attribute("aria-valuenow")
            if cur and abs(float(cur) - value) < 0.5:
                break
            sl.press("ArrowRight" if value > float(cur or 0) else "ArrowLeft")
        page.wait_for_timeout(600)
    except Exception as exc:
        print(f"[capture] slider {testid} -> {value} failed: {exc}")


def shoot(page, name: str) -> pathlib.Path:
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    out = SHOT_DIR / f"{name}.png"
    page.screenshot(path=str(out), full_page=True)
    print(f"[capture] {out.relative_to(BASE)}  ({out.stat().st_size/1024:.0f} KB)")
    return out


def tour(page, shot: bool = True) -> None:
    """The walkthrough used for both the screenshots and the demo video.

    Five captures, matching the five slots the submission form allows:
      1. Mission Control on a clean input      -> TRUST
      2. Mission Control on a corrupted input  -> REVIEW / ABSTAIN
      3. Drift Monitor scanning a shifted feed -> watchpoint + flagged inputs
      4. Calibration Lab                       -> reliability, raw vs scaled
      5. Method Showdown                       -> all methods head-to-head
    """

    def verify(page, needle: str) -> bool:
        """Confirm a piece of text is actually on screen before we shoot."""
        try:
            return page.get_by_text(needle, exact=False).first.is_visible(
                timeout=8000
            )
        except Exception:
            return False

    def step(panel: str) -> None:
        print(f"[capture] -> {panel}")
        pick(page, re.escape(panel))
        settle(page, 3000)

    # 1 -- clean input, expect a TRUST verdict with all guardrails green.
    step("Mission Control")
    pick(page, r"Clean CIFAR")
    settle(page, 3600)
    print(f"[capture]   verdict TRUST visible: {verify(page, 'TRUST')}")
    if shot:
        shoot(page, "01_mission_control_trust")

    # 2 -- severely corrupted input: the familiarity guardrail must trip.
    pick(page, r"Corrupted sample")
    settle(page, 3000)
    pick_select(page, r"corruption", r"^gaussian_noise$")
    settle(page, 3000)
    pick_select(page, r"severity", r"^5$")
    settle(page, 4000)
    ood = verify(page, "ABSTAIN") or verify(page, "REVIEW")
    print(f"[capture]   OOD verdict visible: {ood}")
    if shot:
        shoot(page, "02_mission_control_abstain_ood")

    # 3 -- stream-level monitoring. Severity 2 of gaussian noise is the most
    # dangerous regime and the one worth showing: the trust rate still looks
    # healthy (70.2% accepted) while accuracy on those accepted inputs has
    # collapsed to 38.4%, which trips the WATCHPOINT alarm. This is exactly
    # the silent failure a naive classifier cannot surface.
    step("Drift Monitor")
    pick_select(page, r"drift type", r"^gaussian_noise$")
    settle(page, 3600)
    pick_select(page, r"severity", r"^2$")
    settle(page, 5200)
    print(f"[capture]   WATCHPOINT alarm visible: {verify(page, 'WATCHPOINT')}")
    if shot:
        shoot(page, "03_drift_monitor_watchpoint")

    # 4 -- calibration.
    step("Calibration Lab")
    if shot:
        shoot(page, "04_calibration_lab")

    # 5 -- head-to-head method comparison.
    step("Method Showdown")
    if shot:
        shoot(page, "05_method_showdown")


def record_video(url: str, out_name: str) -> pathlib.Path:
    """Record the walkthrough as a webm (VP9, no external encoder needed)."""
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    out = VIDEO_DIR / out_name
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
        ctx = browser.new_context(
            viewport=VIEWPORT,
            record_video_dir=str(VIDEO_DIR),
            record_video_size=VIEWPORT,
        )
        page = ctx.new_page()
        page.goto(url, wait_until="load", timeout=120_000)
        wait_for_app(page)
        settle(page, 3500)
        tour(page, shot=False)
        settle(page, 2500)
        ctx.close()  # flushes the recording
        browser.close()

    produced = sorted(
        VIDEO_DIR.glob("*.webm"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if produced and produced[0] != out:
        produced[0].rename(out)
    print(f"[capture] video -> {out.relative_to(BASE)}  ({out.stat().st_size/1e6:.1f} MB)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8502")
    ap.add_argument("--video", action="store_true", help="also record the walkthrough")
    ap.add_argument("--video-only", action="store_true")
    ap.add_argument("--name", default="TrustGuard_demo_walkthrough.webm")
    ap.add_argument("--headful", action="store_true")
    args = ap.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headful)
        ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=2)
        page = ctx.new_page()
        print(f"[capture] opening {args.url}")
        page.goto(args.url, wait_until="load", timeout=120_000)
        wait_for_app(page)
        settle(page, 3500)
        if not args.video_only:
            tour(page, shot=True)
        ctx.close()
        browser.close()

    if args.video or args.video_only:
        record_video(args.url, args.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
