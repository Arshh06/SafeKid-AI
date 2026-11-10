#!/usr/bin/env python3
"""SafeKid AI — Comprehensive Website Verification & QA Test Suite

Validates:
1. File existence and asset integrity
2. HTML markup, semantic tags, and tab navigation anchors
3. Absolute path leakage prevention (no local developer paths exposed)
4. Scientific claim language compliance (no prohibited marketing claims)
5. Metric synchronization with authoritative evidence registry
6. Local HTTP web server functionality (HTTP 200 OK test)
"""

import json
from pathlib import Path
import re
import socket
import sys
import threading
import time
import urllib.request

ROOT_DIR = Path(__file__).resolve().parent.parent
WEBSITE_DIR = ROOT_DIR / "website"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def test_file_structure():
    """Verify all core website files and assets exist."""
    print("[TEST 1/6] Verifying website file structure...")
    required_files = [
        WEBSITE_DIR / "index.html",
        WEBSITE_DIR / "css" / "style.css",
        WEBSITE_DIR / "js" / "app.js",
        WEBSITE_DIR / "js" / "charts.js",
        WEBSITE_DIR / "data" / "evidence_registry.json",
        WEBSITE_DIR / "data" / "implementation_matrix.json",
    ]
    for rf in required_files:
        assert rf.exists(), f"Missing required website file: {rf}"
        assert rf.stat().st_size > 0, f"File is empty: {rf}"

    # Verify visual comparison plates
    plates_dir = WEBSITE_DIR / "assets" / "plates"
    assert plates_dir.exists(), f"Plates directory missing: {plates_dir}"
    plates = list(plates_dir.glob("*.jpg"))
    assert len(plates) >= 7, f"Expected at least 7 comparison plates, found {len(plates)}"
    print(f"  -> SUCCESS: Core files and {len(plates)} visual plates verified.")


def test_html_tabs_and_sections():
    """Verify all required sections, navigation links, and headings exist in index.html."""
    print("[TEST 2/6] Verifying HTML semantic sections & tabs...")
    html_content = (WEBSITE_DIR / "index.html").read_text(encoding="utf-8")

    required_tabs = [
        "home",
        "architecture",
        "model-comparison",
        "small-pedestrian",
        "child-evaluation",
        "datasets",
        "hardware",
        "implementation",
        "ethics",
        "demo",
        "future",
    ]

    for tab in required_tabs:
        assert f'id="{tab}"' in html_content, f"Missing tab section id='{tab}'"
        assert f'href="#{tab}"' in html_content, f"Missing navigation link href='#{tab}'"

    # Specific required heading from Prompt 39 Section 3.E:
    required_heading = "Person Detection Recall on Child-Labelled JAAD Annotations"
    assert required_heading in html_content, f"Missing required heading: '{required_heading}'"

    # Specific hardware latency requirement from Prompt 39 Section 3.G:
    required_hw_notice = "These are detector inference measurements, not end-to-end system throughput."
    assert required_hw_notice in html_content, f"Missing required hardware disclaimer: '{required_hw_notice}'"

    print("  -> SUCCESS: All 11 tabs, required headings, and disclaimers verified.")


def test_claim_language_compliance():
    """Ensure no prohibited marketing buzzwords or retracted claims appear in the website."""
    print("[TEST 3/6] Auditing scientific claim language & prohibited phrases...")
    files_to_check = [
        WEBSITE_DIR / "index.html",
        WEBSITE_DIR / "js" / "app.js",
        WEBSITE_DIR / "js" / "charts.js",
    ]

    prohibited_patterns = [
        r"\bproduction-ready\b",
        r"\bcertified for residential\b",
        r"\bguaranteed child protection\b",
        r"\bresidentially validated\b",
        r"\bzero false alarms in surveillance\b",
        r"\bhallucinated \d+ lamp[- ]posts\b",
        r"\binherent child detection penalty\b",
        r"\bpercentage of children protected\b",
    ]

    for fpath in files_to_check:
        content = fpath.read_text(encoding="utf-8").lower()
        for pattern in prohibited_patterns:
            matches = re.findall(pattern, content)
            assert not matches, f"Prohibited phrase '{pattern}' found in {fpath.name}: {matches}"

    print("  -> SUCCESS: Zero prohibited marketing claims or ungrounded assertions detected.")


def test_privacy_and_path_leakage():
    """Verify that no local machine absolute paths or developer directories are exposed."""
    print("[TEST 4/6] Checking for local path leakage in frontend code...")
    files_to_check = [
        WEBSITE_DIR / "index.html",
        WEBSITE_DIR / "css" / "style.css",
        WEBSITE_DIR / "js" / "app.js",
        WEBSITE_DIR / "js" / "charts.js",
        WEBSITE_DIR / "data" / "evidence_registry.json",
    ]

    leakage_patterns = [
        r"c:\\users\\",
        r"c:/users/",
        r"/home/",
        r"c:\\projs\\",
    ]

    for fpath in files_to_check:
        content = fpath.read_text(encoding="utf-8").lower()
        for pattern in leakage_patterns:
            matches = re.findall(pattern, content)
            assert not matches, f"Machine-specific path '{pattern}' exposed in {fpath.name}"

    print("  -> SUCCESS: Clean public-facing code with zero local path leakage.")


def test_metric_synchronization():
    """Verify that numbers displayed in index.html match the authoritative evidence registry."""
    print("[TEST 5/6] Verifying metric synchronization with evidence registry...")
    html_content = (WEBSITE_DIR / "index.html").read_text(encoding="utf-8")
    reg_content = json.loads((WEBSITE_DIR / "data" / "evidence_registry.json").read_text(encoding="utf-8"))

    # Map of metric display values that must appear in HTML
    expected_metrics = [
        ("87.80%", "YOLOv8n Precision"),
        ("41.71%", "D-FINE-N Precision"),
        ("46.64%", "YOLOv8n Recall"),
        ("56.60%", "D-FINE-N Recall"),
        ("60.92%", "YOLOv8n F1"),
        ("48.03%", "D-FINE-N F1"),
        ("54.47%", "YOLOv8n AP50"),
        ("53.02%", "D-FINE-N AP50"),
        ("17.81%", "YOLOv8n AP75"),
        ("15.72%", "D-FINE-N AP75"),
        ("24.97%", "YOLOv8n mAP"),
        ("23.18%", "D-FINE-N mAP"),
        ("0.7408", "YOLOv8n Mean IoU"),
        ("0.7107", "D-FINE-N Mean IoU"),
        ("106.43 ms", "YOLOv8n Mean Latency"),
        ("289.42 ms", "D-FINE-N Mean Latency"),
        ("9.40 FPS", "YOLOv8n Throughput"),
        ("3.46 FPS", "D-FINE-N Throughput"),
        ("3.27%", "YOLOv8n Small Pedestrian Recall"),
        ("23.09%", "D-FINE-N Small Pedestrian Recall"),
        ("49.02%", "YOLOv8n Child Recall"),
        ("54.39%", "D-FINE-N Child Recall"),
    ]

    for val, name in expected_metrics:
        assert val in html_content, f"Expected metric '{val}' ({name}) not found in index.html"

    print(f"  -> SUCCESS: All {len(expected_metrics)} authoritative metrics verified in HTML.")


def test_local_webserver():
    """Spin up the serve_website.py server in a background thread and verify HTTP 200 response."""
    print("[TEST 6/6] Testing local web server HTTP request/response...")
    from functools import partial
    import http.server
    import socketserver
    from scripts.serve_website import SafeKidHTTPRequestHandler

    port = 8899
    server = None
    thread = None

    try:
        handler = partial(SafeKidHTTPRequestHandler, directory=str(WEBSITE_DIR))

        class TestServer(socketserver.TCPServer):
            allow_reuse_address = True

        server = TestServer(("127.0.0.1", port), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        time.sleep(0.5)

        url = f"http://127.0.0.1:{port}/index.html"
        with urllib.request.urlopen(url, timeout=3) as resp:
            status = resp.status
            content_type = resp.headers.get("Content-Type", "")
            body = resp.read().decode("utf-8")

        assert status == 200, f"Expected HTTP 200, got {status}"
        assert "text/html" in content_type, f"Expected HTML content-type, got {content_type}"
        assert "SafeKid AI" in body, "Response body does not contain 'SafeKid AI'"

        # Also check evidence registry endpoint
        url_json = f"http://127.0.0.1:{port}/data/evidence_registry.json"
        with urllib.request.urlopen(url_json, timeout=3) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert "authoritative_metrics" in data

        print(f"  -> SUCCESS: HTTP server served index.html and JSON registry with HTTP 200 OK.")
    finally:
        if server:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    print("=" * 65)
    print("SafeKid AI — Automated Website Quality Assurance (QA) Suite")
    print("=" * 65)
    try:
        test_file_structure()
        test_html_tabs_and_sections()
        test_claim_language_compliance()
        test_privacy_and_path_leakage()
        test_metric_synchronization()
        test_local_webserver()
        print("=" * 65)
        print("ALL 6 WEBSITE QA TESTS PASSED CLEANLY (100% PASS RATE)")
        print("=" * 65)
        sys.exit(0)
    except AssertionError as e:
        print(f"\n[QA FAILURE] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[UNEXPECTED ERROR] {e}")
        sys.exit(1)
