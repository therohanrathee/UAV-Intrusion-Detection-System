"""
Real-time Local Inference Server for Tactical UAV Intrusion Detection.
Serves a lightweight HTTP endpoint on localhost:8502 for the live 3D flight arena.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import threading
import socket
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from runtime.inference import UAVInferenceEngine
from pipeline.residuals import compute_physics_residuals, FEATURE_COLUMNS

_server_instance: Optional[HTTPServer] = None
_server_thread: Optional[threading.Thread] = None
_engine_instance: Optional[UAVInferenceEngine] = None


class LiveInferenceHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "running", "model": "uav_ids_model.onnx"}).encode("utf-8"))

    def do_POST(self):
        global _engine_instance
        try:
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            payload = json.loads(body)
            frames = payload.get("frames", [])

            if len(frames) < 10:
                res = {"error": "Need at least 10 frames"}
            else:
                df = pd.DataFrame(frames)
                if "mission_id" not in df.columns:
                    df["mission_id"] = 888
                if "time" not in df.columns:
                    df["time"] = np.arange(len(df)) * 0.02

                df_feat = compute_physics_residuals(df)
                
                feat_matrix = df_feat[FEATURE_COLUMNS].values
                if len(feat_matrix) < 50:
                    pad = np.repeat(feat_matrix[:1], 50 - len(feat_matrix), axis=0)
                    window = np.vstack([pad, feat_matrix])
                else:
                    window = feat_matrix[-50:]

                if _engine_instance is None:
                    _engine_instance = UAVInferenceEngine(
                        onnx_model_path=str(PROJECT_ROOT / "weights/uav_ids_model.onnx"),
                        scaler_path=str(PROJECT_ROOT / "checkpoints/scaler.joblib"),
                        manifest_path=str(PROJECT_ROOT / "weights/model_manifest.json")
                    )

                res = _engine_instance.predict(window)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

    def log_message(self, format, *args):
        pass


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_inference_server(port: int = 8502) -> bool:
    global _server_instance, _server_thread
    if is_port_in_use(port):
        return True

    try:
        _server_instance = HTTPServer(("127.0.0.1", port), LiveInferenceHandler)
        _server_thread = threading.Thread(target=_server_instance.serve_forever, daemon=True)
        _server_thread.start()
        return True
    except Exception as e:
        print(f"Could not start inference server on port {port}: {e}")
        return False


if __name__ == "__main__":
    port = 8502
    print(f"Starting UAV Intrusion Detection ONNX Inference Server on http://127.0.0.1:{port}...")
    server = HTTPServer(("127.0.0.1", port), LiveInferenceHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping inference server.")
        server.server_close()
