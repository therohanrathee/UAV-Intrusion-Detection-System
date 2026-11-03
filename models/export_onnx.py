"""
Universal ONNX Model Exporter.
Exports the PyTorch Multi-Task Deep Sequence Model to a portable .onnx file
and verifies numerical parity with ONNX Runtime.
"""

import torch
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
import json

from models.deep_sequence import UAVMultiTaskDeepNet
from pipeline.residuals import FEATURE_COLUMNS
from sim.attacks import ATTACK_NAMES, SENSOR_KEYS


def export_to_onnx(
    checkpoint_path: str = "checkpoints/best_deep_model.pt",
    output_onnx_path: str = "weights/uav_ids_model.onnx",
    window_size: int = 50,
    in_features: int = len(FEATURE_COLUMNS),
) -> str:
    """Exports the trained model to ONNX with dynamic batch dimensions."""
    Path("weights").mkdir(exist_ok=True)
    
    # 1. Initialize PyTorch model and load weights
    model = UAVMultiTaskDeepNet(
        in_features=in_features,
        num_attacks=len(ATTACK_NAMES),
        num_sensors=len(SENSOR_KEYS),
        hidden_dim=64,
        dropout=0.0  # Set to 0 for inference
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    # 2. Dummy input (Batch=1, Timesteps=50, Features=29)
    dummy_input = torch.randn(1, window_size, in_features, dtype=torch.float32)

    # 3. Export to ONNX
    print(f"Exporting PyTorch model to ONNX: {output_onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["telemetry_window"],
        output_names=["attack_logits", "sensor_logits"],
        dynamic_axes={
            "telemetry_window": {0: "batch_size"},
            "attack_logits": {0: "batch_size"},
            "sensor_logits": {0: "batch_size"},
        }
    )

    # 4. Verify ONNX model integrity
    onnx_model = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX structure verification successful.")

    # 5. Verify Numerical Parity against ONNX Runtime
    with torch.no_grad():
        py_att, py_sen = model(dummy_input)
        py_att_np = py_att.numpy()
        py_sen_np = py_sen.numpy()

    ort_session = ort.InferenceSession(output_onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {"telemetry_window": dummy_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    ort_att_np, ort_sen_np = ort_outputs

    diff_att = np.max(np.abs(py_att_np - ort_att_np))
    diff_sen = np.max(np.abs(py_sen_np - ort_sen_np))
    print(f"Numerical Parity Check: Max Attack Diff={diff_att:.2e}, Max Sensor Diff={diff_sen:.2e}")
    assert diff_att < 1e-4, "Attack logits differ between PyTorch and ONNX!"
    assert diff_sen < 1e-4, "Sensor logits differ between PyTorch and ONNX!"

    # 6. Save model metadata manifest for edge deployment
    manifest = {
        "model_file": output_onnx_path,
        "input_name": "telemetry_window",
        "input_shape": [1, window_size, in_features],
        "feature_names": FEATURE_COLUMNS,
        "attack_classes": ATTACK_NAMES,
        "sensor_channels": SENSOR_KEYS,
        "opset_version": 17,
        "framework": "ONNX Runtime / C++ / Python",
    }
    manifest_path = "weights/model_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Export complete! Model saved to {output_onnx_path}")
    print(f"Manifest saved to {manifest_path}")
    return output_onnx_path


if __name__ == "__main__":
    export_to_onnx()
