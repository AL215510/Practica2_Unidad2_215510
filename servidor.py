"""
servidor.py — Servidor local para el Clasificador de Imagenes
=============================================================
Carga modelos .keras y expone una API REST que usa clasificador.html

Uso:
    python servidor.py

Requiere:
    pip install flask flask-cors tensorflow pillow numpy

El servidor queda en: http://localhost:5050
Luego abre clasificador.html en el navegador.
"""

import os
import io
import json
import base64
import numpy as np
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf

# ── Configuracion ────────────────────────────────────────────
HOST = "127.0.0.1"
PORT = 5050
# ─────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)  # Permite que el HTML abra desde file:// o cualquier origen local

# Estado global del servidor
STATE = {
    "model":          None,
    "labels":         None,   # {0: "bishop", 1: "king", ...}
    "img_size":       (224, 224),
    "channels":       3,
    "num_classes":    0,
    "model_name":     None,
    "preprocessing":  "default",  # "default" = /255  |  "mobilenet_v2" = [-1,1]
}


# ── Utilidades ───────────────────────────────────────────────

def preprocess_image(img_bytes: bytes, img_size: tuple, channels: int,
                     preprocessing: str = "default") -> np.ndarray:
    """Decodifica bytes de imagen, redimensiona y preprocesa segun el modelo.

    Maneja PNGs con canal alfa (iconos de piezas) compositeando sobre
    fondo blanco antes de convertir a RGB.
    """
    img = Image.open(io.BytesIO(img_bytes))

    # Compositar sobre fondo blanco si tiene canal alfa
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        if img.mode in ("RGBA", "LA"):
            bg.paste(img, mask=img.split()[-1])
        else:
            bg.paste(img)
        img = bg
    else:
        img = img.convert("RGB" if channels == 3 else "L")

    img = img.resize((img_size[1], img_size[0]), Image.LANCZOS)  # (W, H)
    arr = np.array(img, dtype="float32")

    if preprocessing == "mobilenet_v2":
        # MobileNetV2 espera [-1, 1]
        arr = (arr / 127.5) - 1.0
    else:
        # Normalizacion estandar [0, 1]
        arr = arr / 255.0

    if channels == 1:
        arr = arr[..., np.newaxis]
    return arr[np.newaxis, ...]  # (1, H, W, C)


# ── Endpoints ────────────────────────────────────────────────

@app.route("/status", methods=["GET"])
def status():
    """Devuelve el estado actual del servidor."""
    return jsonify({
        "server":      "online",
        "model_loaded": STATE["model"] is not None,
        "model_name":  STATE["model_name"],
        "num_classes": STATE["num_classes"],
        "img_size":    list(STATE["img_size"]),
        "labels":      {str(k): v for k, v in STATE["labels"].items()} if STATE["labels"] else None,
    })


@app.route("/load_model", methods=["POST"])
def load_model():
    """
    Recibe el archivo .keras codificado en base64 y lo carga en memoria.
    Body JSON: { "filename": "modelo.keras", "data": "<base64>" }
    """
    payload = request.get_json(force=True)
    filename = payload.get("filename", "modelo.keras")
    b64data  = payload.get("data", "")

    if not b64data:
        return jsonify({"ok": False, "error": "No se recibieron datos del modelo."}), 400

    try:
        model_bytes = base64.b64decode(b64data)
        # Guardar temporalmente — tempfile elige la ruta correcta en Windows, Mac y Linux
        import tempfile
        tmp_dir  = Path(tempfile.gettempdir())
        tmp_path = tmp_dir / filename
        tmp_path.write_bytes(model_bytes)

        STATE["model"]      = tf.keras.models.load_model(str(tmp_path))
        STATE["model_name"] = filename

        # Inferir img_size desde la forma de entrada del modelo
        input_shape = STATE["model"].input_shape  # (None, H, W, C) o (None, N)
        if len(input_shape) == 4:
            STATE["img_size"]  = (input_shape[1], input_shape[2])
            STATE["channels"]  = input_shape[3]
        elif len(input_shape) == 2:
            # Modelo con Flatten interno — usar labels.json si esta cargado
            n = input_shape[1]
            if STATE["labels"] and "img_size" in STATE["labels"]:
                pass  # ya configurado al cargar labels
            else:
                # Estimar tamano cuadrado
                side = int(round(n ** 0.5))
                STATE["img_size"] = (side, side)
                STATE["channels"] = 1

        STATE["num_classes"] = STATE["model"].output_shape[-1]

        tmp_path.unlink(missing_ok=True)
        return jsonify({
            "ok":          True,
            "model_name":  filename,
            "input_shape": list(input_shape),
            "num_classes": STATE["num_classes"],
            "img_size":    list(STATE["img_size"]),
        })

    except Exception as e:
        STATE["model"] = None
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/load_labels", methods=["POST"])
def load_labels():
    """
    Recibe labels.json codificado en base64.
    Body JSON: { "data": "<base64 de labels.json>" }
    """
    payload = request.get_json(force=True)
    b64data = payload.get("data", "")

    if not b64data:
        return jsonify({"ok": False, "error": "No se recibieron datos de etiquetas."}), 400

    try:
        raw   = base64.b64decode(b64data).decode("utf-8")
        parsed = json.loads(raw)

        labels = {}
        for k, v in parsed.items():
            if k.isdigit():
                labels[int(k)] = v

        STATE["labels"] = labels

        # Leer img_size si esta en el JSON
        if "img_size" in parsed:
            STATE["img_size"] = tuple(parsed["img_size"])
        if "channels" in parsed:
            STATE["channels"] = int(parsed["channels"])
        if "num_classes" in parsed:
            STATE["num_classes"] = int(parsed["num_classes"])
        if "preprocessing" in parsed:
            STATE["preprocessing"] = str(parsed["preprocessing"])

        return jsonify({
            "ok":         True,
            "num_classes": len(labels),
            "labels":     {str(k): v for k, v in labels.items()},
            "img_size":   list(STATE["img_size"]),
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/classify", methods=["POST"])
def classify():
    """
    Clasifica una imagen enviada como base64.
    Body JSON: { "image": "<base64 de la imagen>", "filename": "foto.png" }
    """
    if STATE["model"] is None:
        return jsonify({"ok": False, "error": "No hay modelo cargado."}), 400

    payload  = request.get_json(force=True)
    b64img   = payload.get("image", "")
    if not b64img:
        return jsonify({"ok": False, "error": "No se recibio imagen."}), 400

    try:
        img_bytes = base64.b64decode(b64img)
        tensor    = preprocess_image(img_bytes, STATE["img_size"], STATE["channels"],
                                      STATE.get("preprocessing", "default"))

        # Si el modelo espera vector plano
        if len(STATE["model"].input_shape) == 2:
            h, w, c = STATE["img_size"][0], STATE["img_size"][1], STATE["channels"]
            tensor  = tensor.reshape(1, h * w * c)

        scores = STATE["model"].predict(tensor, verbose=0)[0]

        results = []
        for i, score in enumerate(scores):
            label = STATE["labels"].get(i, f"Clase {i}") if STATE["labels"] else f"Clase {i}"
            results.append({"index": i, "label": label, "score": float(score)})

        results.sort(key=lambda x: -x["score"])

        return jsonify({
            "ok":         True,
            "prediction": results[0]["label"],
            "confidence": results[0]["score"],
            "all_scores": results,
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Servidor Clasificador de Imagenes")
    print(f"  http://{HOST}:{PORT}")
    print("=" * 55)
    print("  Endpoints disponibles:")
    print(f"    GET  /status")
    print(f"    POST /load_model   — carga un archivo .keras")
    print(f"    POST /load_labels  — carga labels.json")
    print(f"    POST /classify     — clasifica una imagen")
    print("=" * 55)
    print("  Abre clasificador.html en tu navegador.")
    print("  Presiona Ctrl+C para detener el servidor.")
    print()
    app.run(host=HOST, port=PORT, debug=False)
