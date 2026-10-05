import os
import shutil
import base64
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Import the existing pipeline logic from GUI.py
from GUI import modify, make_preview_bytes

app = FastAPI(title="TIMEKEEPER Restoration API")

# Allow CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "output")
TEMP_INPUT_FOLDER = os.path.join(PROJECT_ROOT, "gui_temp_input")

def encode_image(img_path):
    """Read an image and return base64 encoded string"""
    if not os.path.exists(img_path):
        return None
    img = cv2.imread(img_path)
    if img is None:
        return None
    _, buffer = cv2.imencode('.png', img)
    return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

def encode_bytes(img_bytes):
    """Return base64 string from raw png bytes"""
    if not img_bytes:
        return None
    return f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"

@app.post("/restore")
async def restore_photo(
    file: UploadFile = File(...),
    scratchRemoval: bool = Form(False),
    colorization: bool = Form(True),
    highResolution: bool = Form(False),
    uncertainty: bool = Form(True),
    saturation: float = Form(1.0)
):
    try:
        # 1. Setup temp folder
        if os.path.exists(TEMP_INPUT_FOLDER):
            shutil.rmtree(TEMP_INPUT_FOLDER)
        os.makedirs(TEMP_INPUT_FOLDER, exist_ok=True)
        
        # 2. Save uploaded file
        file_path = os.path.join(TEMP_INPUT_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 3. Call the modify function
        def status_callback(msg):
            print(f"[API Status] {msg}")
            
        modify(
            image_filename=TEMP_INPUT_FOLDER,
            with_scratch=scratchRemoval,
            with_colorize=colorization,
            with_hr=highResolution,
            status_callback=status_callback,
            saturation=saturation
        )
        
        # 4. Gather results
        base_name = file.filename
        base_name_no_ext = os.path.splitext(base_name)[0]
        final_dir = os.path.join(OUTPUT_FOLDER, "final_output")
        
        candidates = [
            os.path.join(final_dir, f"{base_name_no_ext}.png"),
            os.path.join(final_dir, base_name),
        ]
        
        out_img_path = None
        for cand in candidates:
            if os.path.exists(cand):
                out_img_path = cand
                break
                
        if not out_img_path:
            return {"error": "Processing failed, output image not found."}
            
        # Result dictionary
        results = {
            "original": encode_image(file_path),
            "restored": encode_image(out_img_path),
            "colorized": None,
            "uncertainty": None,
            "confidence": None
        }
        
        if colorization:
            # The final image is colorized if colorization was selected
            results["colorized"] = results["restored"]
            
        # 5. Handle Uncertainty Map
        if uncertainty:
            try:
                from research.uncertainty import UncertaintyInferencer
                from research.calibration import UncertaintyCalibrator
                
                inferencer = UncertaintyInferencer("research_checkpoints/best_uncertainty_model.pth")
                calibrator = UncertaintyCalibrator.load("research_checkpoints/calibrator.pkl")
                
                orig_img = cv2.imread(file_path)
                out_img = cv2.imread(out_img_path)
                
                raw_unc = inferencer.infer(orig_img, out_img)
                calib_unc = calibrator.calibrate(raw_unc)
                confidence = np.clip(1.0 - (calib_unc / 255.0), 0.0, 1.0)
                
                raw_vis = ((raw_unc / max(1e-5, raw_unc.max())) * 255).astype(np.uint8)
                raw_vis = cv2.cvtColor(raw_vis, cv2.COLOR_GRAY2BGR)
                
                conf_vis = (confidence * 255).astype(np.uint8)
                conf_heatmap = cv2.applyColorMap(conf_vis, cv2.COLORMAP_JET)
                
                results["uncertainty"] = encode_bytes(make_preview_bytes(raw_vis))
                results["confidence"] = encode_bytes(make_preview_bytes(conf_heatmap))
            except Exception as e:
                print(f"Uncertainty generation failed: {e}")
                
        # Clean up
        if os.path.exists(TEMP_INPUT_FOLDER):
            shutil.rmtree(TEMP_INPUT_FOLDER, ignore_errors=True)
            
        return results

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
