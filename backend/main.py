"""
main.py
FastAPI backend for SaaS Métré Automatisé.

Extraction pipeline (hybrid):
  1. Rule-based / Regex on DXF text entities (fast, free, ~70% of cases)
  2. Vision LLM on PDF page images (accurate, ~$0.01–0.05 per page)
  3. RulesDB enrichment: merge kimiya, poids/ml from EN profile tables
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional, Dict
import uuid
import asyncio

TASKS_STORE: Dict[str, dict] = {}

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from engines.pdf_parser import PDFParser
from engines.vision_llm_engine import VisionLLMEngine, VisionResult, merge_tile_results
from engines.llamaparse_engine import LlamaParseEngine
from engines.text_llm_engine import TextLLMEngine

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Métré Automatisé API",
    description="Extract steel profiles from structural drawings using Vision AI",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singletons (initialized once at startup)
_parser: PDFParser | None = None
_vision: VisionLLMEngine | None = None
_llamaparse: LlamaParseEngine | None = None
_text_llm: TextLLMEngine | None = None


@app.on_event("startup")
async def startup():
    global _parser, _vision, _llamaparse, _text_llm
    _parser = PDFParser(dpi=int(os.getenv("RENDER_DPI", "200")))
    provider = os.getenv("VISION_PROVIDER", "claude")
    _vision = VisionLLMEngine(fallback=True)
    _llamaparse = LlamaParseEngine()
    _text_llm = TextLLMEngine(provider=provider)
    logger.info("Engines ready — provider: %s", provider)

from fastapi.responses import JSONResponse
from fastapi.requests import Request
import traceback

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
        headers={"Access-Control-Allow-Origin": "*"}
    )



# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ProfileOut(BaseModel):
    id: str
    designation: str
    type: str
    role: str
    length_m: float | None
    quantity: int
    zone: str
    confidence: float
    # Enriched from RulesDB
    masse_lineaire_kg_m: float | None = None
    poids_unitaire: Optional[float] = None
    poids_total_kg: float | None = None
    surface_peinture_m2: float | None = None


class ExtractionResponse(BaseModel):
    project: str
    filename: str
    pages_processed: int
    scale_detected: str | None
    drawing_type: str
    profiles: list[ProfileOut]
    unreadable_zones: list[str]
    warnings: list[str]
    provider_used: str
    total_weight_kg: float
    needs_review_count: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "provider": os.getenv("VISION_PROVIDER", "claude")}


@app.post("/extract", response_model=ExtractionResponse)
async def extract(
    file: UploadFile = File(..., description="PDF of the structural drawing"),
    project: str = Form(default="unknown", description="Project name"),
    scale_hint: str = Form(default="", description="Expected scale e.g. '1:50' (optional)"),
    pages: str = Form(default="all", description="'all' or comma-separated page numbers e.g. '1,2,3'"),
    mode: str = Form(default="vision", description="'vision' | 'regex' | 'hybrid'"),
):
    """
    Extract steel profiles from a structural drawing PDF.

    Modes
    -----
    - vision  : Vision LLM only (most accurate, higher cost)
    - regex   : Rule-based only (fast, free, lower accuracy on complex drawings)
    - hybrid  : Regex first; Vision LLM for low-confidence or missing results (recommended)
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # Save upload to temp file
    tmp_path = Path(f"/tmp/{file.filename}")
    content = await file.read()
    tmp_path.write_bytes(content)

    try:
        # Determine pages to process
        page_images = _parser.render_pages(str(tmp_path))
        if pages != "all":
            requested = {int(p.strip()) for p in pages.split(",")}
            page_images = [p for p in page_images if p.page_number in requested]

        if not page_images:
            raise HTTPException(status_code=400, detail="No valid pages found")

        context = {
            "project": project,
            "ref": file.filename,
            "scale_hint": scale_hint or "unknown",
        }

        all_results: list[VisionResult] = []

        # Agentic Zoning Architecture
        logger.info("Using VisionLLMEngine with Agentic Zoning.")
        _parser.dpi = 300 # Force high resolution for maximum precision
        page_images = _parser.render_pages(str(tmp_path))
        if pages != "all":
            requested = {int(p.strip()) for p in pages.split(",")}
            page_images = [p for p in page_images if p.page_number in requested]

        for page_img in page_images:
            # Pass 1: Detect zones using a lower resolution version to save tokens/time
            logger.info(f"Detecting zones for page {page_img.page_number}...")
            downscaled = page_img.image.copy()
            downscaled.thumbnail((1024, 1024))
            zones = _vision.detect_zones(downscaled)
            logger.info(f"Page {page_img.page_number} zones detected: {len(zones)}")

            zone_results = []
            img_w, img_h = page_img.image.size
            
            for z_idx, zone in enumerate(zones):
                zt = zone.get("zone_type", "unknown")
                bbox = zone.get("bbox_normalized", [0.0, 0.0, 1.0, 1.0])
                
                # Ensure bbox is valid
                if not isinstance(bbox, list) or len(bbox) != 4:
                    bbox = [0.0, 0.0, 1.0, 1.0]
                
                y_min, x_min, y_max, x_max = bbox
                box_px = (
                    int(x_min * img_w),
                    int(y_min * img_h),
                    int(x_max * img_w),
                    int(y_max * img_h)
                )
                
                # Add a small padding to the crop (5% overlap) to avoid cutting text
                padding_x = int(img_w * 0.05)
                padding_y = int(img_h * 0.05)
                box_px = (
                    max(0, box_px[0] - padding_x),
                    max(0, box_px[1] - padding_y),
                    min(img_w, box_px[2] + padding_x),
                    min(img_h, box_px[3] + padding_y)
                )
                
                logger.info(f"Processing zone {z_idx+1}/{len(zones)}: {zt} at {box_px}")
                crop_img = page_img.image.crop(box_px)
                
                ctx = context.copy()
                ctx["zone_type"] = zt
                
                res = _vision.analyze(crop_img, page_number=page_img.page_number, tile_index=z_idx, context=ctx)
                zone_results.append(res)

            if zone_results:
                merged = merge_tile_results(zone_results)
                all_results.append(merged)

        if not all_results:
            raise HTTPException(status_code=422, detail="No profiles extracted — check PDF and API keys")

        # Merge across pages
        all_profiles_raw = []
        all_warnings = []
        all_unreadable = []
        scale_detected = None
        drawing_type = "unknown"
        provider_used = "none"

        for r in all_results:
            all_profiles_raw.extend(r.profiles)
            all_warnings.extend(r.warnings)
            all_unreadable.extend(r.unreadable_zones)
            if r.scale_detected and not scale_detected:
                scale_detected = r.scale_detected
            if r.drawing_type != "unknown":
                drawing_type = r.drawing_type
            provider_used = r.provider_used

        # Enrich with RulesDB (masse linéaire from EN tables)
        profiles_out = [_enrich_profile(p) for p in all_profiles_raw]

        total_weight = sum(
            p.poids_total_kg for p in profiles_out if p.poids_total_kg is not None
        )
        needs_review = sum(1 for p in profiles_out if p.confidence < 0.7)

        return ExtractionResponse(
            project=project,
            filename=file.filename,
            pages_processed=len(all_results),
            scale_detected=scale_detected,
            drawing_type=drawing_type,
            profiles=profiles_out,
            unreadable_zones=list(set(all_unreadable)),
            warnings=list(set(all_warnings)),
            provider_used=provider_used,
            total_weight_kg=round(total_weight, 2),
            needs_review_count=needs_review,
        )

    finally:
        tmp_path.unlink(missing_ok=True)


from fastapi import Response


@app.post('/extract_async')
async def extract_async(
    file: UploadFile = File(...),
    mode: str = Form('vision'),
    pages: str = Form('all'),
    scale_hint: str = Form(''),
    project: str = Form(''),
    ref: str = Form('')
):
    task_id = str(uuid.uuid4())
    TASKS_STORE[task_id] = {'status': 'processing'}
    
    file_bytes = await file.read()
    filename = file.filename
    
    async def process_task():
        import tempfile
        from pathlib import Path
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(file_bytes)
                tmp_path = Path(tmp.name)
            
            try:
                context = {
                    'project': project,
                    'ref': filename,
                    'scale_hint': scale_hint or 'unknown',
                }

                logger.info("Using VisionLLMEngine with Agentic Zoning (Async).")
                _parser.dpi = 300
                page_images = _parser.render_pages(str(tmp_path))
                if pages != "all":
                    requested = {int(p.strip()) for p in pages.split(",")}
                    page_images = [p for p in page_images if p.page_number in requested]

                all_results = []
                for page_img in page_images:
                    logger.info(f"Detecting zones for page {page_img.page_number}...")
                    downscaled = page_img.image.copy()
                    downscaled.thumbnail((1024, 1024))
                    zones = _vision.detect_zones(downscaled)
                    
                    zone_results = []
                    img_w, img_h = page_img.image.size
                    
                    for z_idx, zone in enumerate(zones):
                        zt = zone.get("zone_type", "unknown")
                        bbox = zone.get("bbox_normalized", [0.0, 0.0, 1.0, 1.0])
                        if not isinstance(bbox, list) or len(bbox) != 4:
                            bbox = [0.0, 0.0, 1.0, 1.0]
                        
                        y_min, x_min, y_max, x_max = bbox
                        box_px = (
                            int(x_min * img_w),
                            int(y_min * img_h),
                            int(x_max * img_w),
                            int(y_max * img_h)
                        )
                        padding_x = int(img_w * 0.05)
                        padding_y = int(img_h * 0.05)
                        box_px = (
                            max(0, box_px[0] - padding_x),
                            max(0, box_px[1] - padding_y),
                            min(img_w, box_px[2] + padding_x),
                            min(img_h, box_px[3] + padding_y)
                        )
                        
                        crop_img = page_img.image.crop(box_px)
                        ctx = context.copy()
                        ctx["zone_type"] = zt
                        
                        res = _vision.analyze(crop_img, page_number=page_img.page_number, tile_index=z_idx, context=ctx)
                        zone_results.append(res)

                    if zone_results:
                        merged = merge_tile_results(zone_results)
                        all_results.append(merged)

                if not all_results:
                    TASKS_STORE[task_id] = {'status': 'error', 'detail': 'No profiles extracted'}
                    return

                all_profiles_raw = []
                all_warnings = []
                all_unreadable = []
                scale_detected = None
                drawing_type = 'unknown'
                provider_used = 'none'

                for r in all_results:
                    all_profiles_raw.extend(r.profiles)
                    all_warnings.extend(r.warnings)
                    all_unreadable.extend(r.unreadable_zones)
                    if r.scale_detected and not scale_detected:
                        scale_detected = r.scale_detected
                    if r.drawing_type != 'unknown':
                        drawing_type = r.drawing_type
                    provider_used = r.provider_used

                profiles_out = [_enrich_profile(p) for p in all_profiles_raw]
                total_weight = sum(p.poids_total_kg for p in profiles_out if p.poids_total_kg is not None)
                needs_review = sum(1 for p in profiles_out if p.confidence < 0.7)

                final_res = ExtractionResponse(
                    project=project,
                    filename=filename,
                    pages_processed=len(all_results),
                    scale_detected=scale_detected,
                    drawing_type=drawing_type,
                    profiles=profiles_out,
                    unreadable_zones=list(set(all_unreadable)),
                    warnings=list(set(all_warnings)),
                    provider_used=provider_used,
                    total_weight_kg=round(total_weight, 2),
                    needs_review_count=needs_review,
                )
                TASKS_STORE[task_id] = {'status': 'done', 'result': final_res.model_dump()}

            finally:
                tmp_path.unlink(missing_ok=True)
                
        except Exception as e:
            import traceback
            logger.error(f"Task {task_id} failed: {e}")
            TASKS_STORE[task_id] = {"status": "error", "detail": traceback.format_exc()}

    asyncio.create_task(process_task())
    return {'task_id': task_id}

@app.get('/extract_status/{task_id}')
async def extract_status(task_id: str):
    if task_id not in TASKS_STORE:
        raise HTTPException(status_code=404, detail='Task not found')
    return TASKS_STORE[task_id]


class ExportRequest(BaseModel):
    data: list[dict]

@app.post("/export/excel")
async def export_excel(req: ExportRequest):
    from engines.export_engine import ExportEngine
    excel_bytes = ExportEngine.to_excel(req.data)
    return Response(
        content=excel_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=metrai_export.xlsx"}
    )

@app.get("/profiles/catalog")
async def profile_catalog():
    """Return the EN profile table (masse linéaire kg/m) for reference."""
    return {"profiles": _RULES_DB}


# ---------------------------------------------------------------------------
# RulesDB — EN 10034 / EN 10279 masse linéaire (kg/m)
# ---------------------------------------------------------------------------
# Extend this dict with all profiles from the EN tables.
# Source: ArcelorMittal sections catalogue.

_RULES_DB: dict[str, float] = {
    # IPE
    "IPE 80": 6.00, "IPE 100": 8.10, "IPE 120": 10.40, "IPE 140": 12.90,
    "IPE 160": 15.80, "IPE 180": 18.80, "IPE 200": 22.40, "IPE 220": 26.20,
    "IPE 240": 30.70, "IPE 270": 36.10, "IPE 300": 42.20, "IPE 330": 49.10,
    "IPE 360": 57.10, "IPE 400": 66.30, "IPE 450": 77.60, "IPE 500": 90.70,
    "IPE 550": 106.0, "IPE 600": 122.0,
    
    # HEA
    "HEA 100": 16.70, "HEA 120": 19.90, "HEA 140": 24.70, "HEA 160": 30.40,
    "HEA 180": 35.50, "HEA 200": 42.30, "HEA 220": 50.50, "HEA 240": 60.30,
    "HEA 260": 68.20, "HEA 280": 76.40, "HEA 300": 88.30, "HEA 320": 97.60,
    "HEA 340": 105.0, "HEA 360": 112.0, "HEA 400": 125.0, "HEA 450": 140.0,
    "HEA 500": 155.0, "HEA 550": 166.0, "HEA 600": 178.0, "HEA 650": 190.0,
    "HEA 700": 204.0, "HEA 800": 224.0, "HEA 900": 252.0, "HEA 1000": 272.0,
    
    # HEB
    "HEB 100": 20.40, "HEB 120": 26.70, "HEB 140": 33.70, "HEB 160": 42.60,
    "HEB 180": 51.20, "HEB 200": 61.30, "HEB 220": 71.50, "HEB 240": 83.20,
    "HEB 260": 93.00, "HEB 280": 103.0, "HEB 300": 117.0, "HEB 320": 127.0,
    "HEB 340": 134.0, "HEB 360": 142.0, "HEB 400": 155.0, "HEB 450": 171.0,
    "HEB 500": 187.0, "HEB 550": 199.0, "HEB 600": 212.0, "HEB 650": 225.0,
    "HEB 700": 241.0, "HEB 800": 262.0, "HEB 900": 291.0, "HEB 1000": 314.0,
    
    # HEM
    "HEM 100": 41.80, "HEM 120": 52.10, "HEM 140": 63.20, "HEM 160": 76.20,
    "HEM 180": 88.90, "HEM 200": 103.0, "HEM 220": 117.0, "HEM 240": 157.0,
    "HEM 260": 172.0, "HEM 280": 189.0, "HEM 300": 238.0, "HEM 320": 245.0,

    # UPN
    "UPN 80": 8.64, "UPN 100": 10.60, "UPN 120": 13.40, "UPN 140": 16.00,
    "UPN 160": 18.80, "UPN 180": 22.00, "UPN 200": 25.30, "UPN 220": 29.40,
    "UPN 240": 33.20, "UPN 260": 37.90, "UPN 280": 41.80, "UPN 300": 46.20,
    "UPN 320": 59.50, "UPN 350": 60.60, "UPN 380": 63.10, "UPN 400": 71.80,
    
    # UPE
    "UPE 80": 7.90, "UPE 100": 9.82, "UPE 120": 12.10, "UPE 140": 14.50,
    "UPE 160": 17.00, "UPE 180": 19.70, "UPE 200": 22.80, "UPE 220": 26.60,
    "UPE 240": 30.20, "UPE 270": 35.20, "UPE 300": 44.40, "UPE 330": 53.20,
    "UPE 360": 61.20, "UPE 400": 72.20,

    # COR / L (Les Angles/Cornières avec TOUTES les épaisseurs du Mémotech)
    "L 20*20*3": 0.88,
    "L 25*25*3": 1.12, "L 25*25*4": 1.46, "L 25*25*5": 1.79,
    "L 30*30*3": 1.36, "L 30*30*3.5": 1.57, "L 30*30*4": 1.78, "L 30*30*5": 2.18,
    "L 35*35*3.5": 1.84, "L 35*35*4": 2.09, "L 35*35*5": 2.57,
    "L 40*40*3": 1.83, "L 40*40*4": 2.42, "L 40*40*5": 2.97, "L 40*40*6": 3.52,
    "L 45*45*3": 2.07, "L 45*45*4": 2.72, "L 45*45*4.5": 3.06, "L 45*45*5": 3.38, "L 45*45*6": 4.00,
    "L 50*50*3": 2.31, "L 50*50*4": 3.04, "L 50*50*5": 3.77, "L 50*50*6": 4.47, "L 50*50*7": 5.15, "L 50*50*8": 5.82,
    "L 55*55*6": 4.94,
    
    # Reste des cornières (les grandes tailles et toutes leurs épaisseurs)
    "L 60*60*4": 3.66, "L 60*60*5": 4.54, "L 60*60*6": 5.42, "L 60*60*7": 6.26, "L 60*60*8": 7.09, "L 60*60*10": 8.76,
    "L 65*65*5": 4.95, "L 65*65*6": 5.89, "L 65*65*7": 6.81, "L 65*65*8": 7.72, "L 65*65*9": 8.62,
    "L 70*70*5": 5.33, "L 70*70*6": 6.38, "L 70*70*7": 7.38, "L 70*70*9": 9.32,
    "L 75*75*5": 5.72, "L 75*75*6": 6.85, "L 75*75*7": 7.93, "L 75*75*8": 8.99, "L 75*75*10": 11.07,
    "L 80*80*5": 6.11, "L 80*80*5.5": 6.75, "L 80*80*6": 7.34, "L 80*80*6.5": 7.92, "L 80*80*8": 9.63, "L 80*80*10": 11.86,
    "L 90*90*6": 8.30, "L 90*90*7": 9.61, "L 90*90*8": 10.90, "L 90*90*9": 12.18, "L 90*90*10": 13.45, "L 90*90*11": 14.70, "L 90*90*12": 15.93,
    "L 100*100*10": 15.10, "L 120*120*12": 21.60, "L 150*150*15": 33.80, "L 200*200*20": 59.90,

    # CORNIÈRES À AILES INÉGALES
    "L 80*40*6": 5.40, "L 80*40*8": 7.09, "L 80*60*7": 7.36, "L 80*60*8": 8.34, 
    "L 80*65*6": 6.58, "L 80*65*8": 8.66, "L 80*65*10": 10.68,
    "L 90*65*6": 7.05, "L 90*65*8": 9.29, "L 90*70*8": 9.60,
    "L 100*50*6": 6.81, "L 100*50*8": 8.97, "L 100*50*10": 11.07,
    "L 120*80*8": 12.16, "L 120*80*10": 15.02,
    "L 130*65*8": 11.85, "L 130*65*10": 14.62,
    "L 150*90*10": 18.18, "L 150*90*11": 18.90, "L 150*100*10": 18.98, "L 150*100*12": 22.56,
    "L 160*80*10": 18.20, "L 160*80*12": 21.62,
    "L 200*100*10": 22.95, "L 200*100*12": 27.32, "L 200*100*14": 31.62,
}


import re

def _enrich_profile(p: Any) -> ProfileOut:
    """Look up masse linéaire and compute poids total from RulesDB."""
    designation = p.designation.upper().strip()
    designation = designation.replace("CORNIÈRE", "").replace("CORNIERE", "").strip()
    
    # Format "IPE400" to "IPE 400" to match RulesDB
    designation = re.sub(r'^([A-Z]+)(\d+)', r'\1 \2', designation)
    
    # Fix 'X' or 'x' instead of '*' in L or 2L profiles (e.g. "L 60X6" -> "L 60*6")
    designation = re.sub(r'(L|2L)\s*(\d+)\s*[X]\s*(\d+)', r'\1 \2*\3', designation)
    
    masse = _RULES_DB.get(designation)
    # Fallback to check if it's L A*A*T
    if not masse and designation.startswith('L '):
        masse = _RULES_DB.get(designation.replace(' ', ''))
        if not masse:
            # handle L 60*6
            l_match = re.match(r'L\s*(\d+)\*(\d+)', designation)
            if l_match:
                masse = _RULES_DB.get(f"L {l_match.group(1)}*{l_match.group(1)}*{l_match.group(2)}")

    import math
    d_match = re.search(r'(?:D|ROND.*?)\s*(\d+)', designation, re.IGNORECASE)
    if d_match and not masse:
        d = float(d_match.group(1))
        masse = round(math.pi * (d**2) / 4000000 * 7850, 2)

    # TUBES Ronds et Carrés/Rectangulaires (Bulletproof Regex)
    tube_match = re.search(r'TUBE.*?(\d+(?:\.\d+)?)\s*[xX\*]\s*(\d+(?:\.\d+)?)(?:\s*[xX\*]\s*(\d+(?:\.\d+)?))?', designation, re.IGNORECASE)
    perimeter_m = None  # Pour la peinture
    
    if tube_match and not masse:
        val1 = float(tube_match.group(1))
        val2 = float(tube_match.group(2))
        val3 = tube_match.group(3)
        if val3:
            # Tube Rect / Carré: A x B x E
            a, b, e = val1, val2, float(val3)
            masse = round((a + b - 2*e) * e * 0.0157, 2)
            perimeter_m = 2 * (a + b) / 1000.0
        else:
            # Tube Rond: Dia x E
            d, e = val1, val2
            masse = round((d - e) * e * 0.02466, 2)
            perimeter_m = (math.pi * d) / 1000.0

    try:
        l_float = float(p.length_m) if p.length_m is not None else 0.0
    except:
        l_float = 0.0
    length_val = l_float if l_float > 0 else 1.0
    try:
        q_int = int(p.quantity) if getattr(p, 'quantity', None) is not None else 0
    except:
        q_int = 0
    qty_val = q_int if q_int > 0 else 1

    poids = None
    if masse is not None:
        poids = round(masse * length_val * qty_val, 2)
        
    # Check for PL, TN, PLATINE, GOUSSET, RAIDISSEUR A*B*C (Bulletproof Regex)
    pl_match = re.search(r'(?:PL|TN|PLAT|GOUSSET|RAID).*?(\d+(?:\.\d+)?)\s*[xX\*]\s*(\d+(?:\.\d+)?)\s*[xX\*]\s*(\d+(?:\.\d+)?)', designation, re.IGNORECASE)
    poids_unitaire = None
    surface_peinture = None
    
    if pl_match:
        a, b, c = map(float, pl_match.groups())
        # Volume en m3 * 8000 kg/m3 (Densité métier charpente)
        poids_unitaire = round((a * b * c / 1e9) * 8000, 3)
        poids = round(poids_unitaire * qty_val, 2)
        # Peinture pour les platines (les 2 faces principales)
        surface_peinture = round(2 * (a * b) / 1000000.0 * qty_val, 2)
    else:
        # Calcul de la surface de peinture pour les profilés (formules approchées Mémotech)
        if not perimeter_m:
            if "IPE" in designation:
                m = re.search(r'IPE\s*(\d+)', designation)
                if m:
                    h = float(m.group(1))
                    perimeter_m = (4 * h) / 1000.0 * 0.95
            elif "HE" in designation:
                m = re.search(r'HE[ABM]\s*(\d+)', designation)
                if m:
                    h = float(m.group(1))
                    b_val = min(h, 300.0)
                    perimeter_m = (2 * h + 4 * b_val) / 1000.0 * 0.95
            elif "UPN" in designation or "UPE" in designation:
                m = re.search(r'UP[NE]\s*(\d+)', designation)
                if m:
                    h = float(m.group(1))
                    b_val = h / 3.0 + 10
                    perimeter_m = (2 * h + 4 * b_val) / 1000.0 * 0.9
            elif "L " in designation:
                m = re.search(r'L\s*(\d+)\s*\*\s*(\d+)', designation)
                if m:
                    perimeter_m = 2 * (float(m.group(1)) + float(m.group(2))) / 1000.0
                    
        if perimeter_m and length_val > 0:
            surface_peinture = round(perimeter_m * length_val * qty_val, 2)

    out = ProfileOut(
        id=p.id,
        designation=designation, # Return the formatted one
        type=p.type,
        role=getattr(p, 'role', ''),
        length_m=length_val,
        quantity=qty_val,
        zone=p.zone,
        confidence=p.confidence,
        masse_lineaire_kg_m=masse,
        poids_unitaire=poids_unitaire,
        poids_total_kg=poids,
        surface_peinture_m2=surface_peinture,
    )
    return out
