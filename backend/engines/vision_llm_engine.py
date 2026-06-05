"""
vision_llm_engine.py
Multi-provider vision engine for steel profile detection on structural drawings.

Supports:
  - Google Gemini (gemini-1.5-pro-vision)
  - Anthropic Claude (claude-opus-4-6)  ← recommended for technical drawings

Provider selection via VISION_PROVIDER env variable ("gemini" | "claude").
Falls back to the other provider if the primary fails.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema (new — vision-specific)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are un ingénieur senior en charpente métallique avec plus de 20 ans d'expérience dans l'analyse de plans de fabrication et de montage (bureaux d'études marocains et français : Sinertech, BET BTP Maroc, OFPPT).
 
Tu analyses des images de plans PDF (DWG exportés en PDF) et tu extrais avec précision maximale tous les profilés de structure métallique.
 
DIFFÉRENCE FONDAMENTALE AVEC UN INGÉNIEUR HUMAIN :
Tu vois une IMAGE — pas un fichier CAO. Tu dois donc :
1. D'abord comprendre VISUELLEMENT ce que tu vois (quelle vue, quelle zone)
2. Ensuite lire les ANNOTATIONS TEXTUELLES sur les éléments
3. Enfin CROISER les informations entre vues avant de conclure
 
╔══════════════════════════════════════════════════════════════════════╗
║  ÉTAPE 0 — CARTOGRAPHIE DU PLAN (AVANT TOUT)                         ║
╚══════════════════════════════════════════════════════════════════════╝
 
Avant d'extraire quoi que ce soit, identifie et localise dans l'image chaque zone de dessin présente. Un plan contient typiquement :
 
  ┌─────────────────────────────────────────────┐
  │  VUE PRINCIPALE        │  ÉLÉVATION PIGNON  │
  │  (Plan de toiture ou   │  (vue de face)     │
  │   élévation long-pan)  │                    │
  ├────────────────────────┴────────────────────┤
  │  COUPES (AA, BB, PP, QQ...)                  │
  ├─────────────────────────────────────────────┤
  │  DÉTAILS (Dét.A, Dét.B, K, L, M...)         │
  ├─────────────────────────────────────────────┤
  │  VUE PLANCHER (Solivage)                    │
  └─────────────────────────────────────────────┘
 
→ Identifie chaque zone et note son type dans "views_identified".
→ Les DÉTAILS (Dét.A, K...) sont des zooms sur des assemblages. INTERDIT d'y extraire des profilés d'ossature (IPE, HEA), MAIS TU DOIS OBLIGATOIREMENT y extraire les PLATINES (TN), GOUSSETS, ÉCHANTIGNOLLES et TIGES D'ANCRAGE.
→ Les COUPES confirment les sections mais ne donnent pas les longueurs.
 
╔══════════════════════════════════════════════════════════════════════╗
║  VOCABULAIRE VISUEL — CE QUE CHAQUE FORME SIGNIFIE                   ║
╚══════════════════════════════════════════════════════════════════════╝
 
VUE LONG-PAN (élévation latérale) :
  Forme : rectangle vertical épais              → POTEAU (IPE, HEA, HEB)
  Forme : rectangle horizontal en haut          → SABLIÈRE (top chord / wall beam)
  Forme : rectangle horizontal intermédiaire    → PANNE (purlin)
  Forme : diagonale simple dans un panneau      → PALÉE DE STABILITÉ (bracing)
  Forme : deux diagonales en X qui se croisent  → CROIX DE SAINT-ANDRÉ = CONTREVENTEMENT (CVT)
  Forme : élément biseauté/triangle sous nœud   → JARRET (haunch)
 
VUE TOITURE (plan de dessus) :
  Forme : grandes poutres longitudinales        → TRAVERSE (IPE/HEA typ.)
  Forme : barres fines perpendiculaires         → LIERNE ou TIRANT (souvent rond D12/D14)
  Forme : grille régulière parallèle            → PANNES COURANTES
 
VUE PIGNON (élévation de face) :
  Forme : poteaux verticaux en façade           → POTEAU PIGNON
  Forme : petits poteaux intermédiaires         → POTELET (IPE/UAP typ.)
  Forme : grille horizontale/verticale dense    → LISSES + MONTANTS BARDAGE
 
VUE PLANCHER :
  Forme : Poutres principales supportant plancher → POUTRE (HEA/IPE)
  Forme : Poutrelles secondaires transversales   → SOLIVE (IPE typ.)
 
FERME EN TREILLIS (Truss - souvent en toiture) :
  Forme : barre périphérique supérieure inclinée→ ARBALÉTRIER ou MEMBRURE SUPÉRIEURE (souvent 2L)
  Forme : barre périphérique inférieure droite  → ENTRAIT ou MEMBRURE INFÉRIEURE (souvent 2L)
  Forme : barre verticale centrale              → POINÇON
  Forme : autres barres verticales              → MONTANT (souvent 2L ou Tube)
  Forme : barres diagonales dans le triangle    → DIAGONALE (souvent 2L ou Tube)
  Forme : barre supportant le tirant            → AIGUILLE
 
╔══════════════════════════════════════════════════════════════════════╗
║  CONVENTIONS MAROCAINES ET RÈGLES DE LECTURE (CRITIQUE)              ║
╚══════════════════════════════════════════════════════════════════════╝
 
1. LECTURE DES COTES (MÈTRES vs MILLIMÈTRES) :
   - Les cotes sans virgule (ex: 600, 4000) sont en millimètres.
   - Les cotes avec un point ou une virgule (ex: 5.000 ou 10,250) sont en MÈTRES.
   → Tu dois IMPÉRATIVEMENT les convertir en millimètres dans la sortie JSON (ex: 5.000 → 5000).
 
2. REPÈRES ET GESTION DES DOUBLONS :
   - Si le plan utilise des repères (B1, B2, P1, P2...), tu DOIS les utiliser comme "repere" dans le JSON pour grouper les éléments et NE PAS les compter deux fois si tu les vois dans une autre vue.
   - Si le plan n'a pas de repères (annotations directes comme "IPE 300"), tu DOIS INVENTER un repère unique (ex: P001) pour chaque élément physique distinct. Croise la longueur et la zone pour t'assurer que l'IPE 300 vu en plan de toiture n'est pas recompté dans la coupe AA.
 
3. FORMAT DES PROFILÉS SPÉCIAUX :
   - "2L 100x10" désigne une DOUBLE CORNIÈRE. Extraire le type comme "2L". L'app multipliera le poids par 2.
   - INTERDICTION FORMELLE d'utiliser la lettre "X" dans les cornières simples. Toujours écrire "L60*6" au lieu de "L 60X6".
 
╔══════════════════════════════════════════════════════════════════════╗
║  ÉTAPE 1 À 6 — PROCÉDURE D'EXTRACTION                                ║
╚══════════════════════════════════════════════════════════════════════╝
 
Étape 1 : Lire l'échelle dans le cartouche.
Étape 2 : Cross-validation. Un profilé doit être confirmé par au moins 2 sources.
 
Étape 3 : Ossature Principale
ATTENTION : Les POTEAUX et POUTRES PRINCIPALES sont souvent raccourcis ou invisibles sur le Plan de Toiture. Tu DOIS scanner l'ÉLÉVATION LONG-PAN et PIGNON pour les extraire.
 
Étape 4 : Éléments Secondaires
Scanner agressivement pour : JARRETS (triangles sous les nœuds), LISSES, CONTREVENTEMENTS (X), TIRANTS (Ronds Φ), SOLIVES.
 
Étape 5 : Boulonnerie, Platines, Goussets et Échantignolles
Chercher OBLIGATOIREMENT dans les Vues de DÉTAILS :
- Platines (TN) : extraire le label, mettre length_mm=null.
- Goussets : plaques de liaison aux nœuds des fermes en treillis.
- Échantignolles : équerres fixant les pannes sur les arbalétriers.
 
Étape 6 : Contrôle Anti-Erreur
- RÈGLE DES PANNES : Les pannes courent souvent sur toute la longueur du bâtiment (ex: 15m). Si le plan est divisé en travées de 4m, ne découpe pas la panne ! La longueur est 15000mm, JAMAIS la largeur de la travée (4000mm).
- Ne jamais confondre entraxe et longueur de pièce.
- Ne pas inventer les quantités (indiquer "×N travées" en note si nécessaire).
 
╔══════════════════════════════════════════════════════════════════════╗
║  TABLE DE RÉFÉRENCE — MASSE LINÉAIRE (kg/m)                          ║
╚══════════════════════════════════════════════════════════════════════╝
 
IPE: 80→6.0, 100→8.1, 120→10.4, 140→12.9, 160→15.8, 180→18.8, 200→22.4...
HEA: 100→16.7, 120→19.9, 140→24.7, 160→30.4, 180→35.5, 200→42.3...
UPN: 80→8.70, 100→10.6, 120→13.4, 140→16.0, 160→18.8, 180→22.0...
UAP: 80→8.38, 100→10.5, 130→13.7, 150→17.9, 175→21.2, 200→25.1
Cornières (L): L50*5→3.77, L60*6→5.42, L70*7→7.38, L80*8→9.63, L100*10→15.0
Doubles Cornières (2L): Multiplier le poids de la cornière simple par 2.
Ronds (D/ø): ø12→0.888, ø14→1.21, ø16→1.58, ø20→2.47, ø24→3.55
Tubes carrés: 40*40*2→2.31, 50*50*3→4.35, 60*60*4→6.97, 80*80*4→9.41
 
╔══════════════════════════════════════════════════════════════════════╗
║  FORMAT DE SORTIE — JSON UNIQUEMENT                                   ║
╚══════════════════════════════════════════════════════════════════════╝
 
{
  "scale_detected": "1:70",
  "scale_ratio": 70,
  "scale_confidence": 0.92,
  "drawing_type": "mixed | plan de toiture | élévation long-pan | coupe",
  "steel_grade": "S275JR",
 
  "views_identified": [...],
 
  "profiles": [
    {
      "id": "P001",
      "repere": "P1",
      "nomenclature": "POTEAU",
      "category": "ossature_principale",
      "type": "IPE",
      "designation": "IPE400",
      "length_mm": 4000,
      "length_source": "explicit_dimension",
      "quantity": 14,
      "quantity_note": null,
      "views_confirmed": ["élévation long-pan", "coupe PP"],
      "zone": "File 1 à 7 — long-pan",
      "masse_lineaire_kg_m": 66.3,
      "poids_unitaire_kg": 265.2,
      "poids_total_kg": 3712.8,
      "confidence": 0.92,
      "bbox_normalized": [0.12, 0.34, 0.45, 0.38]
    }
  ],
  "unreadable_zones": [
    "détail assemblage pied de poteau — annotations trop denses"
  ],
  "warnings": [
    "IPE450 traverse detected but length unclear",
    "Requires manual input: platines, goussets, boulonnerie"
  ]
}

CRITICAL RULES:
- Return ONLY the JSON object. No prose. No markdown. No backticks.
- length_m MUST be in meters (e.g., 4000 mm -> 4.0).
- role MUST be from the Visual Vocabulary (POTEAU, TRAVERSE, SABLIERE...).
- Détails d'assemblage: skip profile extraction, only list in unreadable_zones.
"""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class DetectedProfile:
    id: str
    type: str
    designation: str
    role: str
    length_m: float | None
    quantity: int
    zone: str
    confidence: float
    bbox_normalized: list[float] = field(default_factory=list)


@dataclass
class VisionResult:
    scale_detected: str | None
    scale_confidence: float
    profiles: list[DetectedProfile]
    unreadable_zones: list[str]
    warnings: list[str]
    drawing_type: str
    raw_response: str
    provider_used: str
    page_number: int = 1
    tile_index: int | None = None

    @property
    def high_confidence_profiles(self) -> list[DetectedProfile]:
        return [p for p in self.profiles if p.confidence >= 0.7]

    @property
    def needs_review(self) -> list[DetectedProfile]:
        return [p for p in self.profiles if p.confidence < 0.7]


# ---------------------------------------------------------------------------
# Provider enum
# ---------------------------------------------------------------------------

class VisionProvider(str, Enum):
    GEMINI = "gemini"
    CLAUDE = "claude"


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class VisionLLMEngine:
    """
    Detects steel profiles in structural drawing images using vision LLMs.

    Usage:
        engine = VisionLLMEngine()
        result = engine.analyze(pil_image, page_number=1)
        print(result.profiles)
    """

    def __init__(
        self,
        provider: VisionProvider | str | None = None,
        fallback: bool = True,
    ):
        env_provider = os.getenv("VISION_PROVIDER", "claude").lower()
        self.primary = VisionProvider(provider or env_provider)
        self.fallback_enabled = fallback
        self.fallback_provider = (
            VisionProvider.CLAUDE if self.primary == VisionProvider.GEMINI
            else VisionProvider.GEMINI
        )
        logger.info(f"VisionLLMEngine: primary={self.primary}, fallback={self.fallback_provider if fallback else 'disabled'}")

    def analyze(
        self,
        image: Image.Image,
        page_number: int = 1,
        tile_index: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> VisionResult:
        """
        Send an image to the vision model and return structured detections.

        context: optional metadata {"project": "...", "ref": "...", "scale_hint": "1:50"}
        """
        context = context or {}
        user_msg = self._build_user_message(context)

        try:
            raw = self._call_provider(self.primary, image, user_msg)
            provider_used = self.primary.value
        except Exception as e:
            logger.warning(f"Primary provider {self.primary} failed: {e}")
            if not self.fallback_enabled:
                raise
            logger.info(f"Falling back to {self.fallback_provider}")
            raw = self._call_provider(self.fallback_provider, image, user_msg)
            provider_used = self.fallback_provider.value

        return self._parse_response(raw, provider_used, page_number, tile_index)

    # ------------------------------------------------------------------
    # Pass 1: Cartography / Zoning
    # ------------------------------------------------------------------

    def detect_zones(self, image: Image.Image) -> list[dict]:
        """
        Pass 1: Detect drawing zones in the image using the Vision provider.
        Returns a list of dicts: {"zone_type": "...", "bbox_normalized": [ymin, xmin, ymax, xmax]}
        """
        prompt = """
        You are an AI assistant analyzing a structural steel drawing.
        Identify the distinct drawing zones (e.g., "plan de toiture", "élévation pignon", "détail assemblage", "coupe transversale").
        For each zone, provide its type and a normalized bounding box [ymin, xmin, ymax, xmax] where values are floats between 0.0 and 1.0.
        Return ONLY a JSON array of objects, e.g.:
        [
            {"zone_type": "plan de toiture", "bbox_normalized": [0.0, 0.0, 0.5, 1.0]},
            {"zone_type": "élévation pignon", "bbox_normalized": [0.5, 0.0, 1.0, 0.5]},
            {"zone_type": "détail assemblage", "bbox_normalized": [0.5, 0.5, 1.0, 1.0]}
        ]
        If the entire page is a single drawing or you cannot segment it clearly, return a single zone with [0.0, 0.0, 1.0, 1.0].
        """
        try:
            raw = self._call_provider(self.primary, image, prompt)
            clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            zones = json.loads(clean)
            if not isinstance(zones, list) or len(zones) == 0:
                zones = [{"zone_type": "full_page", "bbox_normalized": [0.0, 0.0, 1.0, 1.0]}]
            return zones
        except Exception as e:
            logger.warning(f"Failed to detect zones: {e}")
            return [{"zone_type": "full_page", "bbox_normalized": [0.0, 0.0, 1.0, 1.0]}]

    # ------------------------------------------------------------------
    # Provider dispatch
    # ------------------------------------------------------------------

    def _call_provider(
        self,
        provider: VisionProvider,
        image: Image.Image,
        user_message: str,
    ) -> str:
        if provider == VisionProvider.CLAUDE:
            return self._call_claude(image, user_message)
        elif provider == VisionProvider.GEMINI:
            return self._call_gemini(image, user_message)
        raise ValueError(f"Unknown provider: {provider}")

    # ------------------------------------------------------------------
    # Claude (Anthropic)
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_claude(self, image: Image.Image, user_message: str) -> str:
        import anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set")

        client = anthropic.Anthropic(api_key=api_key)
        img_b64 = _pil_to_base64(image)

        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_b64,
                            },
                        },
                        {"type": "text", "text": user_message},
                    ],
                }
            ],
        )
        return response.content[0].text

    # ------------------------------------------------------------------
    # Gemini (Google)
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_gemini(self, image: Image.Image, user_message: str) -> str:
        import requests
        import io
        import base64
        from engines.api_keys import get_random_gemini_key

        api_key = get_random_gemini_key()

        logger.info("Converting image to JPEG for Gemini API...")
        buf = io.BytesIO()
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        image.save(buf, format="JPEG", quality=80)
        b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={api_key}"
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": user_message},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": b64_data
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json"
            },
            "systemInstruction": {
                "parts": [{"text": SYSTEM_PROMPT}]
            }
        }

        logger.info("Sending request to Gemini API (raw REST)...")
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=120)
        
        if not resp.ok:
            logger.error(f"Gemini API failed: {resp.status_code} {resp.text}")
            resp.raise_for_status()

        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        raw: str,
        provider_used: str,
        page_number: int,
        tile_index: int | None,
    ) -> VisionResult:
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            data = json.loads(clean)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e}\nRaw: {raw[:500]}")
            return VisionResult(
                scale_detected=None,
                scale_confidence=0.0,
                profiles=[],
                unreadable_zones=["entire page — JSON parse failed"],
                warnings=[f"JSON parse error: {e}"],
                drawing_type="unknown",
                raw_response=raw,
                provider_used=provider_used,
                page_number=page_number,
                tile_index=tile_index,
            )

        profiles = [
            DetectedProfile(
                id=p.get("id", f"P{i:03d}"),
                type=p.get("type", "unknown"),
                designation=p.get("designation", ""),
                role=p.get("role", ""),
                length_m=p.get("length_m"),
                quantity=int(p.get("quantity", 1)),
                zone=p.get("zone", ""),
                confidence=float(p.get("confidence", 0.5)),
                bbox_normalized=p.get("bbox_normalized", []),
            )
            for i, p in enumerate(data.get("profiles", []))
        ]

        return VisionResult(
            scale_detected=data.get("scale_detected"),
            scale_confidence=float(data.get("scale_confidence", 0.0)),
            profiles=profiles,
            unreadable_zones=data.get("unreadable_zones", []),
            warnings=data.get("warnings", []),
            drawing_type=data.get("drawing_type", "unknown"),
            raw_response=raw,
            provider_used=provider_used,
            page_number=page_number,
            tile_index=tile_index,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_user_message(context: dict) -> str:
        lines = ["Analyze this structural steel drawing."]
        if context:
            lines.append("\nContext:")
            if "project" in context:
                lines.append(f"- Project: {context['project']}")
            if "ref" in context:
                lines.append(f"- Drawing ref: {context['ref']}")
            if "scale_hint" in context:
                lines.append(f"- Expected scale (from metadata): {context['scale_hint']}")
            if "drawing_type" in context:
                lines.append(f"- Drawing type: {context['drawing_type']}")
            if "zone_type" in context:
                lines.append(f"- Current Zone Type: {context['zone_type']}")
                zt = context["zone_type"].lower()
                if "toiture" in zt or "long-pan" in zt:
                    lines.append("CRITICAL INSTRUCTION: This is a main elevation or roof plan. Purlins (pannes) and wall beams (lisses) span the ENTIRE building length. DO NOT CHOP THEM into bay segments. Length is usually the total building length (e.g., 15000). Ignore column vertical heights here if they are confusing.")
                elif "élévation" in zt or "coupe" in zt or "pignon" in zt:
                    lines.append("CRITICAL INSTRUCTION: This is a section or elevation. Extract exact column heights and brace lengths here. Do not mistake building width for column height.")
                elif "détail" in zt:
                    lines.append("CRITICAL INSTRUCTION: This is a connection detail. Ignore main beams and columns. Focus strictly on extracting PLATINES (TN), GOUSSETS, RAIDISSEURS, and BOLTS.")
        lines.append("\nExtract all visible steel profiles and return the JSON format specified. Nothing else.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Merging tiles
# ---------------------------------------------------------------------------

def merge_tile_results(results: list[VisionResult]) -> VisionResult:
    """
    Consolidate results from multiple tiles of the same page.
    Deduplicates profiles by designation + zone, keeps highest confidence.
    """
    if not results:
        raise ValueError("No results to merge")
    if len(results) == 1:
        return results[0]

    # Use scale from the tile with highest scale_confidence
    best_scale = max(results, key=lambda r: r.scale_confidence)

    all_profiles: list[DetectedProfile] = []
    seen: dict[str, DetectedProfile] = {}

    for result in results:
        for profile in result.profiles:
            key = f"{profile.designation}|{profile.zone}"
            if key not in seen or profile.confidence > seen[key].confidence:
                seen[key] = profile

    all_profiles = list(seen.values())

    all_warnings = []
    all_unreadable = []
    for r in results:
        all_warnings.extend(r.warnings)
        all_unreadable.extend(r.unreadable_zones)

    return VisionResult(
        scale_detected=best_scale.scale_detected,
        scale_confidence=best_scale.scale_confidence,
        profiles=all_profiles,
        unreadable_zones=list(set(all_unreadable)),
        warnings=list(set(all_warnings)),
        drawing_type=results[0].drawing_type,
        raw_response="[merged from tiles]",
        provider_used=results[0].provider_used,
        page_number=results[0].page_number,
        tile_index=None,
    )


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------

def _pil_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")
