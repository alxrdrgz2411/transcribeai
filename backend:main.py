"""
TranscribeAI — Backend FastAPI
Endpoints: transcripción, resumen, traducción, exportación, créditos
"""

import os
import uuid
import json
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import openai

# ──────────────────────────────────────────────
# APP SETUP
# ──────────────────────────────────────────────
app = FastAPI(
    title="TranscribeAI API",
    description="API para transcripción, resumen y traducción de audio/video",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción: reemplaza con tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directorio temporal para archivos
TEMP_DIR = Path(tempfile.gettempdir()) / "transcribeai"
TEMP_DIR.mkdir(exist_ok=True)

# Cliente OpenAI (usa variable de entorno OPENAI_API_KEY)
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ──────────────────────────────────────────────
# MODELOS (schemas)
# ──────────────────────────────────────────────

class TranscribeYouTubeRequest(BaseModel):
    url: str
    language: str = "auto"         # "auto" o código ISO: "es", "en", "fr"...
    output_format: str = "text"    # "text" | "srt" | "timestamps"
    summarize: bool = False
    translate_to: Optional[str] = None  # código ISO del idioma destino

class TranscriptionResponse(BaseModel):
    job_id: str
    transcript: str
    summary: Optional[str] = None
    translated: Optional[str] = None
    detected_language: Optional[str] = None
    duration_seconds: Optional[float] = None
    credits_used: int
    credits_remaining: int

class CreditBalance(BaseModel):
    user_id: str
    credits: int
    plan: str

# ──────────────────────────────────────────────
# SIMULACIÓN DE BASE DE DATOS DE CRÉDITOS
# En producción: conecta Supabase / PostgreSQL / Redis
# ──────────────────────────────────────────────
CREDITS_DB: dict[str, dict] = {}

CREDIT_COSTS = {
    "transcription_per_minute": 1,   # 1 crédito por minuto de audio
    "summary": 2,                     # 2 créditos extra por resumen
    "translation": 3,                 # 3 créditos extra por traducción
}

PLANS = {
    "free":     {"credits": 30,  "price_usd": 0},
    "starter":  {"credits": 200, "price_usd": 9},
    "pro":      {"credits": 600, "price_usd": 19},
    "business": {"credits": 9999, "price_usd": 49},  # 9999 = ilimitado
}

def get_or_create_user(user_id: str) -> dict:
    if user_id not in CREDITS_DB:
        CREDITS_DB[user_id] = {
            "credits": 30,  # gratis al registrarse
            "plan": "free",
            "transactions": [],
        }
    return CREDITS_DB[user_id]

def deduct_credits(user_id: str, amount: int) -> tuple[bool, int]:
    """Retorna (éxito, créditos_restantes)"""
    user = get_or_create_user(user_id)
    if user["credits"] < amount:
        return False, user["credits"]
    user["credits"] -= amount
    user["transactions"].append({
        "type": "deduction",
        "amount": amount,
        "timestamp": datetime.utcnow().isoformat(),
    })
    return True, user["credits"]

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def extract_youtube_audio(url: str, output_path: Path) -> Path:
    """
    Descarga el audio de YouTube con yt-dlp.
    Instalar: pip install yt-dlp
    """
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "3",
        "--output", str(output_path / "%(id)s.%(ext)s"),
        "--no-playlist",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise HTTPException(
            status_code=422,
            detail=f"No se pudo descargar el video: {result.stderr[:200]}"
        )
    # Encontrar el archivo descargado
    files = list(output_path.glob("*.mp3"))
    if not files:
        raise HTTPException(status_code=500, detail="No se generó archivo de audio")
    return files[0]


def transcribe_with_whisper(
    audio_path: Path,
    language: Optional[str] = None,
    output_format: str = "text",
) -> dict:
    """
    Llama a la API de Whisper con el archivo de audio.
    Retorna: { transcript, detected_language, duration }
    """
    with open(audio_path, "rb") as f:
        kwargs = {
            "model": "whisper-1",
            "file": f,
            "response_format": "verbose_json",  # Incluye timestamps y metadata
        }
        if language and language != "auto":
            kwargs["language"] = language

        response = client.audio.transcriptions.create(**kwargs)

    # Formatear según el formato solicitado
    if output_format == "srt":
        transcript = _build_srt(response.segments or [])
    elif output_format == "timestamps":
        transcript = _build_timestamps(response.segments or [])
    else:
        transcript = response.text

    return {
        "transcript": transcript,
        "detected_language": getattr(response, "language", None),
        "duration_seconds": getattr(response, "duration", None),
        "segments": response.segments or [],
    }


def _build_srt(segments: list) -> str:
    """Genera formato SRT a partir de segmentos de Whisper"""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _seconds_to_srt_time(seg.start)
        end = _seconds_to_srt_time(seg.end)
        lines.append(f"{i}\n{start} --> {end}\n{seg.text.strip()}\n")
    return "\n".join(lines)


def _build_timestamps(segments: list) -> str:
    """Genera formato [HH:MM:SS] texto"""
    lines = []
    for seg in segments:
        ts = _seconds_to_hms(seg.start)
        lines.append(f"[{ts}] {seg.text.strip()}")
    return "\n".join(lines)


def _seconds_to_srt_time(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    ms = int((s % 1) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def _seconds_to_hms(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def summarize_transcript(transcript: str) -> str:
    """Genera un resumen inteligente con GPT-4o-mini"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un asistente especializado en resumir transcripciones. "
                    "Genera un resumen conciso (3-5 oraciones) en el mismo idioma de la transcripción. "
                    "Captura los puntos clave, decisiones importantes y conclusiones principales."
                ),
            },
            {"role": "user", "content": f"Transcripción:\n\n{transcript[:8000]}"},
        ],
        max_tokens=400,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def translate_text(text: str, target_language: str) -> str:
    """Traduce el texto al idioma destino usando GPT-4o-mini"""
    lang_names = {
        "es": "español", "en": "inglés", "fr": "francés",
        "pt": "portugués", "de": "alemán", "it": "italiano",
        "ja": "japonés", "zh": "chino mandarín", "ko": "coreano",
        "ar": "árabe", "ru": "ruso",
    }
    lang_name = lang_names.get(target_language, target_language)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    f"Traduce el siguiente texto al {lang_name}. "
                    "Mantén el formato original (párrafos, timestamps, SRT si aplica). "
                    "Solo devuelve la traducción, sin explicaciones."
                ),
            },
            {"role": "user", "content": text[:8000]},
        ],
        max_tokens=2000,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def estimate_credit_cost(duration_seconds: float, summarize: bool, translate: bool) -> int:
    """Calcula cuántos créditos costará la operación"""
    minutes = max(1, int(duration_seconds / 60) + (1 if duration_seconds % 60 > 0 else 0))
    cost = minutes * CREDIT_COSTS["transcription_per_minute"]
    if summarize:
        cost += CREDIT_COSTS["summary"]
    if translate:
        cost += CREDIT_COSTS["translation"]
    return cost


# ──────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────

@app.get("/")
async def health():
    return {"status": "ok", "service": "TranscribeAI API v1.0"}


@app.get("/credits/{user_id}", response_model=CreditBalance)
async def get_credits(user_id: str):
    """Obtener balance de créditos del usuario"""
    user = get_or_create_user(user_id)
    return CreditBalance(
        user_id=user_id,
        credits=user["credits"],
        plan=user["plan"],
    )


@app.post("/credits/add/{user_id}")
async def add_credits(user_id: str, plan: str):
    """
    Agrega créditos al comprar un plan.
    En producción: llamar desde webhook de Stripe después del pago.
    """
    if plan not in PLANS:
        raise HTTPException(status_code=400, detail=f"Plan inválido. Opciones: {list(PLANS.keys())}")
    user = get_or_create_user(user_id)
    user["credits"] += PLANS[plan]["credits"]
    user["plan"] = plan
    user["transactions"].append({
        "type": "purchase",
        "plan": plan,
        "credits_added": PLANS[plan]["credits"],
        "timestamp": datetime.utcnow().isoformat(),
    })
    return {"success": True, "credits": user["credits"], "plan": plan}


@app.post("/transcribe/file", response_model=TranscriptionResponse)
async def transcribe_file(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    output_format: str = Form("text"),
    summarize: bool = Form(False),
    translate_to: Optional[str] = Form(None),
    user_id: str = Form("demo_user"),
):
    """
    Transcribir un archivo de audio/video subido directamente.
    Formatos soportados: mp3, mp4, wav, m4a, mov, avi, webm, flac, ogg
    """
    # Validar tipo de archivo
    allowed = {".mp3", ".mp4", ".wav", ".m4a", ".mov", ".avi", ".webm", ".flac", ".ogg", ".mkv"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=415, detail=f"Formato no soportado: {ext}")

    # Validar tamaño (máx 500MB)
    MAX_SIZE = 500 * 1024 * 1024
    content = await file.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(status_code=413, detail="Archivo demasiado grande (máx 500MB)")

    job_id = str(uuid.uuid4())[:8]
    audio_path = TEMP_DIR / f"{job_id}{ext}"

    try:
        # Guardar archivo
        with open(audio_path, "wb") as f:
            f.write(content)

        # Verificar créditos (estimación inicial)
        user = get_or_create_user(user_id)
        if user["credits"] < 1:
            raise HTTPException(status_code=402, detail="Sin créditos disponibles. Recarga tu cuenta.")

        # Transcribir
        result = transcribe_with_whisper(
            audio_path,
            language=None if language == "auto" else language,
            output_format=output_format,
        )

        # Calcular costo real
        duration = result.get("duration_seconds") or 60
        credit_cost = estimate_credit_cost(duration, summarize, bool(translate_to))

        # Descontar créditos
        success, remaining = deduct_credits(user_id, credit_cost)
        if not success:
            raise HTTPException(
                status_code=402,
                detail=f"Créditos insuficientes. Necesitas {credit_cost}, tienes {user['credits']}."
            )

        transcript = result["transcript"]
        summary = None
        translated = None

        # Resumen IA
        if summarize:
            summary = summarize_transcript(transcript)

        # Traducción
        if translate_to and translate_to != result.get("detected_language"):
            translated = translate_text(transcript, translate_to)

        return TranscriptionResponse(
            job_id=job_id,
            transcript=transcript,
            summary=summary,
            translated=translated,
            detected_language=result.get("detected_language"),
            duration_seconds=result.get("duration_seconds"),
            credits_used=credit_cost,
            credits_remaining=remaining,
        )

    finally:
        # Limpiar archivo temporal
        if audio_path.exists():
            audio_path.unlink()


@app.post("/transcribe/youtube", response_model=TranscriptionResponse)
async def transcribe_youtube(body: TranscribeYouTubeRequest, user_id: str = "demo_user"):
    """
    Transcribir un video de YouTube por URL.
    Requiere yt-dlp instalado: pip install yt-dlp
    """
    # Validar URL básica
    if "youtube.com" not in body.url and "youtu.be" not in body.url:
        raise HTTPException(status_code=422, detail="URL de YouTube inválida")

    user = get_or_create_user(user_id)
    if user["credits"] < 1:
        raise HTTPException(status_code=402, detail="Sin créditos disponibles.")

    job_id = str(uuid.uuid4())[:8]
    job_dir = TEMP_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    try:
        # Descargar audio de YouTube
        audio_path = extract_youtube_audio(body.url, job_dir)

        # Transcribir
        result = transcribe_with_whisper(
            audio_path,
            language=None if body.language == "auto" else body.language,
            output_format=body.output_format,
        )

        # Costos
        duration = result.get("duration_seconds") or 60
        credit_cost = estimate_credit_cost(duration, body.summarize, bool(body.translate_to))

        success, remaining = deduct_credits(user_id, credit_cost)
        if not success:
            raise HTTPException(status_code=402, detail=f"Créditos insuficientes. Necesitas {credit_cost}.")

        transcript = result["transcript"]
        summary = None
        translated = None

        if body.summarize:
            summary = summarize_transcript(transcript)

        if body.translate_to and body.translate_to != result.get("detected_language"):
            translated = translate_text(transcript, body.translate_to)

        return TranscriptionResponse(
            job_id=job_id,
            transcript=transcript,
            summary=summary,
            translated=translated,
            detected_language=result.get("detected_language"),
            duration_seconds=result.get("duration_seconds"),
            credits_used=credit_cost,
            credits_remaining=remaining,
        )

    finally:
        # Limpiar
        import shutil
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)


@app.get("/export/{job_id}")
async def export_transcript(
    job_id: str,
    format: str = "txt",  # "txt" | "srt" | "pdf"
    content: str = "",
):
    """
    Exportar transcripción en el formato solicitado.
    Para PDF, instalar: pip install reportlab
    """
    if format == "pdf":
        return await _export_pdf(job_id, content)

    # TXT / SRT: devolver como texto plano
    ext = "srt" if format == "srt" else "txt"
    filename = f"transcripcion_{job_id}.{ext}"
    tmp_path = TEMP_DIR / filename
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)

    return FileResponse(
        path=tmp_path,
        filename=filename,
        media_type="text/plain; charset=utf-8",
    )


async def _export_pdf(job_id: str, content: str) -> FileResponse:
    """Genera un PDF con la transcripción usando reportlab"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.units import cm
        from reportlab.lib import colors
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="reportlab no instalado. Ejecuta: pip install reportlab"
        )

    filename = f"transcripcion_{job_id}.pdf"
    pdf_path = TEMP_DIR / filename

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                            leftMargin=2.5*cm, rightMargin=2.5*cm,
                            topMargin=2.5*cm, bottomMargin=2.5*cm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Heading1"],
                                  fontSize=20, spaceAfter=12,
                                  textColor=colors.HexColor("#0a0a0f"))
    body_style = ParagraphStyle("Body", parent=styles["Normal"],
                                 fontSize=11, leading=18,
                                 textColor=colors.HexColor("#333344"))
    meta_style = ParagraphStyle("Meta", parent=styles["Normal"],
                                  fontSize=9, textColor=colors.gray)

    story = [
        Paragraph("Transcripción — TranscribeAI", title_style),
        Paragraph(f"Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')}", meta_style),
        Spacer(1, 0.5*cm),
    ]

    for para in content.split("\n\n"):
        if para.strip():
            story.append(Paragraph(para.replace("\n", "<br/>"), body_style))
            story.append(Spacer(1, 0.3*cm))

    doc.build(story)
    return FileResponse(path=pdf_path, filename=filename, media_type="application/pdf")


# ──────────────────────────────────────────────
# RUN (desarrollo)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
