# 🎙️ TranscribeAI

Plataforma de transcripción de audio/video con IA. Monetización por créditos (pago por uso).

## Stack técnico

| Capa | Tecnología |
|------|-----------|
| Frontend | HTML + JS vanilla (sin frameworks, fácil de desplegar) |
| Backend | Python + FastAPI |
| Transcripción | OpenAI Whisper API |
| Resumen/Traducción | OpenAI GPT-4o-mini |
| Descarga YouTube | yt-dlp |
| PDF export | ReportLab |
| Pagos | Stripe (pendiente integrar) |
| Base de datos | En memoria (dev) → PostgreSQL/Supabase (prod) |

---

## Instalación local (desarrollo)

### 1. Requisitos previos
- Python 3.11+
- pip
- Una API key de OpenAI (https://platform.openai.com/api-keys)

### 2. Backend

```bash
cd backend

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Edita .env y agrega tu OPENAI_API_KEY

# Iniciar servidor
python main.py
# → http://localhost:8000
# → Docs automáticas: http://localhost:8000/docs
```

### 3. Frontend

Abre `frontend/app.html` en tu navegador (o sirve con cualquier servidor estático):

```bash
cd frontend
python -m http.server 3000
# → http://localhost:3000/app.html
```

Ingresa tu API key de OpenAI en el panel izquierdo del app.

---

## Arquitectura de créditos

| Acción | Costo |
|--------|-------|
| 1 minuto de audio transcrito | 1 crédito |
| Resumen IA (GPT-4o-mini) | +2 créditos |
| Traducción | +3 créditos |

| Plan | Créditos | Precio |
|------|----------|--------|
| Gratis (registro) | 30 | $0 |
| Starter | 200 | $9 USD |
| Pro | 600 | $19 USD |
| Business | Ilimitado | $49/mes |

---

## Endpoints del API

```
GET  /                          # Health check
GET  /credits/{user_id}         # Balance de créditos
POST /credits/add/{user_id}     # Agregar créditos (webhook Stripe)
POST /transcribe/file           # Transcribir archivo subido
POST /transcribe/youtube        # Transcribir URL de YouTube
GET  /export/{job_id}           # Exportar en TXT/SRT/PDF
```

Documentación interactiva: `http://localhost:8000/docs`

---

## Despliegue en producción

### Backend (Railway / Render / Fly.io)

```bash
# Railway (recomendado — más fácil)
npm install -g @railway/cli
railway login
railway init
railway up
railway variables set OPENAI_API_KEY=sk-...
```

### Frontend (Vercel / Netlify / Cloudflare Pages)

```bash
# Vercel
npm install -g vercel
cd frontend
vercel
```

### Dominio personalizado
1. Compra dominio en Namecheap/GoDaddy
2. Apunta DNS a tu deployment
3. Actualiza `API_BASE` en `app.html` con tu URL de backend

---

## Próximos pasos para monetización

### Fase 1 — MVP (ya tienes esto)
- [x] Transcripción con Whisper
- [x] Resumen con GPT-4o-mini  
- [x] Traducción automática
- [x] Editor manual de transcripción
- [x] Exportar TXT / SRT / PDF
- [x] Sistema de créditos

### Fase 2 — Pagos reales
- [ ] Integrar Stripe Checkout
  - `pip install stripe`
  - Crear productos en Stripe Dashboard
  - Agregar webhook para acreditar al usuario
- [ ] Autenticación de usuarios (Supabase Auth)
- [ ] Dashboard de usuario (historial, facturación)

### Fase 3 — Crecimiento
- [ ] API pública para developers
- [ ] Integración con Zapier / Make
- [ ] Chrome Extension para transcribir cualquier video
- [ ] Bulk transcription (múltiples archivos)
- [ ] Speaker diarization (identificar quién habla)

---

## Integrar Stripe (guía rápida)

```python
# backend/main.py — agregar endpoint de checkout
import stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

@app.post("/checkout")
async def create_checkout(plan: str, user_id: str):
    prices = {
        "starter": "price_xxx",  # ID de tu precio en Stripe
        "pro": "price_yyy",
        "business": "price_zzz",
    }
    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        line_items=[{"price": prices[plan], "quantity": 1}],
        mode="payment",  # "subscription" para Business
        success_url=f"https://tuapp.com/success?user={user_id}",
        cancel_url="https://tuapp.com/pricing",
        metadata={"user_id": user_id, "plan": plan},
    )
    return {"checkout_url": session.url}

@app.post("/webhook/stripe")
async def stripe_webhook(request: Request):
    # Verificar firma y acreditar créditos al usuario
    ...
```

---

## Estimación de costos OpenAI

| Operación | Costo aproximado |
|-----------|-----------------|
| Whisper (por minuto) | $0.006 USD |
| GPT-4o-mini (resumen) | ~$0.001 USD |
| GPT-4o-mini (traducción) | ~$0.002 USD |
| **Total por transcripción 10 min** | **~$0.07 USD** |

Con créditos a $19 USD por 600 créditos ≈ $0.032/crédito → margen de ~4x.

---

## Soporte

Para dudas sobre la implementación, revisa los comentarios en el código fuente.
Cada función está documentada con su propósito y cómo extenderla.
