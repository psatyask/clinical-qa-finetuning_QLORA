# Clinical QA Web Application

A polished **Gradio + FastAPI** web application for a **Qwen3-4B Clinical QA GPTQ INT4** model served through an external **vLLM** GPU inference endpoint.

- [Published checkpoint](https://huggingface.co/PardhuSSK/Qwen3-4B-Clinical-QA-GPTQ-INT4)
- UI: `/`
- FastAPI request endpoint: `POST /api/predict`
- API documentation: `/docs`
- Health: `/health`

**Research and educational use only. Not for clinical diagnosis, treatment, or patient care.** Do not submit identifiable patient information.

## Architecture

```
Browser -> Gradio (Render CPU) -> FastAPI / Python inference client -> external vLLM GPU API -> published HF model
```

This repository deliberately **does not** load GPU model weights into the Render process. Hosting a web frontend is not the same as hosting model inference.

## Run locally with the existing Notebook 05 vLLM server

1. Start vLLM on the same computer as this web application (or use a reachable GPU host).
2. Install Python 3.11 and requirements:

```bash
python -m pip install -r requirements.txt
```

3. Set environment variables (Linux/macOS):

```bash
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_MODEL_NAME=clinical-qa
```

For Windows PowerShell:

```powershell
$env:VLLM_BASE_URL="http://127.0.0.1:8000/v1"
$env:VLLM_MODEL_NAME="clinical-qa"
```

4. Start the web application:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Open http://127.0.0.1:7860/ or http://127.0.0.1:7860/docs.

## API

```bash
curl -X POST http://127.0.0.1:7860/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"task":"medqa","question":"Which vitamin deficiency causes scurvy? A. Vitamin A B. Vitamin B12 C. Vitamin C D. Vitamin D"}'
```

Expected fields: `task`, `prediction`, `raw_response`, `latency_ms`, `model`.

## Deploy the web app to Render Free

1. Push these source files to a GitHub repository (without model weights or API keys).
2. In Render, create **New → Blueprint**, select the repo, and use `render.yaml`.
3. Set `VLLM_BASE_URL` to a **publicly reachable HTTPS** URL ending in `/v1` for your separate GPU server, if available. Set `VLLM_API_KEY` when your GPU server requires authentication.
4. Deploy. Render will expose the Gradio homepage at `/` and FastAPI at `/docs`.

**Important:** Render Free has no GPU for running this vLLM checkpoint. A Colab server bound to `127.0.0.1` cannot be accessed from Render; a privately authenticated, reachable inference service is required. If `VLLM_BASE_URL` is unset, the frontend and `/health` still work and accurately report **model offline**. There is no fake fallback model. Render Free may spin down after inactivity, and availability is subject to Render's free-tier limits.

## Model prompt behavior

- MedQA: predict exactly `A`, `B`, `C`, or `D`.
- PubMedQA: predict `yes`, `no`, or `maybe` from the supplied abstract context.
- Deterministic decoding (`temperature=0`), `max_tokens=24`, Qwen3 non-thinking mode.

## Security and limitations

- Do not commit `.env` files, tokens, or private patient data.
- Public inference endpoints must be authenticated, rate limited, and HTTPS-protected.
- The model can be medically incorrect; no clinical safety validation has been established.
- The public web UI alone does not imply that vLLM GPU inference is online.
