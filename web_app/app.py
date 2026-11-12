"""Clinical QA: Gradio UI + FastAPI API backed by an external vLLM endpoint.

The application never loads model weights; it is suitable for CPU web hosting.
"""
import os
import re
import time
from typing import Literal

import gradio as gr
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_REPO = "PardhuSSK/Qwen3-4B-Clinical-QA-GPTQ-INT4"
MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "clinical-qa")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "").strip().rstrip("/")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "").strip()

SYSTEM = {
    "medqa": (
        "You are answering educational medical multiple-choice questions. "
        "Select the single best answer. Begin with A, B, C, or D."
    ),
    "pubmedqa": (
        "You answer biomedical research questions using the provided abstract evidence. "
        "Begin your answer with yes, no, or maybe."
    ),
}


class PredictionRequest(BaseModel):
    task: Literal["medqa", "pubmedqa"]
    question: str = Field(min_length=5, max_length=12000)


def parse_label(text: str, task: str) -> str | None:
    """Parse a label only if present at the beginning of the response."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.I | re.S).strip()
    if "<think>" in text.lower():
        return None
    if task == "medqa":
        match = re.match(r"^(?:answer\s*:\s*)?\(?([ABCD])\)?(?=$|[\s.):-])", text, flags=re.I)
        return match.group(1).upper() if match else None
    match = re.match(r"^(?:answer\s*:\s*)?(yes|no|maybe)\b", text, flags=re.I)
    return match.group(1).lower() if match else None


async def vllm_available() -> bool:
    """Check model availability; return False when not configured or unreachable."""
    if not VLLM_BASE_URL:
        return False
    try:
        headers = {"Authorization": f"Bearer {VLLM_API_KEY}"} if VLLM_API_KEY else {}
        async with httpx.AsyncClient(timeout=4) as client:
            resp = await client.get(f"{VLLM_BASE_URL}/models", headers=headers)
            resp.raise_for_status()
            return MODEL_NAME in [item.get("id") for item in resp.json().get("data", [])]
    except (httpx.HTTPError, ValueError, KeyError):
        return False


async def predict(task: str, question: str) -> dict:
    """Execute one inference request; never silently substitute another model."""
    if task not in SYSTEM:
        raise ValueError("Unsupported task")
    question = question.strip()
    if not 5 <= len(question) <= 12000:
        raise ValueError("Question must contain 5–12,000 characters")
    if not VLLM_BASE_URL:
        raise RuntimeError("Inference is offline: VLLM_BASE_URL is not configured.")

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM[task]},
            {"role": "user", "content": question},
        ],
        "temperature": 0,
        "max_tokens": 24,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Authorization": f"Bearer {VLLM_API_KEY}"} if VLLM_API_KEY else {}
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{VLLM_BASE_URL}/chat/completions", json=payload, headers=headers
            )
            response.raise_for_status()
        raw = (response.json()["choices"][0]["message"].get("content") or "").strip()
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
        raise RuntimeError("Inference server could not complete the request.") from exc

    return {
        "task": task,
        "prediction": parse_label(raw, task),
        "raw_response": raw,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "model": MODEL_NAME,
    }


app = FastAPI(
    title="Clinical QA — Research API",
    description="FastAPI + Gradio frontend for a Qwen3-4B Clinical QA vLLM endpoint",
    version="1.0.0",
)


@app.get("/health")
async def health():
    return {
        "web_app": "online",
        "inference": "online" if await vllm_available() else "offline",
        "model": MODEL_REPO,
    }


@app.post("/api/predict")
async def api_predict(request: PredictionRequest):
    try:
        return await predict(request.task, request.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


CSS = """
.gradio-container {max-width: 1160px !important; margin: auto !important;}
#hero {background: linear-gradient(120deg,#0f2940,#135f75); color: white;
       border-radius: 18px; padding: 30px 32px; margin-bottom: 20px;}
#hero h1, #hero p {color: white !important;}
#hero h1 {font-size: 30px; letter-spacing: -0.5px; margin: 0 0 8px;}
#hero p {opacity: 0.85; margin: 0;}
#answer-panel {border-radius: 14px;}
#submit-btn {border-radius: 10px; font-weight: 650;}
"""


async def check_status():
    ready = await vllm_available()
    if ready:
        return "🟢 Model connected — ready for questions"
    return (
        "🟠 Model offline — the web application is available, but GPU inference "
        "requires a reachable vLLM endpoint."
    )


async def answer(task: str, question: str):
    try:
        result = await predict(task, question)
        label = result["prediction"] or "Unparseable response"
        return (
            label,
            result["raw_response"],
            f'{result["latency_ms"]:,.1f} ms',
            "✅ Inference completed" if result["prediction"] else "⚠️ Invalid answer format",
        )
    except (RuntimeError, ValueError) as exc:
        return "—", "", "—", f"⚠️ {exc}"


with gr.Blocks(title="Clinical QA | Research Assistant") as demo:
    gr.HTML(f"<style>{CSS}</style>")
    gr.HTML(
        '<section id="hero"><h1>Clinical QA Assistant</h1>'
        '<p>Domain-adapted Qwen3-4B · QLoRA · GPTQ INT4 · vLLM</p></section>'
    )
    gr.Markdown(
        "**Research and educational use only.** Not medical advice. "
        "Do not enter personal or identifiable patient information."
    )
    with gr.Row():
        with gr.Column(scale=5):
            task = gr.Radio(
                [("MedQA · Multiple choice", "medqa"), ("PubMedQA · Research", "pubmedqa")],
                value="medqa",
                label="Question type",
            )
            question = gr.Textbox(
                label="Clinical question",
                placeholder="Enter a MedQA question with options A–D, or a PubMedQA question with abstract evidence...",
                lines=9,
                max_lines=15,
            )
            run = gr.Button("Generate answer", variant="primary", elem_id="submit-btn")
            gr.Examples(
                examples=[
                    ["medqa", "Which vitamin deficiency causes scurvy?\nA. Vitamin A\nB. Vitamin B12\nC. Vitamin C\nD. Vitamin D"],
                    ["pubmedqa", "Question: Does tobacco smoking increase lung cancer risk?\nContext: Epidemiological studies report a strong association between tobacco smoking and lung cancer."],
                ],
                inputs=[task, question],
                label="Try an example",
            )
        with gr.Column(scale=4, elem_id="answer-panel"):
            status = gr.Markdown("Checking model connection...")
            refresh = gr.Button("Refresh model status", size="sm")
            prediction = gr.Textbox(label="Predicted answer", interactive=False)
            raw = gr.Textbox(label="Model output", lines=5, interactive=False)
            latency = gr.Textbox(label="Request latency", interactive=False)
            message = gr.Markdown("")
            gr.Markdown(
                f"**Model:** [{MODEL_REPO}](https://huggingface.co/{MODEL_REPO})\n\n"
                "**Supported output:** A/B/C/D for MedQA, yes/no/maybe for PubMedQA."
            )
    run.click(answer, [task, question], [prediction, raw, latency, message])
    question.submit(answer, [task, question], [prediction, raw, latency, message])
    demo.load(check_status, outputs=status)
    refresh.click(check_status, outputs=status)

# Gradio is the public homepage; FastAPI endpoints remain accessible at /api/*.
app = gr.mount_gradio_app(app, demo, path="/")
