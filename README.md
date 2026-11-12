# Clinical QA: Fine-Tuning Qwen3-4B with QLoRA, GPTQ Quantization & vLLM

**An end-to-end LLM post-training and inference optimization project for medical and biomedical question answering.**

[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/PardhuSSK/Qwen3-4B-Clinical-QA-GPTQ-INT4)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Training-orange)
![QLoRA](https://img.shields.io/badge/Fine--Tuning-QLoRA-green)
![GPTQ](https://img.shields.io/badge/Quantization-GPTQ%20INT4-purple)
![vLLM](https://img.shields.io/badge/Inference-vLLM-blueviolet)

## Why This Project?

General-purpose large language models demonstrate strong language understanding, but their performance can vary significantly when answering specialized medical questions.

Medical question answering presents several challenges:

- **Domain-specific knowledge:** Clinical terminology, medical reasoning, and biomedical research evidence require specialized understanding.
- **Structured prediction:** Medical QA benchmarks often require a single correct answer rather than an open-ended response.
- **Computational cost:** Full-parameter fine-tuning and inference can require substantial GPU memory.
- **Model quality preservation:** Quantization can reduce memory requirements but may also degrade accuracy.
- **Deployment efficiency:** A model that performs well during evaluation must also support practical inference workloads.

This project investigates whether **parameter-efficient supervised fine-tuning can improve a 4B-parameter language model's medical question-answering accuracy**, and how much of that improvement remains after 4-bit quantization.

Rather than stopping at fine-tuning, the project evaluates the model at multiple stages, quantifies accuracy regressions, and measures inference performance under concurrent requests.

**Research objective:**

> Fine-tune Qwen3-4B on MedQA and PubMedQA using QLoRA, compare it against the original pretrained model, evaluate the effects of GPTQ INT4 quantization, and benchmark inference using vLLM.

The project is intended for **research and educational purposes only**, not clinical diagnosis or medical decision-making.

---

## System Architecture

```text
                 Medical QA Datasets
                  MedQA + PubMedQA
                         |
                         v
                 Data Preparation
              Normalization + Filtering
              Chat-Format Construction
                         |
                         v
                 Qwen3-4B Base Model
                         |
                         v
                 QLoRA Fine-Tuning
               4-bit Model Loading
               LoRA Adapter Training
                         |
                         v
                Fine-Tuned Checkpoint
                         |
                         v
                  Model Evaluation
             Base vs Fine-Tuned Model
             MedQA / PubMedQA Accuracy
                         |
                         v
                   LoRA Merging
                         |
                         v
                  GPTQ Quantization
                  INT4 / Group 128
                         |
                         v
               Quantization Evaluation
              Accuracy Regression Analysis
                         |
                         v
                   vLLM Inference
                Concurrency Benchmarking
                Throughput / TTFT / Memory
                         |
                         v
                 Hugging Face Model Hub
                         |
                         v
                Gradio + FastAPI Web App
```

The core contribution of this project is the **model adaptation, controlled evaluation, and optimization workflow**. The web application demonstrates how the resulting model can be integrated into an inference-driven product.

---

## Headline Results

| Metric | Result | Evaluation |
|---|---|---|
| Base model macro-average accuracy | 50.5% | 200 questions |
| Fine-tuned model macro-average accuracy | **75.0%** | 200 questions |
| Fine-tuning improvement | **+24.5 percentage points** | Base → QLoRA |
| MedQA accuracy | 58% → **67%** | 100 questions |
| PubMedQA accuracy | 43% → **83%** | 100 questions |
| GPTQ INT4 macro-average accuracy | **72.5%** | 200 questions |
| GPTQ checkpoint size | **2.497 GiB** | Model artifacts |
| Maximum tested output throughput | **85.16 tokens/sec** | vLLM, concurrency 8 |
| Median TTFT at concurrency 1 | **45.73 ms** | vLLM benchmark |

**Key finding:** QLoRA fine-tuning increased macro-average accuracy from **50.5% to 75.0%**, an improvement of **24.5 percentage points** across the selected MedQA and PubMedQA evaluation sets.

GPTQ INT4 retained most of the overall improvement over the base model, although quantization introduced a measurable accuracy regression on MedQA.

All accuracy results above are calculated on selected evaluation samples, not the complete official benchmark test sets.

---

# 1. Dataset Preparation

Two medical question-answering datasets were used for supervised fine-tuning.

### MedQA

MedQA contains medical multiple-choice questions designed to evaluate domain knowledge and question-answering performance.

An example task:

**Question:** Which vitamin deficiency causes scurvy?

A. Vitamin A  
B. Vitamin B12  
C. Vitamin C  
D. Vitamin D  

**Expected answer:** C

For MedQA, the model is instructed to return a single answer label: `A`, `B`, `C`, or `D`.

### PubMedQA

PubMedQA focuses on biomedical research questions that require interpretation of associated scientific abstract evidence.

The model receives a research question and supporting context and predicts one of three labels:

- `yes`
- `no`
- `maybe`

This makes PubMedQA different from MedQA: rather than selecting from four medical answer options, the model must interpret biomedical evidence and produce a categorical conclusion.

## Dataset Processing Pipeline

The data preparation stage included:

1. Loading MedQA and PubMedQA datasets.
2. Normalizing dataset-specific fields into a consistent schema.
3. Preserving question text, available answer options, abstract context, and ground-truth labels.
4. Building supervised fine-tuning examples.
5. Formatting training inputs for Qwen3 chat-based inference.
6. Filtering examples to respect the configured sequence-length constraints.
7. Saving training and evaluation artifacts for subsequent notebooks.

### Dataset Summary

| Property | Value |
|---|---|
| Datasets | MedQA + PubMedQA |
| Initial prepared training examples | 12,000 |
| Final training examples | 11,999 |
| Training validation examples | 64 |
| Fine-tuning maximum sequence length | 1,024 |
| Evaluation selection | 100 MedQA + 100 PubMedQA |

Separate evaluation records were maintained so that model comparisons could use **the same questions across checkpoints**.

### Why Combine MedQA and PubMedQA?

The two datasets test related but different capabilities.

MedQA focuses on medical knowledge and multiple-choice answer selection.

PubMedQA emphasizes biomedical research questions and evidence-based categorical classification.

Combining them allowed us to explore whether a single fine-tuned model could learn both task formats while maintaining consistent, structured predictions.

---

# 2. Base Model and Fine-Tuning Strategy

### Base Model

**Model:** [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)

Qwen3-4B was selected as the foundation model for the experiment.

A 4B-parameter architecture provides a useful balance between model capacity and the computational resources available for practical fine-tuning experiments.

However, directly updating every parameter of a multi-billion-parameter model can be computationally expensive.

To address this limitation, the project uses **Quantized Low-Rank Adaptation (QLoRA)**.

## Why QLoRA?

QLoRA combines quantized base-model loading with Low-Rank Adaptation (LoRA).

Instead of updating the full pretrained weight matrices, LoRA introduces smaller trainable matrices that learn task-specific adjustments.

A simplified LoRA update is:

\[
W' = W + \Delta W
\]

where

\[
\Delta W = BA
\]

Here:

- \(W\) represents the original pretrained weight matrix.
- \(A\) and \(B\) are trainable low-rank matrices.
- The original pretrained weights remain frozen during LoRA training.

QLoRA further reduces memory requirements by loading the base model using low-bit quantization during training.

### Benefits of This Approach

- Reduces the number of trainable parameters.
- Makes fine-tuning a multi-billion-parameter model more practical.
- Preserves the original pretrained model weights.
- Produces compact adapter checkpoints.
- Allows trained adapters to be merged into the base model for later quantization and inference experiments.

An important distinction: **QLoRA's 4-bit loading during training and the later GPTQ INT4 quantization are separate stages.**

QLoRA makes adapter training memory-efficient. GPTQ creates a quantized inference checkpoint after training.

---

# 3. QLoRA Fine-Tuning Experiment

The model was fine-tuned using supervised learning on the combined medical QA dataset.

### Fine-Tuning Configuration

| Parameter | Configuration |
|---|---|
| Base model | Qwen3-4B |
| Training method | QLoRA |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Learning rate | 2e-4 |
| Maximum sequence length | 1,024 tokens |
| Training records | 11,999 |
| Validation records | 64 |
| Training epochs | 1 |
| Maximum optimization steps | 1,500 |
| GPU | NVIDIA A100 |
| Training duration | Approximately 169.65 minutes |

The fine-tuning stage trained adapter parameters while keeping the pretrained backbone frozen.

The model learned to map the medical question-answering inputs into their corresponding structured answer labels.

## Fine-Tuning Workflow

```text
Qwen3-4B Pretrained Model
           |
           v
Load Quantized Base Model
           |
           v
Attach LoRA Layers
     Rank = 16
     Alpha = 32
           |
           v
Prepare Medical QA Training Data
           |
           v
Supervised Fine-Tuning
     1,500 Training Steps
           |
           v
Validation and Loss Monitoring
           |
           v
Save Fine-Tuned LoRA Adapter
```

### Training Outcomes

| Metric | Value |
|---|---|
| Completed training steps | 1,500 |
| Final reported training loss | 0.4462 |
| Final validation loss | 0.59846 |
| Training status | Completed |
| Saved artifact | Fine-tuned LoRA adapter |

The final losses provide useful optimization diagnostics, but **loss alone does not establish whether the model became better at medical question answering**.

To assess real task performance, we evaluated the base and fine-tuned models on the same question sets.

---

# 4. Model Evaluation: Before vs After Fine-Tuning

This is the primary experimental result of the project.

The objective was to determine whether fine-tuning improved the model's ability to answer medical and biomedical questions, rather than relying only on training-loss reductions.

## Evaluation Methodology

We selected:

- 100 MedQA validation questions.
- 100 PubMedQA validation questions.

The same 200 questions were used for the base model and fine-tuned model.

Inference used deterministic generation settings and task-specific answer extraction.

For MedQA, outputs were normalized to `A`, `B`, `C`, or `D`.

For PubMedQA, outputs were normalized to `yes`, `no`, or `maybe`.

Accuracy was calculated by comparing the extracted prediction against the ground-truth label.

### Fine-Tuning Accuracy Results

| Dataset | Base Qwen3-4B | Fine-Tuned Qwen3-4B | Improvement |
|---|---:|---:|---:|
| MedQA | 58.0% | **67.0%** | **+9 pp** |
| PubMedQA | 43.0% | **83.0%** | **+40 pp** |
| Macro-average | 50.5% | **75.0%** | **+24.5 pp** |

*pp = percentage points.*

## Results Analysis

### MedQA: 58% → 67%

The fine-tuned model correctly answered 67 out of 100 selected medical multiple-choice questions, compared with 58 correct answers from the base model.

This represents a **9-percentage-point absolute improvement**.

The result suggests that supervised fine-tuning improved performance on the medical multiple-choice task, although the evaluation sample was relatively small.

### PubMedQA: 43% → 83%

The largest measured improvement occurred on PubMedQA.

The fine-tuned model answered 83 out of 100 selected biomedical research questions correctly, compared with 43 correct predictions from the base model.

This represents a **40-percentage-point absolute improvement**.

The result indicates substantial task-specific adaptation to the PubMedQA question format and evidence-based classification objective.

### Combined Performance: 50.5% → 75.0%

Across the two equally weighted evaluation sets, macro-average accuracy increased from **50.5% to 75.0%**.

That is an improvement of **24.5 percentage points** over the pretrained model.

The improvement was observed on the selected evaluation questions and should not be interpreted as clinical accuracy on real-world patient cases.

## Statistical Uncertainty

Paired bootstrap analysis was used to estimate uncertainty in the accuracy improvements.

| Dataset | Accuracy improvement | Bootstrap 95% interval |
|---|---:|---:|
| MedQA | +9 pp | +2 to +17 pp |
| PubMedQA | +40 pp | +29 to +51 pp |

These intervals summarize uncertainty under the project's evaluation and resampling procedure.

They do not replace evaluation on larger independent test sets.

## What We Learned from Fine-Tuning

Three findings stood out:

**1. Parameter-efficient fine-tuning produced measurable improvements.**

QLoRA enabled the model to adapt to medical QA tasks without full-parameter retraining.

**2. Improvement differed significantly by dataset.**

PubMedQA improved more than MedQA, demonstrating that adaptation gains are task-dependent.

**3. Evaluation must be separate from training diagnostics.**

A completed training run and reduced loss are not sufficient to claim better model performance. Controlled comparisons on identical questions provide stronger evidence.

---

# 5. GPTQ INT4 Quantization

After fine-tuning and evaluation, the next objective was to reduce the model's storage requirements for inference.

The project used **GPTQ**, a post-training quantization method, to convert the merged fine-tuned checkpoint into a 4-bit model.

## Quantization Workflow

```text
Fine-Tuned QLoRA Adapter
           +
Original Qwen3-4B Model
           |
           v
      Merge LoRA
           |
           v
   Merged BF16 Checkpoint
           |
           v
     GPTQ Calibration
           |
           v
   GPTQ 4-bit Quantization
           |
           v
   Standalone INT4 Checkpoint
           |
           v
   Accuracy Regression Testing
```

### Quantization Configuration

| Parameter | Value |
|---|---|
| Quantization library | GPTQModel |
| Quantization method | GPTQ |
| Weight precision | 4-bit |
| Group size | 128 |
| Calibration samples | 256 |
| Calibration sequence length | 512 |
| Final model checkpoint size | **2.497 GiB** |

The resulting checkpoint contains the merged and quantized model, allowing it to be served independently of the original LoRA adapter.

### Published Model

**[Qwen3-4B Clinical QA GPTQ INT4 — Hugging Face](https://huggingface.co/PardhuSSK/Qwen3-4B-Clinical-QA-GPTQ-INT4)**

---

# 6. Quantization Accuracy: Did GPTQ Preserve Fine-Tuning Gains?

Reducing precision is useful only if the resulting model maintains acceptable task performance.

We evaluated the quantized model on the same selected MedQA and PubMedQA questions used during fine-tuning evaluation.

### Three-Model Comparison

| Dataset | Base Model | Fine-Tuned Model | GPTQ INT4 Model |
|---|---:|---:|---:|
| MedQA | 58% | **67%** | 62% |
| PubMedQA | 43% | **83%** | 83% |
| Macro-average | 50.5% | **75.0%** | 72.5% |

## Quantization Trade-Offs

Compared with the fine-tuned model, GPTQ introduced the following accuracy changes:

| Metric | Change after GPTQ |
|---|---:|
| MedQA | -5 percentage points |
| PubMedQA | 0 percentage points |
| Macro-average | -2.5 percentage points |

GPTQ preserved the observed PubMedQA accuracy but reduced MedQA accuracy from 67% to 62%.

The project's original quality criterion allowed a maximum accuracy regression of 2 percentage points.

**The GPTQ model did not pass this quality gate.**

Both the MedQA regression (-5 pp) and macro-average regression (-2.5 pp) exceeded the allowed threshold.

Rather than ignoring this result, the project retained GPTQ INT4 as a **research and deployment-optimization candidate** and documented the trade-off.

### Key Observation

Although GPTQ reduced performance relative to the fine-tuned checkpoint, the quantized model still outperformed the original base model:

- MedQA: +4 percentage points over base.
- PubMedQA: +40 percentage points over base.
- Macro-average: +22 percentage points over base.

This demonstrates why optimization decisions should be based on measured quality and resource trade-offs, not checkpoint size alone.

---

# 7. vLLM Inference and Performance Benchmarking

After quantization, the model was served through **vLLM**, an inference engine designed for efficient large language model serving.

The purpose of this stage was to measure serving performance under different levels of concurrency.

## Inference Environment

| Property | Configuration |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB |
| GPU memory | Approximately 79.25 GiB |
| Inference engine | vLLM 0.33.0 |
| Model | Fine-tuned GPTQ INT4 |
| Model architecture | Qwen3ForCausalLM |
| Maximum model length | 2,048 tokens |
| Maximum concurrent sequences | 16 |
| GPU memory utilization target | 0.80 |
| Prefix caching | Enabled |
| Maximum generation length | 24 tokens |

The model was served through vLLM's OpenAI-compatible API.

The benchmarking workflow recorded throughput, time to first token, latency, generated tokens, and GPU memory measurements.

## Concurrency Benchmark

The experiment tested three concurrency settings:

- 1 concurrent request.
- 4 concurrent requests.
- 8 concurrent requests.

Each configuration used 16 clinical question-answering requests, balanced between MedQA and PubMedQA.

### Benchmark Results

| Concurrency | Output Throughput | Request Throughput | Median TTFT |
|---|---:|---:|---:|
| 1 | 39.66 tokens/sec | 19.83 requests/sec | 45.73 ms |
| 4 | 71.18 tokens/sec | 35.59 requests/sec | 98.12 ms |
| 8 | **85.16 tokens/sec** | **42.58 requests/sec** | 146.29 ms |

### Interpretation

Increasing concurrency improved aggregate serving throughput.

From concurrency 1 to concurrency 8:

- Output throughput increased from 39.66 to 85.16 tokens/sec.
- Request throughput increased from 19.83 to 42.58 requests/sec.
- Median time to first token increased from 45.73 to 146.29 ms.

This illustrates the trade-off between overall serving throughput and individual request responsiveness.

The benchmark results are specific to the selected workload and A100 environment.

They are not a measured comparison against the unquantized BF16 model, because a BF16 reference checkpoint was unavailable during that benchmarking session.

## Prefix Caching Experiment

The project also compared vLLM inference with prefix caching enabled and disabled.

At concurrency 4:

| Metric | Prefix Cache Disabled | Prefix Cache Enabled |
|---|---:|---:|
| Median TTFT | 104.11 ms | 98.12 ms |
| Output throughput | 68.28 tokens/sec | 71.18 tokens/sec |

The cache-enabled run showed modestly better measured performance.

However, this was a limited experiment, and cache-hit behavior was not independently measured. The results do not establish that prefix caching alone caused the differences.

---

# 8. What Worked, What Didn't, and Engineering Decisions

A major objective of the project was to document experimental outcomes accurately, including limitations and unsuccessful optimization attempts.

| Experiment | Observation | Decision |
|---|---|---|
| QLoRA fine-tuning | Macro accuracy improved from 50.5% to 75.0% | Successful fine-tuning checkpoint |
| MedQA adaptation | Accuracy improved by 9 pp | Retained improvement |
| PubMedQA adaptation | Accuracy improved by 40 pp | Retained improvement |
| GPTQ INT4 quantization | Produced a 2.497 GiB checkpoint | Published as quantized candidate |
| GPTQ quality evaluation | MedQA regressed by 5 pp | Quality gate not satisfied |
| vLLM concurrency scaling | Throughput increased with concurrency | Documented scaling results |
| Prefix caching | Modest differences in measured throughput and TTFT | Recorded without causal overclaim |
| BF16 vs GPTQ serving comparison | BF16 checkpoint unavailable during benchmark | No unsupported speedup claim |

## Main Engineering Lessons

**Fine-tuning performance must be measured independently.**

Training loss is useful for monitoring optimization, but downstream accuracy determines whether domain adaptation achieved its objective.

**Quantization is not automatically quality-preserving.**

A smaller checkpoint may improve deployment feasibility while introducing accuracy regressions that matter for the intended task.

**Inference throughput and response latency are different metrics.**

Increasing concurrency can improve aggregate tokens per second while increasing the time individual requests wait for their first generated token.

**Experiment limitations should remain visible.**

Smaller evaluation sets, missing reference checkpoints, and failed quality thresholds are important parts of a reproducible machine-learning project.

---

# 9. Reproducibility and Project Structure

The project is organized into five primary notebooks covering the complete experimental workflow.

```text
clinical-qa-finetuning/
│
├── notebooks/
│   ├── 01_Data_Preparation.ipynb
│   ├── 02_QLoRA_FineTuning.ipynb
│   ├── 03_Model_Evaluation.ipynb
│   ├── 04_GPTQ_Quantization.ipynb
│   └── 05_vLLM_Inference.ipynb
│
├── webapp/
│   └── Gradio + FastAPI application files
│
├── results/
│   └── Exported evaluation and benchmark results
│
├── .gitignore
└── README.md
```

*Notebook filenames above describe their roles; use the filenames in the published repository when reproducing the experiments.*

### Notebook Responsibilities

| Notebook | Purpose |
|---|---|
| 01 — Data Preparation | Load, normalize, filter, and prepare MedQA/PubMedQA training examples |
| 02 — QLoRA Fine-Tuning | Train Qwen3-4B adapters and save the fine-tuned checkpoint |
| 03 — Model Evaluation | Compare base and fine-tuned models on paired medical QA questions |
| 04 — GPTQ Quantization | Merge LoRA, quantize to INT4, and evaluate accuracy regression |
| 05 — vLLM Inference | Serve the quantized checkpoint and benchmark throughput, latency, and concurrency |

### Reproducing the Experiments

The recommended execution order is:

1. Prepare the medical QA datasets.
2. Fine-tune Qwen3-4B using QLoRA.
3. Evaluate base and fine-tuned checkpoints.
4. Merge the adapter and perform GPTQ quantization.
5. Evaluate the GPTQ checkpoint.
6. Serve the quantized model using vLLM.
7. Run controlled inference benchmarks.

The original experiments used Google Colab GPU environments and Google Drive to store intermediate artifacts.

Exact package dependencies, GPU availability, and data access requirements should be checked before rerunning the notebooks.

The GitHub repository should not contain large model weights, private credentials, or restricted datasets.

The published quantized checkpoint is available separately through Hugging Face.

---

# 10. Limitations and Future Improvements

The current results support task-specific improvements on selected benchmark samples, but additional work is required before broader conclusions can be drawn.

### Current Limitations

- Evaluation used 100 selected examples per dataset rather than the full official test sets.
- MedQA accuracy remains below 70% after fine-tuning.
- GPTQ introduced a measurable MedQA accuracy regression.
- The model produces short categorical answers rather than detailed, verified medical explanations.
- No clinical validation was performed using real patient cases.
- The project does not establish improved factual reliability beyond its measured benchmark tasks.
- The inference benchmarks were conducted on a particular GPU and workload.
- The web application requires a reachable inference backend to generate live predictions.

### Future Work

- Evaluate on larger independent medical QA test sets.
- Investigate error categories and incorrect predictions.
- Test alternative LoRA configurations and training strategies.
- Explore quantization calibration methods that better preserve MedQA accuracy.
- Compare BF16 and GPTQ inference on identical hardware and workloads.
- Measure GPU memory use, inference cost, and latency under sustained traffic.
- Add automated evaluation and regression checks.
- Investigate retrieval-augmented evidence grounding for biomedical questions.
- Deploy a persistent inference backend when suitable GPU hosting is available.

---

# 11. Clinical QA Web Application — Gradio + FastAPI

After completing fine-tuning, evaluation, quantization, and benchmarking, the project was extended with a web interface to demonstrate the model's integration into an application.

**The web application is a deployment and demonstration component; fine-tuning and model evaluation remain the primary focus of the project.**

## Application Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | Gradio | Interactive medical QA interface |
| Backend | FastAPI | API endpoints and request validation |
| Inference | vLLM | Model serving |
| Model | Qwen3-4B GPTQ INT4 | Medical QA predictions |
| Model storage | Hugging Face Hub | Published quantized checkpoint |
| Source control | GitHub | Code, notebooks, and documentation |
| Web hosting target | Render | Hosting the Gradio + FastAPI application |

### Application Architecture

```text
             User
               |
               v
        Gradio Web Interface
               |
               v
         FastAPI Backend
               |
               v
       vLLM Inference API
               |
               v
     Fine-Tuned Qwen3-4B Model
          GPTQ INT4
               |
               v
        Predicted Answer
               |
               v
        Gradio Interface
```

### Application Features

The application supports:

- **MedQA:** Medical multiple-choice question answering.
- **PubMedQA:** Biomedical research question classification.
- Task-specific answer extraction and formatting.
- Inference latency reporting.
- Model availability and health status.
- Example questions for interactive testing.
- Research-use and medical safety disclaimers.

### Running the Web Application Locally

From the `webapp` directory, install the application dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI application:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 7860
```

Then open:

**Gradio application:** http://127.0.0.1:7860/

**FastAPI documentation:** http://127.0.0.1:7860/docs

If the inference endpoint is unavailable, the application displays a model-offline status rather than reporting a fabricated prediction.

### Deployment Status

The Gradio + FastAPI application has been tested locally.

The quantized model has been published on Hugging Face.

Public deployment of the web application and provision of an always-available GPU inference endpoint are separate remaining tasks.

Hosting the web interface on a free CPU service does not automatically provide GPU resources for Qwen3-4B inference.

---

# 12. Model Repository and Acknowledgments

### Published Fine-Tuned Model

[**PardhuSSK/Qwen3-4B-Clinical-QA-GPTQ-INT4**](https://huggingface.co/PardhuSSK/Qwen3-4B-Clinical-QA-GPTQ-INT4)

### Technologies

**Model training:** Python, PyTorch, Hugging Face Transformers, PEFT, QLoRA

**Data processing:** Hugging Face Datasets, Pandas, NumPy

**Quantization:** GPTQModel

**Inference:** vLLM, OpenAI-compatible APIs

**Application:** FastAPI, Gradio

**Infrastructure:** Google Colab, NVIDIA A100, Hugging Face Hub, GitHub

### Acknowledgments

This project uses the Qwen3 base model and medical question-answering datasets including MedQA and PubMedQA.

It also builds on open-source libraries and research in parameter-efficient fine-tuning, post-training quantization, and efficient LLM inference.

Model weights and benchmark datasets remain subject to their respective upstream licenses and usage conditions.

---

## Final Project Outcome

This project demonstrates an end-to-end workflow for adapting and optimizing a pretrained language model for specialized medical question-answering tasks.

The principal result was a **24.5-percentage-point improvement in macro-average accuracy** after QLoRA fine-tuning, increasing performance from **50.5% to 75.0%** on the selected 200-question evaluation.

The subsequent GPTQ INT4 stage produced a **2.497 GiB quantized checkpoint**, preserving a macro-average accuracy of **72.5%**, while revealing a measurable quantization-related MedQA regression.

Finally, vLLM benchmarking demonstrated aggregate output throughput of **85.16 tokens/sec at concurrency 8** on an NVIDIA A100 GPU.

Together, these experiments demonstrate practical experience across:

**Dataset preparation → QLoRA fine-tuning → Controlled model evaluation → Quantization → Accuracy regression analysis → vLLM inference optimization → Application integration.**

**Disclaimer:** This is an experimental medical NLP research project. It is not intended for clinical diagnosis, treatment, or patient-specific medical advice.