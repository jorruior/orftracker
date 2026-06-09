# orftracker

Local RAG pipeline for microprotein and uORF literature retrieval and summarisation.

Given a microprotein name, gene symbol, or peptide sequence, the pipeline queries PubMed, embeds the retrieved abstracts locally, and uses a local LLM to produce a structured summary covering source, proposed function, experimental approaches, and disease associations.

No API keys. No cloud. Only PubMed requires internet access.

| Component | Tool |
|-----------|------|
| LLM | [Ollama](https://ollama.com) (Llama 3, Mistral, Gemma 2, …) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS |
| Literature | PubMed via Biopython Entrez |

---

## Repository structure

```
orftracker/
├── orftracker.py          # main pipeline (single query)
├── batch_orftracker.py    # batch mode over a gene list
├── example_genes.txt      # example input
├── setup_env.sh           # environment setup script
├── environment.yml        # conda environment spec
└── README.md
```

---

## Installation

### 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

On a cluster without sudo:

```bash
curl -fsSL https://ollama.com/install.sh | OLLAMA_INSTALL_DIR=$HOME/.local sh
export PATH="$HOME/.local/bin:$PATH"   # add to ~/.bashrc
```

If home quota is tight, redirect model storage to scratch:

```bash
export OLLAMA_MODELS=/fast/AG_Huebner/$USER/ollama_models   # add to ~/.bashrc
```

### 2. Pull a model

```bash
ollama pull llama3.2        # 3B — fast, ~2 GB
ollama pull llama3.1:8b     # 8B — better reasoning, ~5 GB  ← recommended
ollama pull mistral         # 7B — strong structured output
ollama pull gemma2:9b       # 9B — good for science text
ollama pull biomistral      # 7B — biomedical-tuned
```

### 3. Create the Python environment

**Option A — automated (recommended):**

```bash
bash setup_env.sh           # creates 'orftracker' env
bash setup_env.sh --update  # refresh packages in existing env
```

**Option B — from spec:**

```bash
mamba env create -f environment.yml
conda activate orftracker
```

**Option C — into an existing env:**

```bash
conda activate myenv
pip install langchain langchain-classic langchain-community langchain-ollama \
            langchain-core langchain-text-splitters faiss-cpu sentence-transformers biopython
```

### 4. Start Ollama

```bash
ollama serve &
```

On a SLURM interactive node:

```bash
srun --pty --mem=16G --cpus-per-task=8 bash
ollama serve &
conda activate orftracker
python orftracker.py "MOTS-c"
```

### 5. Verify

```bash
conda activate orftracker
python orftracker.py --list-models
python orftracker.py "humanin"
```

---

## Usage

### Single query

```bash
python orftracker.py "MOTS-c"
python orftracker.py "humanin" --model llama3.1:8b --output humanin.md
python orftracker.py "BRAWNIN" --save-index indices/brawnin
python orftracker.py "BRAWNIN" --load-index indices/brawnin   # skip PubMed
python orftracker.py "MLGTVLVAVGAALVGMAVL"                    # peptide sequence
python orftracker.py --list-models
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | llama3.2:latest | Ollama model tag |
| `--ollama-url` | http://localhost:11434 | Ollama server URL |
| `--max-pubmed` | 30 | Max abstracts fetched from PubMed |
| `--top-k` | 6 | Chunks passed to LLM (≤6 for 3B; 8–10 for 8B+) |
| `--save-index` | — | Save FAISS index to directory |
| `--load-index` | — | Load FAISS index, skip PubMed |
| `--email` | user@mdc-berlin.de | NCBI Entrez email |
| `--output` | stdout | Write markdown summary to file |
| `--list-models` | — | List local Ollama models and exit |

### Batch mode

```bash
python batch_orftracker.py example_genes.txt --outdir results/

python batch_orftracker.py example_genes.txt \
    --model llama3.1:8b \
    --max-pubmed 40 \
    --top-k 8 \
    --outdir results/

python batch_orftracker.py example_genes.txt --no-cache   # skip FAISS caching
```

Already-processed genes are skipped. FAISS indices are cached under `results/faiss_indices/<gene>/`.

### Python API

```python
from orftracker import run_orftracker

summary = run_orftracker(
    query="MOTS-c",
    model="llama3.1:8b",
    max_pubmed=30,
    top_k=8,
    save_index="indices/MOTS-c",
)
print(summary)
```

---

## Output

Each summary contains the following sections:

- **Overview** — brief introduction from the literature
- **Source & Expression** — tissue, organism, biotype (uORF / sORF / lncRNA / annotated)
- **Proposed Function** — molecular and cellular roles with evidence
- **Experimental Approaches** — CRISPR, KO, IP-MS, ribosome profiling, etc.
- **Disease Associations** — cancer, cardiac, metabolic, neurological, etc.
- **Key References** — first author + PMID per claim
- **Retrieved Sources** — clickable PubMed links

---

## Notes

**Model size vs. quality:**

| Model | Size | Notes |
|-------|------|-------|
| llama3.2 | 3B | Fast; good for quick queries |
| llama3.1:8b | 8B | Recommended balance |
| mistral | 7B | Strong structured output |
| gemma2:9b | 9B | Best science reasoning |
| biomistral | 7B | Biomedical vocabulary |

**Context window:** Use `--top-k 4` for 3B models if output is truncated. Raise to `--top-k 10` for 8B+.

**Biomedical embeddings:** Change `EMBED_MODEL` in `orftracker.py` to `"pritamdeka/S-PubMedBert-MS-MARCO"` for domain-specific retrieval.

**Adding local PDFs:**
```python
from langchain_community.document_loaders import PyPDFDirectoryLoader
extra = PyPDFDirectoryLoader("my_papers/").load()
# merge with PubMed docs before passing to build_vectorstore()
```
