#!/usr/bin/env bash
# setup_env.sh — Jorge Ruiz-Orera
# Create the orftracker mamba environment and install all dependencies.
# Conda-forge packages are installed via mamba; PyPI-only packages via pip.
#
# Usage:
#   bash setup_env.sh            # create env + install
#   bash setup_env.sh --update   # refresh packages in existing env

set -euo pipefail

ENV_NAME="orftracker"
PYTHON_VERSION="3.11"

### 1. Helpers
log()  { echo "[setup] $*"; }
warn() { echo "[warn]  $*" >&2; }
die()  { echo "[error] $*" >&2; exit 1; }

### 2. Prefer mamba, fall back to conda
if command -v mamba &>/dev/null; then
    CONDA_CMD="mamba"
elif command -v conda &>/dev/null; then
    CONDA_CMD="conda"
    warn "mamba not found -- using conda (slower). Install mamba for speed:"
    warn "  conda install -n base -c conda-forge mamba"
else
    die "Neither mamba nor conda found. Install miniforge first."
fi

log "Using: $CONDA_CMD"

### 3. Parse flags
UPDATE=0
for arg in "$@"; do
    [[ "$arg" == "--update" ]] && UPDATE=1
done

### 4. Create or update environment
if $CONDA_CMD env list | grep -qE "^${ENV_NAME}\s"; then
    if [[ $UPDATE -eq 0 ]]; then
        log "Environment '$ENV_NAME' already exists."
        log "Run with --update to refresh, or activate with: conda activate $ENV_NAME"
        exit 0
    fi
    log "Updating existing environment '$ENV_NAME'..."
else
    log "Creating environment '$ENV_NAME' (Python ${PYTHON_VERSION})..."
    $CONDA_CMD create -n "$ENV_NAME" python="${PYTHON_VERSION}" -y \
        --channel conda-forge
fi

CONDA_RUN="$CONDA_CMD run -n $ENV_NAME --no-capture-output"

### 5. Conda packages (conda-forge; compiled dependencies handled cleanly)
log "Installing conda packages..."
$CONDA_CMD install -n "$ENV_NAME" -y \
    --channel conda-forge \
    biopython \
    numpy \
    requests \
    pip

### 6. Pip packages (no conda-forge equivalent or version lags significantly)
log "Installing pip packages..."
$CONDA_RUN pip install --upgrade pip

## LangChain stack — langchain-classic backfills langchain.chains in >=1.0
$CONDA_RUN pip install \
    "langchain>=1.0.0" \
    "langchain-classic>=1.0.0" \
    "langchain-community>=0.3.0" \
    "langchain-ollama>=0.2.0" \
    "langchain-core>=1.0.0" \
    "langchain-text-splitters>=0.3.0"

## vector store
$CONDA_RUN pip install "faiss-cpu>=1.7.4"

## local embeddings — no API key needed
$CONDA_RUN pip install "sentence-transformers>=2.7.0"

### 7. Verify imports
log "Verifying installation..."
$CONDA_RUN python - << 'PYCHECK'
import sys

checks = [
    ("langchain_core",                       "langchain-core"),
    ("langchain_classic.chains.retrieval",   "langchain-classic"),
    ("langchain_community.vectorstores",     "langchain-community"),
    ("langchain_ollama",                     "langchain-ollama"),
    ("langchain_text_splitters",             "langchain-text-splitters"),
    ("faiss",                                "faiss-cpu"),
    ("sentence_transformers",                "sentence-transformers"),
    ("Bio",                                  "biopython"),
]

ok = True
for mod, pkg in checks:
    try:
        __import__(mod)
        print(f"  OK      {pkg}")
    except ImportError:
        print(f"  MISSING {pkg}")
        ok = False

if not ok:
    print("\nSome packages missing. Re-run: bash setup_env.sh --update", file=sys.stderr)
    sys.exit(1)

print("\nAll packages OK.")
PYCHECK

### 8. Ollama check (system-wide install, outside conda)
log "Checking Ollama..."
if command -v ollama &>/dev/null; then
    log "Ollama found: $(ollama --version 2>/dev/null || echo 'version unknown')"
    log "Pulled models:"
    ollama list 2>/dev/null | sed 's/^/  /' || true
else
    warn "Ollama not found. Install it with:"
    warn "  curl -fsSL https://ollama.com/install.sh | sh"
    warn "  # or without sudo (cluster):"
    warn "  curl -fsSL https://ollama.com/install.sh | OLLAMA_INSTALL_DIR=\$HOME/.local sh"
    warn "Then pull a model: ollama pull llama3.2"
fi

cat << DONE

============================================================
  Environment '$ENV_NAME' ready.

  Activate:
    conda activate $ENV_NAME

  Run:
    python orftracker.py "MOTS-c"
    python orftracker.py "humanin" --model llama3.1:8b

  Batch:
    python batch_orftracker.py example_genes.txt --outdir results/

  Start Ollama if needed:
    ollama serve &
    ollama pull llama3.2
============================================================
DONE
