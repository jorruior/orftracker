#!/usr/bin/env python3
# orftracker.py — Jorge Ruiz-Orera
# RAG pipeline for microprotein / uORF literature using local LLMs via Ollama.
# No API keys required. PubMed abstracts are fetched, embedded locally with
# sentence-transformers, indexed in FAISS, and summarised by the LLM.

from __future__ import annotations

import json
import sys
import time
import urllib.request
from typing import Optional

### 1. LangChain imports (>=1.0 / langchain-classic)
## chains live in langchain_classic; core types in langchain_core
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

### 2. PubMed
from Bio import Entrez


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL  = "llama3.2:latest"
DEFAULT_OLLAMA = "http://localhost:11434"
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
# Biomedical alternative: "pritamdeka/S-PubMedBert-MS-MARCO"


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def _ollama_is_running(base_url: str = DEFAULT_OLLAMA) -> bool:
    try:
        urllib.request.urlopen(f"{base_url}/api/tags", timeout=3)
        return True
    except Exception:
        return False


def _ollama_list_models(base_url: str = DEFAULT_OLLAMA) -> list[str]:
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as r:
            data = json.loads(r.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def _check_ollama(model: str, base_url: str = DEFAULT_OLLAMA) -> None:
    ## exit with a clear message if Ollama is not reachable or model is missing
    if not _ollama_is_running(base_url):
        print(
            f"\n[ERROR] Ollama is not running at {base_url}.\n"
            "  Start it with:  ollama serve\n"
            "  Or on systemd:  systemctl --user start ollama\n"
        )
        sys.exit(1)

    available = _ollama_list_models(base_url)
    if model not in available:
        avail_str = "\n    ".join(available) if available else "(none pulled yet)"
        print(
            f"\n[ERROR] Model '{model}' not found in Ollama.\n"
            f"  Pull it with:  ollama pull {model}\n"
            f"  Available:\n    {avail_str}\n"
            f"  Recommended:\n"
            f"    ollama pull llama3.2        # 3B, fast\n"
            f"    ollama pull llama3.1:8b     # 8B, better reasoning\n"
            f"    ollama pull mistral         # 7B, strong instruction following\n"
            f"    ollama pull gemma2:9b       # 9B, good for science text\n"
        )
        sys.exit(1)

    print(f"[Ollama] Model '{model}' ready at {base_url}")


# ---------------------------------------------------------------------------
# PubMed retrieval
# ---------------------------------------------------------------------------

def _build_pubmed_query(query: str) -> str:
    ### 1. Detect sequence vs name/gene input
    ## PubMed cannot search peptide sequences; fall back to context-based query
    aa = set("ACDEFGHIKLMNPQRSTVWY")
    cleaned = query.strip().upper().replace("\n", "").replace(" ", "")
    is_sequence = len(cleaned) > 10 and sum(c in aa for c in cleaned) / len(cleaned) > 0.85

    if is_sequence:
        print(
            "[INFO] Sequence input detected -- PubMed does not support peptide "
            "sequence search. Falling back to a microprotein context query."
        )
        length = len(cleaned)
        return (
            f"(microprotein OR sORF OR uORF OR smORF OR micropeptide) "
            f"AND ({length}[TIAB] OR microprotein[TIAB])"
        )

    term = query.strip()
    context = (
        "(microprotein OR sORF OR uORF OR smORF OR micropeptide OR "
        '"small ORF" OR "upstream ORF" OR "non-canonical ORF")'
    )
    return f'("{term}"[TIAB] OR "{term}"[Gene Name]) AND {context}'


def fetch_pubmed_abstracts(
    query: str,
    max_results: int = 30,
    email: str = "user@mdc-berlin.de",
) -> list[Document]:
    ### 1. Search PubMed and fetch abstracts in batches of 10
    ## respects NCBI rate limit of 3 req/s with a 350 ms sleep between batches
    Entrez.email = email
    pubmed_query = _build_pubmed_query(query)
    print(f"[PubMed] Query: {pubmed_query}")

    handle = Entrez.esearch(db="pubmed", term=pubmed_query, retmax=max_results, sort="relevance")
    record = Entrez.read(handle)
    handle.close()
    ids = record["IdList"]
    print(f"[PubMed] Found {len(ids)} articles.")
    if not ids:
        return []

    docs: list[Document] = []
    for i in range(0, len(ids), 10):
        batch = ids[i : i + 10]
        handle = Entrez.efetch(
            db="pubmed", id=",".join(batch), rettype="abstract", retmode="xml"
        )
        records = Entrez.read(handle)
        handle.close()

        for article in records["PubmedArticle"]:
            med = article["MedlineCitation"]
            art = med["Article"]
            title = str(art.get("ArticleTitle", ""))

            abstract_obj = art.get("Abstract", {})
            abstract_parts = abstract_obj.get("AbstractText", [])
            abstract = (
                " ".join(str(p) for p in abstract_parts)
                if isinstance(abstract_parts, list)
                else str(abstract_parts)
            )
            if not abstract.strip():
                continue

            authors = art.get("AuthorList", [])
            author_str = "; ".join(
                f"{a.get('LastName', '')} {a.get('Initials', '')}".strip()
                for a in authors if "LastName" in a
            )
            journal     = art.get("Journal", {})
            journal_name = str(journal.get("Title", ""))
            pub_date    = journal.get("JournalIssue", {}).get("PubDate", {})
            year        = str(pub_date.get("Year", pub_date.get("MedlineDate", "n.d.")))
            pmid        = str(med["PMID"])

            docs.append(Document(
                page_content=f"TITLE: {title}\n\nABSTRACT: {abstract}",
                metadata={
                    "pmid":    pmid,
                    "title":   title,
                    "authors": author_str,
                    "journal": journal_name,
                    "year":    year,
                    "url":     f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                },
            ))
        time.sleep(0.35)

    print(f"[PubMed] Retrieved {len(docs)} abstracts with text.")
    return docs


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

def build_vectorstore(docs: list[Document]) -> FAISS:
    ### 1. Chunk abstracts and embed with sentence-transformers
    ## chunk_size=600 / overlap=80 fits comfortably within the LLM context window
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"[Embed] {len(docs)} docs -> {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.from_documents(chunks, embeddings)


def load_vectorstore(path: str) -> FAISS:
    ## reload a previously saved FAISS index to skip re-embedding
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def get_llm(model: str = DEFAULT_MODEL, base_url: str = DEFAULT_OLLAMA) -> OllamaLLM:
    ### 1. Initialise Ollama LLM
    ## num_ctx=4096 requested explicitly; default 2048 is tight for 6 chunks + prompt
    _check_ollama(model, base_url)
    return OllamaLLM(
        model=model,
        base_url=base_url,
        temperature=0.1,
        num_ctx=4096,
    )


# ---------------------------------------------------------------------------
# Summarisation prompt
# ---------------------------------------------------------------------------

### 1. Prompt template for structured microprotein summary
## {input} = query, {context} = retrieved chunks (filled by create_stuff_documents_chain)
SUMMARY_PROMPT = ChatPromptTemplate.from_template(
"""You are an expert computational biologist specialising in microproteins,
small open reading frames (sORFs/uORFs), and translational regulation.

Your task: summarise the literature for the microprotein or gene "{input}".

RULES:
- Use ONLY information from the provided literature excerpts below.
- Do NOT use your prior training knowledge for specific claims.
- If a section has no evidence, write exactly: "Not reported in retrieved literature."
- Be concise. Avoid repetition.

OUTPUT FORMAT (keep these exact headings):

## Overview
1-2 sentences introducing the microprotein/gene.

## Source & Expression
Organism(s), tissue/cell type, genomic origin (uORF, sORF, lncRNA, annotated gene...).

## Proposed Function
Molecular and cellular roles with evidence from the abstracts.

## Experimental Approaches
Methods mentioned: CRISPR, KO, overexpression, IP-MS, ribosome profiling,
proteomics, reporter assays, structural studies, etc.

## Disease Associations
Diseases or phenotypes linked to this factor. Include cancer, cardiac, metabolic,
neurological, etc. if reported.

## Key References
Author et al., Year (PMID: XXXXX) for each key claim.

---
LITERATURE EXCERPTS:
{context}
---
Begin your structured summary now:
"""
)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_orftracker(
    query: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA,
    max_pubmed: int = 30,
    top_k: int = 6,
    save_index: Optional[str] = None,
    load_index: Optional[str] = None,
    email: str = "user@mdc-berlin.de",
) -> str:
    """
    Full pipeline: PubMed fetch -> chunk -> embed -> FAISS -> Ollama -> markdown.

    Parameters
    ----------
    query       : microprotein name, gene symbol, or amino-acid sequence
    model       : Ollama model tag (e.g. "llama3.2", "mistral", "gemma2:9b")
    base_url    : Ollama server URL
    max_pubmed  : max PubMed abstracts to retrieve
    top_k       : chunks passed to LLM (keep <=6 for 3B models; 8-10 for 8B+)
    save_index  : save FAISS index to this directory for future reuse
    load_index  : load existing FAISS index (skips PubMed fetch)
    email       : NCBI Entrez email
    """
    ### 1. Build or load vector store
    if load_index:
        print(f"[Index] Loading FAISS index from {load_index}")
        vs = load_vectorstore(load_index)
    else:
        docs = fetch_pubmed_abstracts(query, max_results=max_pubmed, email=email)
        if not docs:
            return (
                f"# {query}\n\n"
                "**No PubMed abstracts found.** "
                "Try a different name, check spelling, or broaden the search."
            )
        vs = build_vectorstore(docs)
        if save_index:
            vs.save_local(save_index)
            print(f"[Index] FAISS index saved to {save_index}")

    ### 2. Build RAG chain and generate summary
    ## create_retrieval_chain runs the retriever then passes top_k chunks to
    ## create_stuff_documents_chain which fills {context} in the prompt
    llm = get_llm(model=model, base_url=base_url)
    retriever = vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    combine_chain = create_stuff_documents_chain(llm, SUMMARY_PROMPT)
    chain         = create_retrieval_chain(retriever, combine_chain)

    print(f"[LLM] Generating summary with {model} (top_k={top_k})...")
    result  = chain.invoke({"input": query})
    summary = result["answer"]

    ### 3. Append deduplicated source list with PubMed links
    source_docs = result.get("context", [])
    if source_docs:
        seen: set[str] = set()
        lines = ["\n\n---\n## Retrieved Sources\n"]
        for doc in source_docs:
            pmid = doc.metadata.get("pmid", "?")
            if pmid in seen:
                continue
            seen.add(pmid)
            lines.append(
                f"- **PMID {pmid}** -- "
                f"{doc.metadata.get('title', 'n/a')}. "
                f"{doc.metadata.get('authors', '')}. "
                f"*{doc.metadata.get('journal', '')}* "
                f"({doc.metadata.get('year', '')}). "
                f"<{doc.metadata.get('url', '')}>"
            )
        summary += "\n".join(lines)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="orftracker",
        description="orftracker -- local RAG pipeline for microprotein literature.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python orftracker.py "MOTS-c"
  python orftracker.py "humanin" --model mistral --output humanin.md
  python orftracker.py "BRAWNIN" --save-index indices/brawnin
  python orftracker.py "BRAWNIN" --load-index indices/brawnin
  python orftracker.py "MOTS-c" --list-models
        """,
    )
    parser.add_argument("query", nargs="?",
        help="Microprotein name, gene symbol, or amino-acid sequence.")
    parser.add_argument("--model",       default=DEFAULT_MODEL,
        help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--ollama-url",  default=DEFAULT_OLLAMA,
        help=f"Ollama server URL (default: {DEFAULT_OLLAMA})")
    parser.add_argument("--max-pubmed",  type=int, default=30,
        help="Max PubMed abstracts (default: 30)")
    parser.add_argument("--top-k",       type=int, default=6,
        help="Chunks passed to LLM (default: 6; use 8-10 for 8B+ models)")
    parser.add_argument("--save-index",  default=None,
        help="Save FAISS index to directory")
    parser.add_argument("--load-index",  default=None,
        help="Load FAISS index from directory (skips PubMed)")
    parser.add_argument("--email",       default="user@mdc-berlin.de",
        help="NCBI Entrez email")
    parser.add_argument("--output",      default=None,
        help="Write summary to file (default: stdout)")
    parser.add_argument("--list-models", action="store_true",
        help="List locally available Ollama models and exit")

    args = parser.parse_args()

    if args.list_models:
        if not _ollama_is_running(args.ollama_url):
            print(f"[ERROR] Ollama not running at {args.ollama_url}.  Run: ollama serve")
            sys.exit(1)
        models = _ollama_list_models(args.ollama_url)
        print("Available Ollama models:")
        for m in (models or ["(none -- run: ollama pull llama3.2)"]):
            print(f"  {m}")
        sys.exit(0)

    if not args.query:
        parser.error("'query' is required unless --list-models is given.")

    summary = run_orftracker(
        query=args.query,
        model=args.model,
        base_url=args.ollama_url,
        max_pubmed=args.max_pubmed,
        top_k=args.top_k,
        save_index=args.save_index,
        load_index=args.load_index,
        email=args.email,
    )

    if args.output:
        with open(args.output, "w") as fh:
            fh.write(f"# {args.query}\n\n{summary}\n")
        print(f"[Done] Summary written to {args.output}")
    else:
        print("\n" + "=" * 72)
        print(summary)
        print("=" * 72)


if __name__ == "__main__":
    main()
