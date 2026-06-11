#!/usr/bin/env python3
# orftracker.py — Jorge Ruiz-Orera
# RAG pipeline for microprotein / uORF literature using local LLMs via Ollama.
# No API keys required. PubMed abstracts are fetched, embedded locally with
# sentence-transformers, indexed in FAISS, and summarised by the LLM.

from __future__ import annotations

import json
import re
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


# Defaults #

DEFAULT_MODEL  = "llama3.2:latest"
DEFAULT_OLLAMA = "http://localhost:11434"
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
# Biomedical alternative: "pritamdeka/S-PubMedBert-MS-MARCO"

## generic terms that are class labels, not specific microprotein names
_GENERIC_NAMES = {
    "microprotein", "micropeptide", "smorf", "sorf", "uorf", "upstream orf",
    "small orf", "non-canonical orf", "peptide", "protein", "orf", "transcript",
    "gene", "lncrna", "novel microprotein", "novel micropeptide", "novel smorf",
    "unknown", "uncharacterized", "putative microprotein",
}


# Ollama helpers #

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


# PubMed retrieval #

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


def _fetch_abstracts_for_query(
    pubmed_query: str,
    max_results: int,
    email: str,
) -> list[dict]:
    ### 1. Run esearch + efetch in batches of 10
    ## returns a flat list of abstract dicts with pmid, title, abstract, url
    Entrez.email = email
    handle = Entrez.esearch(
        db="pubmed", term=pubmed_query, retmax=max_results, sort="relevance"
    )
    record = Entrez.read(handle)
    handle.close()
    ids = record["IdList"]
    print(f"[PubMed] Found {len(ids)} articles.")
    if not ids:
        return []

    abstracts: list[dict] = []
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

            abstract_obj   = art.get("Abstract", {})
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
            journal      = art.get("Journal", {})
            journal_name = str(journal.get("Title", ""))
            pub_date     = journal.get("JournalIssue", {}).get("PubDate", {})
            year         = str(pub_date.get("Year", pub_date.get("MedlineDate", "n.d.")))
            pmid         = str(med["PMID"])

            abstracts.append({
                "pmid":    pmid,
                "title":   title,
                "abstract": abstract,
                "authors": author_str,
                "journal": journal_name,
                "year":    year,
                "url":     f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })
        time.sleep(0.35)  ## stay under NCBI rate limit

    return abstracts


def fetch_pubmed_abstracts(
    query: str,
    max_results: int = 30,
    email: str = "user@mdc-berlin.de",
) -> list[Document]:
    ## wrapper used by the RAG pipeline; returns LangChain Documents
    pubmed_query = _build_pubmed_query(query)
    print(f"[PubMed] Query: {pubmed_query}")
    abstracts = _fetch_abstracts_for_query(pubmed_query, max_results, email)
    print(f"[PubMed] Retrieved {len(abstracts)} abstracts with text.")

    return [
        Document(
            page_content=f"TITLE: {a['title']}\n\nABSTRACT: {a['abstract']}",
            metadata={k: a[k] for k in ("pmid", "title", "authors", "journal", "year", "url")},
        )
        for a in abstracts
    ]


# Vector store #

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


# LLM #

def get_llm(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA,
    temperature: float = 0.1,
) -> OllamaLLM:
    ### 1. Initialise Ollama LLM
    ## num_ctx=4096 requested explicitly; default 2048 is tight for 6 chunks + prompt
    _check_ollama(model, base_url)
    return OllamaLLM(
        model=model,
        base_url=base_url,
        temperature=temperature,
        num_ctx=4096,
    )


# Catalog mode — shared helpers #

def _parse_llm_json(raw: str) -> list:
    ### 1. Strip markdown fences and parse JSON
    ## some models wrap output in ```json ... ``` despite instructions
    cleaned = re.sub(r"^```(?:json)?", "", raw.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


def _is_generic_name(name: str) -> bool:
    ## reject class labels, single characters, and overly short strings
    if len(name) < 3:
        return True
    return name.lower().strip() in _GENERIC_NAMES


def _update_catalog(
    catalog: dict[str, dict],
    records: list,
    entry: dict,
) -> None:
    ### 1. Merge extracted records into the running catalog
    ## flatten nested lists (some models return [[{...}]] instead of [{...}])
    if records and isinstance(records[0], list):
        records = [item for sublist in records for item in sublist]

    for rec in records:
        ## skip non-dict or malformed records
        if not isinstance(rec, dict):
            continue
        name    = (rec.get("name")     or "").strip()
        func    = (rec.get("function") or "").strip()
        species = (rec.get("species")  or "unknown").strip() or "unknown"
        context = (rec.get("context")  or "unknown").strip() or "unknown"
        if context not in ("aberrant", "normal", "unknown"):
            context = "unknown"
        if not name or _is_generic_name(name):
            continue
        key = name.lower()
        if key not in catalog:
            catalog[key] = {"name": name, "function": func, "species": species, "context": context, "pmids": [], "urls": []}
        ## keep the longest function description seen across papers
        if func and len(func) > len(catalog[key]["function"]):
            catalog[key]["function"] = func
        ## keep first non-unknown species seen; do not overwrite with unknown
        if catalog[key]["species"] == "unknown" and species != "unknown":
            catalog[key]["species"] = species
        ## aberrant takes priority over normal/unknown; normal over unknown
        if catalog[key]["context"] == "unknown" and context != "unknown":
            catalog[key]["context"] = context
        elif catalog[key]["context"] == "normal" and context == "aberrant":
            catalog[key]["context"] = "aberrant"
        ## accumulate unique PMIDs
        if entry["pmid"] not in catalog[key]["pmids"]:
            catalog[key]["pmids"].append(entry["pmid"])
            catalog[key]["urls"].append(entry["url"])


def _write_catalog_tsv(catalog: dict[str, dict], output: str, label: str) -> None:
    ## sort by paper count descending; best-characterised entries first
    rows = sorted(catalog.values(), key=lambda r: len(r["pmids"]), reverse=True)
    with open(output, "w") as fh:
        fh.write("name\tspecies\tcontext\tfunction\tpaper_count\tpmids\turls\n")
        for row in rows:
            pmids = ";".join(row["pmids"])
            urls  = ";".join(row["urls"])
            fh.write(
                f"{row['name']}\t{row['species']}\t{row['context']}\t{row['function']}\t"
                f"{len(row['pmids'])}\t{pmids}\t{urls}\n"
            )
    print(f"[Catalog] {len(rows)} {label} written to {output}")


def _run_catalog(
    pubmed_query: str,
    extraction_prompt: ChatPromptTemplate,
    model: str,
    base_url: str,
    max_pubmed: int,
    output: str,
    email: str,
    label: str,
) -> None:
    ### 1. Fetch abstracts
    print(f"[Catalog] PubMed query: {pubmed_query}")
    abstracts = _fetch_abstracts_for_query(pubmed_query, max_pubmed, email)
    print(f"[Catalog] Processing {len(abstracts)} abstracts with {model}...")

    ### 2. Per-abstract LLM extraction at temperature=0 for deterministic JSON
    ## lower temperature reduces hallucinated names not present in the abstract
    llm     = get_llm(model=model, base_url=base_url, temperature=0.0)
    chain   = extraction_prompt | llm
    catalog: dict[str, dict] = {}

    for idx, entry in enumerate(abstracts, 1):
        print(f"  [{idx}/{len(abstracts)}] PMID {entry['pmid']}", end="\r")
        text = f"TITLE: {entry['title']}\n\nABSTRACT: {entry['abstract']}"
        try:
            raw     = chain.invoke({"context": text})
            records = _parse_llm_json(raw)
        except Exception:
            ## skip abstracts where the model returns unparseable output
            continue
        _update_catalog(catalog, records, entry)

    print()  ## newline after \r progress

    ### 3. Write TSV
    _write_catalog_tsv(catalog, output, label)


# Catalog mode — microproteins #

## tighter PubMed query: requires a named/characterised microprotein in the abstract
MICROPROTEIN_QUERY = (
    '(microprotein[TIAB] OR micropeptide[TIAB] OR smORF[TIAB] OR '
    '"small open reading frame"[TIAB]) '
    "AND (function[TIAB] OR characterize[TIAB] OR characterise[TIAB] "
    "OR identified[TIAB] OR encodes[TIAB] OR encoded[TIAB] OR novel[TIAB])"
)

MICROPROTEIN_PROMPT = ChatPromptTemplate.from_template(
"""You are an expert in microproteins and small open reading frames (sORFs).

Read the abstract below and extract every SPECIFIC microprotein or micropeptide
that is the SUBJECT OF STUDY in this paper — i.e. a named or characterised entity,
not a general concept.

DO NOT extract:
- Generic class labels such as "microprotein", "sORF", "micropeptide", "uORF"
- Proteins that are merely mentioned as controls or background
- Anything without a specific name or identifier

For each specific microprotein found, return a JSON list where every element has:
  "name"     : specific name or identifier (e.g. "MOTS-c", "humanin", "BRAWNIN")
  "function" : one sentence on its function based solely on this abstract, or null
  "species"  : organism studied (e.g. "Homo sapiens", "Mus musculus"); use "unknown" if not stated — do NOT guess
  "context"  : "aberrant" if the microprotein is specifically associated with disease, pathogenic conditions, stress response, or aberrant translation; "normal" if it has a described function in healthy physiology; "unknown" if not clearly stated — do NOT guess

Return ONLY the JSON list. No markdown, no explanation.
If no specific microprotein is the subject of study, return: []

ABSTRACT:
{context}
"""
)


def fetch_microprotein_catalog(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA,
    max_pubmed: int = 200,
    output: str = "microprotein_catalog.tsv",
    email: str = "user@mdc-berlin.de",
) -> None:
    _run_catalog(
        pubmed_query=MICROPROTEIN_QUERY,
        extraction_prompt=MICROPROTEIN_PROMPT,
        model=model,
        base_url=base_url,
        max_pubmed=max_pubmed,
        output=output,
        email=email,
        label="microproteins",
    )


# Catalog mode — uORFs #

## focused on regulatory uORFs and upstream ORF studies
UORF_QUERY = (
    '(uORF[TIAB] OR "upstream ORF"[TIAB] OR "upstream open reading frame"[TIAB]) '
    "AND (function[TIAB] OR regulation[TIAB] OR translation[TIAB] "
    "OR characterize[TIAB] OR characterise[TIAB] OR repression[TIAB] "
    "OR identified[TIAB] OR novel[TIAB])"
)

UORF_PROMPT = ChatPromptTemplate.from_template(
"""You are an expert in upstream open reading frames (uORFs) and translational regulation.

Read the abstract below and extract every SPECIFIC uORF or upstream ORF
that is the SUBJECT OF STUDY in this paper — i.e. a named, characterised,
or functionally described entity, not a generic mention of uORFs as a concept.

DO NOT extract:
- Generic labels such as "uORF", "upstream ORF", "sORF"
- uORFs mentioned only in passing or as background
- Anything without a specific gene context or identifier

For each specific uORF found, return a JSON list where every element has:
  "name"     : specific identifier (e.g. "ATF4-uORF1", "GADD34-uORF2", gene name + uORF)
  "function" : one sentence on its regulatory role based solely on this abstract, or null
  "species"  : organism studied (e.g. "Homo sapiens", "Mus musculus"); use "unknown" if not stated — do NOT guess
  "context"  : "aberrant" if the uORF is specifically associated with disease, pathogenic translation, or activated under stress/pathogenic conditions; "normal" if it regulates translation in healthy physiology; "unknown" if not clearly stated — do NOT guess

Return ONLY the JSON list. No markdown, no explanation.
If no specific uORF is the subject of study, return: []

ABSTRACT:
{context}
"""
)


def fetch_uorf_catalog(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA,
    max_pubmed: int = 200,
    output: str = "uorf_catalog.tsv",
    email: str = "user@mdc-berlin.de",
) -> None:
    _run_catalog(
        pubmed_query=UORF_QUERY,
        extraction_prompt=UORF_PROMPT,
        model=model,
        base_url=base_url,
        max_pubmed=max_pubmed,
        output=output,
        email=email,
        label="uORFs",
    )


# RAG summarisation #

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


# Main RAG pipeline #

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


# CLI #

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
  python orftracker.py --search_microproteins --max-pubmed 300 --output catalog.tsv
  python orftracker.py --search_uorfs --max-pubmed 300 --output uorf_catalog.tsv
  python orftracker.py --list-models
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
        help="Write output to file (default: stdout for RAG; catalog.tsv for catalog modes)")
    parser.add_argument("--list-models", action="store_true",
        help="List locally available Ollama models and exit")
    parser.add_argument("--search_microproteins", action="store_true",
        help=(
            "Catalog mode: broad PubMed search; extract name, function, and "
            "PubMed links for each microprotein found. Output: TSV."
        ))
    parser.add_argument("--search_uorfs", action="store_true",
        help=(
            "Catalog mode: broad PubMed search; extract name, function, and "
            "PubMed links for each uORF found. Output: TSV."
        ))

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

    if args.search_microproteins and args.search_uorfs:
        parser.error("Use --search_microproteins or --search_uorfs, not both at once.")

    if args.search_microproteins:
        fetch_microprotein_catalog(
            model=args.model,
            base_url=args.ollama_url,
            max_pubmed=args.max_pubmed,
            output=args.output or "microprotein_catalog.tsv",
            email=args.email,
        )
        sys.exit(0)

    if args.search_uorfs:
        fetch_uorf_catalog(
            model=args.model,
            base_url=args.ollama_url,
            max_pubmed=args.max_pubmed,
            output=args.output or "uorf_catalog.tsv",
            email=args.email,
        )
        sys.exit(0)

    if not args.query:
        parser.error(
            "'query' is required unless --list-models, "
            "--search_microproteins, or --search_uorfs is given."
        )

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
