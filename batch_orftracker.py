#!/usr/bin/env python3
# batch_orftracker.py — Jorge Ruiz-Orera
# Run orftracker over a gene list and save individual markdown summaries.
# Already-processed entries are skipped; FAISS indices are cached per gene.

import argparse
import sys
from pathlib import Path

from orftracker import run_orftracker, _ollama_is_running, DEFAULT_MODEL, DEFAULT_OLLAMA


def main():
    parser = argparse.ArgumentParser(
        prog="batch_orftracker",
        description="Batch orftracker over a list of microproteins/genes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python batch_orftracker.py example_genes.txt --outdir results/
  python batch_orftracker.py my_genes.txt --model llama3.1:8b --max-pubmed 40
  python batch_orftracker.py my_genes.txt --outdir results/ --top-k 8
        """,
    )
    parser.add_argument("gene_list",
        help="File with one gene/protein per line (# = comment)")
    parser.add_argument("--outdir",     default="results",
        help="Output directory (default: results/)")
    parser.add_argument("--model",      default=DEFAULT_MODEL,
        help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA,
        help=f"Ollama URL (default: {DEFAULT_OLLAMA})")
    parser.add_argument("--max-pubmed", type=int, default=30,
        help="Max PubMed abstracts per gene (default: 30)")
    parser.add_argument("--top-k",      type=int, default=6,
        help="Chunks passed to LLM (default: 6)")
    parser.add_argument("--email",      default="user@mdc-berlin.de",
        help="NCBI Entrez email")
    parser.add_argument("--no-cache",   action="store_true",
        help="Skip FAISS index caching")
    args = parser.parse_args()

    ### 1. Pre-flight check
    ## verify Ollama is running before starting the loop
    if not _ollama_is_running(args.ollama_url):
        print(f"[ERROR] Ollama not running at {args.ollama_url}. Run: ollama serve")
        sys.exit(1)

    out_dir   = Path(args.outdir)
    index_dir = out_dir / "faiss_indices"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_cache:
        index_dir.mkdir(exist_ok=True)

    with open(args.gene_list) as fh:
        queries = [
            line.strip()
            for line in fh
            if line.strip() and not line.startswith("#")
        ]

    print(f"[Batch] {len(queries)} entries | model={args.model} | top_k={args.top_k}")
    ok, skipped, failed = 0, 0, 0

    ### 2. Process each gene
    for i, query in enumerate(queries, 1):
        safe = query.replace("/", "_").replace(" ", "_")
        outf = out_dir / f"{safe}.md"
        idx  = str(index_dir / safe) if not args.no_cache else None

        print(f"\n[{i}/{len(queries)}] {query}")

        ## skip if summary already exists
        if outf.exists():
            print("  -> already done, skipping.")
            skipped += 1
            continue

        try:
            summary = run_orftracker(
                query=query,
                model=args.model,
                base_url=args.ollama_url,
                max_pubmed=args.max_pubmed,
                top_k=args.top_k,
                save_index=idx,
                email=args.email,
            )
            outf.write_text(f"# {query}\n\n{summary}\n")
            print(f"  -> saved to {outf}")
            ok += 1
        except Exception as e:
            print(f"  -> ERROR: {e}")
            outf.write_text(f"# {query}\n\n**Error:** {e}\n")
            failed += 1

    print(f"\n[Batch done] ok={ok}  skipped={skipped}  failed={failed}")
    print(f"Results in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
