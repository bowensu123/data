"""CLI entry point: python -m aura_viz --work-dir <dir> --port 8000"""

import argparse

from .server import serve


def main():
    ap = argparse.ArgumentParser(description="Local platform to visualize AURA streaming training data.")
    ap.add_argument("--work-dir", default="real_test/output",
                    help="Directory containing training_instances.jsonl (and optionally prepared_videos/)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    serve(args.work_dir, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
