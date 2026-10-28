"""Run the full pipeline: fetch -> clean -> features/train -> figures -> dashboard export -> site."""
import sys
import time

from src import build_site, clean, export_dashboard, fetch_data, figures, train


def main() -> None:
    refresh = "--refresh" in sys.argv
    steps = [("fetch", lambda: fetch_data.main(refresh)), ("clean", clean.main),
             ("train", train.main), ("figures", figures.main),
             ("export", export_dashboard.main),
             ("site", build_site.build)]
    for name, fn in steps:
        t = time.time()
        print(f"\n=== {name} ===")
        fn()
        print(f"[{name}] done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
