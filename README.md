# Data Science Portfolio

End-to-end data science projects built on real, messy public data. Each one goes from raw data to cleaning, modeling, honest evaluation and an interactive dashboard.

🔗 **[Browse the dashboards](https://soushiki.github.io/data-science/)**

## Projects

| Project | Question | Data | Methods | Links |
|---|---|---|---|---|
| [**NYC 311 Delays**](projects/nyc-311-delays) | Which city service requests will be resolved slower than comparable requests? | NYC Open Data API, ~400k requests (2026) | Leakage-safe features, out-of-time validation, gradient boosting, lift analysis | [Dashboard](https://soushiki.github.io/data-science/nyc-311-delays/) · [Write-up](projects/nyc-311-delays/README.md) |

## Repository layout

```
projects/<name>/     self-contained project: src/, README, figures, requirements.txt, run_all.py
docs/                GitHub Pages site: landing page + one dashboard per project
```

Every project is reproducible on its own:

```bash
cd projects/<name>
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
python run_all.py
```
