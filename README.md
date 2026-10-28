# Data Science Portfolio

End-to-end data science projects built on real, messy public data. Each one goes from raw data to cleaning, modeling, honest evaluation and an interactive dashboard.

🔗 **[Browse the dashboards](https://soushiki.github.io/data-science/)**

## Projects

| Project | Question | Data | Methods | Links |
|---|---|---|---|---|
| [**NYC 311 Delays**](projects/nyc-311-delays) | Which city service requests will be resolved slower than comparable requests? | NYC Open Data API, ~400k requests (2026) | Leakage-safe features, out-of-time validation, gradient boosting, lift analysis | [Dashboard](https://soushiki.github.io/data-science/nyc-311-delays/) · [Write-up](projects/nyc-311-delays/README.md) |
| [**GB Electricity Demand**](projects/gb-electricity-demand) | Can a weather-driven model beat Great Britain's grid operator at forecasting tomorrow's demand? | NESO demand + forecast archive, Open-Meteo historical forecasts (2022–2026) | Time-series features, gradient boosting, conformal prediction intervals, benchmark vs. official forecasts | [Dashboard](https://soushiki.github.io/data-science/gb-electricity-demand/) · [Write-up](projects/gb-electricity-demand/README.md) |

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
