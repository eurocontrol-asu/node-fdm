# Sweep run 1 — analyse intra-axe

33 runs · 11 axes × 3 seeds · `train_limit=5000` · ~16 000 s / run.

**Baseline** : `bs=128, lr=1e-3, weighting=on, α=0.5, silu, epochs=50`
- alt = **278.8 ± 21.3 m**
- tas = **2.940 ± 0.038 m/s**
- γ   = **0.2867 ± 0.0053 rad**
- hdg = **1.93 ± 0.32°**
- durée = **287.5 ± 1.3 min** (wall-clock, train+predict+evaluate sur 1 GPU)

Notation : `Δm` = écart de la moyenne vs baseline (− = mieux). `Δs` = écart de l'écart-type vs baseline (− = plus stable).

## Lecture d'ensemble

⚠️ **Comparer les axes entre eux est trompeur** : chaque axe contient un nombre différent de valeurs et la baseline n'apparaît qu'une seule fois. Toute la lecture utile se fait **intra-axe**, valeur par valeur, contre la baseline.

## `batch_size` (baseline=128)

| bs  | alt mean±std       | Δm_alt | Δs_alt   | tas mean±std       | Δm_tas | Δs_tas | γ mean±std         | Δm_γ    | Δs_γ    | hdg mean±std    | Δm_hdg | Δs_hdg | durée (min) | Δ_wc   |
|-----|--------------------|--------|----------|--------------------|--------|--------|--------------------|---------|---------|-----------------|--------|--------|-------------|--------|
| 32  | 259.2 ± 10.4       | −19.6  | **−10.9** | 2.98 ± 0.06       | +0.04  | +0.02  | 0.2805 ± 0.0043    | −0.0062 | −0.0010 | 1.48 ± 0.05     | −0.45  | **−0.27** | 710 ± 6   | **+423** |
| 64  | 256.5 ± 8.9        | **−22.3** | **−12.4** | 2.96 ± 0.02   | +0.02  | −0.02  | 0.2854 ± 0.0023    | −0.0012 | −0.0030 | 1.70 ± 0.25     | −0.24  | −0.07  | 443 ± 1     | +156   |
| 128 | *278.8 ± 21.3*     | base   | base     | 2.94 ± 0.04        | base   | base   | *0.2867 ± 0.0053*  | base    | base    | 1.93 ± 0.32     | base   | base   | *287 ± 1*   | base   |
| 256 | 309.6 ± **60.9**   | +30.8  | **+39.6** | 3.01 ± 0.10       | +0.07  | +0.06  | 0.2953 ± 0.0170    | +0.0086 | +0.0117 | 2.10 ± 0.13     | +0.17  | −0.20  | 197 ± 1     | −91    |

→ bs=32/64 gagnent **et** stabilisent (alt_std divisé par 2). bs=256 fait exploser la variance.
→ bs=32 est **dominant sur heading** (−0.45°, le plus gros gain hdg du sweep).

## `lr` (baseline=1e-3)

| lr   | alt mean±std         | Δm_alt | Δs_alt    | tas mean±std       | Δm_tas | Δs_tas | γ mean±std        | Δm_γ    | Δs_γ    | hdg mean±std    | Δm_hdg | Δs_hdg | durée (min) | Δ_wc |
|------|----------------------|--------|-----------|--------------------|--------|--------|-------------------|---------|---------|-----------------|--------|--------|-------------|------|
| 5e-4 | 309.2 ± 13.3         | +30.4  | −8.0      | 2.96 ± 0.01        | +0.02  | −0.03  | 0.2895 ± 0.0027   | +0.0028 | −0.0026 | 2.08 ± 0.11     | +0.15  | −0.21  | 269 ± 2     | −19  |
| 1e-3 | *278.8 ± 21.3*       | base   | base      | 2.94 ± 0.04        | base   | base   | *0.2867 ± 0.0053* | base    | base    | 1.93 ± 0.32     | base   | base   | *287 ± 1*   | base |
| 3e-3 | 375.9 ± **156.7**    | +97.1  | **+135.4** | 3.12 ± 0.11       | +0.18  | +0.07  | 0.3015 ± **0.0241** | +0.0149 | **+0.0188** | 2.41 ± 0.36     | +0.48  | +0.03  | 269 ± 2     | −19  |

→ lr=3e-3 dégrade **toutes** les variables et entre dans un régime divergent (seed 0 → 556 m).
→ lr=5e-4 sous-apprend ; l'espace `(1e-3, 3e-3)` n'a pas été exploré et reste l'inconnue.

## `weighting` (baseline=on, α=0.5)

| weighting | alt mean±std    | Δm_alt | Δs_alt    | tas mean±std    | Δm_tas | Δs_tas | γ mean±std         | Δm_γ    | Δs_γ    | hdg mean±std    | Δm_hdg | Δs_hdg | durée (min) | Δ_wc |
|-----------|-----------------|--------|-----------|-----------------|--------|--------|--------------------|---------|---------|-----------------|--------|--------|-------------|------|
| on        | *278.8 ± 21.3*  | base   | base      | 2.94 ± 0.04     | base   | base   | *0.2867 ± 0.0053*  | base    | base    | 1.93 ± 0.32     | base   | base   | *287 ± 1*   | base |
| **off**   | **251.8 ± 1.2** | **−27.0** | **−20.2** | 2.93 ± 0.01 | −0.01  | −0.03  | **0.2845 ± 0.0010** | −0.0021 | **−0.0043** | **1.79 ± 0.06** | −0.14  | **−0.26** | 269 ± 0     | −18  |

→ Gain le plus net du sweep : **gagne sur les 3 moyennes ET divise les 3 std par ~5–20**, aucune contrepartie.

## `alpha` (baseline=0.5, weighting forcé on)

| α   | alt mean±std    | Δm_alt | Δs_alt | tas mean±std    | Δm_tas | Δs_tas | γ mean±std        | Δm_γ    | Δs_γ    | hdg mean±std    | Δm_hdg | Δs_hdg | durée (min) | Δ_wc |
|-----|-----------------|--------|--------|-----------------|--------|--------|-------------------|---------|---------|-----------------|--------|--------|-------------|------|
| 0.3 | 286.0 ± 47.1    | +7.2   | +25.8  | 2.97 ± 0.07     | +0.03  | +0.03  | 0.2899 ± 0.0091   | +0.0032 | +0.0038 | 1.88 ± 0.07     | −0.05  | **−0.26** | 269 ± 2     | −19  |
| 0.5 | *278.8 ± 21.3*  | base   | base   | 2.94 ± 0.04     | base   | base   | *0.2867 ± 0.0053* | base    | base    | 1.93 ± 0.32     | base   | base   | *287 ± 1*   | base |
| 0.7 | 300.6 ± 66.5    | +21.8  | **+45.2** | 2.99 ± 0.10  | +0.05  | +0.06  | 0.2926 ± 0.0157   | +0.0060 | +0.0104 | 1.80 ± 0.17     | −0.13  | −0.15  | 268 ± 2     | −20  |

→ α≠0.5 améliore hdg mais fait exploser la variance alt. Mauvais trade-off, et dominé par weighting=off de toute façon.

## `epochs` (baseline=50)

| epochs | alt mean±std       | Δm_alt | Δs_alt   | tas mean±std    | Δm_tas | Δs_tas | γ mean±std        | Δm_γ    | Δs_γ    | hdg mean±std    | Δm_hdg | Δs_hdg | durée (min) | Δ_wc   |
|--------|--------------------|--------|----------|-----------------|--------|--------|-------------------|---------|---------|-----------------|--------|--------|-------------|--------|
| 50     | *278.8 ± 21.3*     | base   | base     | 2.94 ± 0.04     | base   | base   | *0.2867 ± 0.0053* | base    | base    | 1.93 ± 0.32     | base   | base   | *287 ± 1*   | base   |
| 100    | 291.1 ± **62.8**   | +12.3  | **+41.5** | 2.99 ± 0.09    | +0.05  | +0.05  | 0.2938 ± 0.0148   | +0.0071 | +0.0095 | 2.00 ± 0.16     | +0.07  | −0.16  | 418 ± 5     | **+130** |

→ +50 époques = +40 m de bruit inter-seed. Overfit cohérent à `train_limit=5000`.

## `activation` (baseline=silu)

| act  | alt mean±std    | Δm_alt | Δs_alt    | tas mean±std    | Δm_tas | Δs_tas | γ mean±std         | Δm_γ    | Δs_γ    | hdg mean±std    | Δm_hdg | Δs_hdg | durée (min) | Δ_wc |
|------|-----------------|--------|-----------|-----------------|--------|--------|--------------------|---------|---------|-----------------|--------|--------|-------------|------|
| silu | *278.8 ± 21.3*  | base   | base      | 2.94 ± 0.04     | base   | base   | *0.2867 ± 0.0053*  | base    | base    | 1.93 ± 0.32     | base   | base   | *287 ± 1*   | base |
| relu | 256.9 ± 7.7     | −21.9  | **−13.6** | 2.98 ± 0.03     | +0.04  | −0.01  | **0.2775 ± 0.0025** | **−0.0092** | −0.0028 | 1.78 ± 0.21     | −0.15  | −0.11  | 269 ± 1     | −18  |

→ relu : meilleur mean **et** ~3× plus stable sur alt.

---

## Synthèse stabilité (alt_std)

Configs qui **réduisent fortement le bruit inter-seed** :

| config            | alt_std baseline → variant | facteur |
|-------------------|----------------------------|---------|
| weighting=off     | 21.3 → **1.2**             | ×18     |
| activation=relu   | 21.3 → 7.7                 | ×2.8    |
| bs=64             | 21.3 → 8.9                 | ×2.4    |
| bs=32             | 21.3 → 10.4                | ×2.0    |

Configs qui **font exploser la variance** (régime instable, le seed décide) :

| config            | alt_std baseline → variant | facteur |
|-------------------|----------------------------|---------|
| lr=3e-3           | 21.3 → 156.7               | ×7.4    |
| α=0.7             | 21.3 → 66.5                | ×3.1    |
| epochs=100        | 21.3 → 62.8                | ×2.9    |
| bs=256            | 21.3 → 60.9                | ×2.9    |
| α=0.3             | 21.3 → 47.1                | ×2.2    |

## TAS — insensibilité

`tas` varie très peu (2.93–3.12 m/s, baseline 2.94). Tous les axes sauf `lr=3e-3` restent à ±0.07 m/s. La dynamique longitudinale vitesse est **insensible aux hyperparams testés** ; le bottleneck est ailleurs (données, archi, ou seq_len).

## Coût compute (wall-clock par run)

| config       | durée (min) | facteur vs baseline | qualité (Δm_alt) |
|--------------|-------------|---------------------|------------------|
| bs=256       | 197         | ×0.69               | +30.8 m (pire)   |
| weighting=off | 269        | ×0.94               | **−27.0 m** (best) |
| activation=relu | 269      | ×0.94               | −21.9 m          |
| lr (toutes)  | 269         | ×0.94               | variable         |
| α (toutes)   | 268         | ×0.93               | +7 à +22 m       |
| **baseline** (bs=128, 50ep) | **287** | ×1.00 | base       |
| epochs=100   | 418         | ×1.45               | +12.3 m (pire)   |
| bs=64        | 443         | ×1.54               | −22.3 m (best)   |
| bs=32        | **710**     | **×2.47**           | −19.6 m          |

→ **Trade-off bs critique** : bs=64 et bs=32 sont quasi équivalents en qualité (Δalt ~−20 m) mais bs=32 coûte **60 % de temps en plus** (710 vs 443 min). Pour la nouvelle baseline, **bs=64 domine bs=32** sauf si le latéral (hdg) prime — auquel cas bs=32 gagne −0.45° pour 267 min de plus par run.
→ **epochs=100 est doublement perdant** : +130 min ET dégradation de toutes les métriques (overfit).
→ **weighting=off / relu sont "gratuits"** : −18 min de durée **et** meilleure qualité.

## γ (flight path angle)

Plage très étroite (0.277–0.302 rad). Hiérarchie cohérente avec alt mais avec un gagnant différent :

- **activation=relu** : meilleur en moyenne (**0.2775**, Δm = −0.0092 — le seul gain γ significatif du sweep).
- **bs=32** : second (0.2805, Δm = −0.0062).
- **weighting=off** : ~neutre sur la moyenne (Δm = −0.002) mais **divise le std par 5** (0.0010 vs 0.0053).
- **lr=3e-3** : pire mean **et** std×4.5 — même régime instable que pour alt.

→ γ est **le seul axe où relu domine clairement** (alt préfère bs=64). Argument additionnel pour passer à relu.

## Recommandations pour run 2

1. **Nouvelle baseline** : `bs=64, lr=1e-3, weighting=off, activation=relu, epochs=50`.
   - Cumule les 3 effets stabilisants (×18 · ×2.8 · ×2.4) — alt_std attendu ~1–3 m si effets indépendants.
2. **Tester l'interaction** weighting_off × bs_64 × relu : le sweep par axe ne les a jamais combinés.
3. **lr** : explorer `{1e-3, 1.5e-3, 2e-3}` — la zone entre 1e-3 et la divergence à 3e-3 reste inconnue.
4. **Abandonner** du prochain sweep : `bs=256`, `lr=3e-3`, `epochs=100`, `α ∈ {0.3, 0.7}` (dominés par weighting=off).
5. **Si l'objectif inclut le latéral** : prioriser **bs=32** sur bs=64 (gain hdg net : −0.45° vs −0.24°), coût alt minime (+3 m).
6. **TAS plafonné** : ne plus chercher de gain TAS via hyperparams — investiguer plutôt seq_len, normalisation des features vent/Mach, ou l'architecture du sous-réseau moteur.

## Note artefacts

- `queue.json` : 33 completed, 0 failed.
- `summary.parquet` présent sur disque est **périmé** (24 lignes — généré avant la fin des `alpha_*` et `activation_relu_*`). Régénérer avec :
  ```bash
  uv run python -m scripts.sweep_runner aggregate --results-dir results_run1
  ```
