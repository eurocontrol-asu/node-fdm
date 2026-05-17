# Sweep run 2 — analyse intra-axe

36 runs · 6 axes × 3 seeds · `train_limit=5000`.

**Baseline run 2** (intègre les 3 acquis run 1 figés : `weighting=off`, `activation=relu`, `bs=64`)

`bs=64, lr=1e-3, weighting=off, activation=relu, epochs=50, seq_len=60, hidden_width=48, backbone_depth=3, head_depth=2`

- alt = **254.92 ± 21.69 m**
- tas = **2.953 ± 0.047 m/s**
- γ   = **0.2846 ± 0.0056 rad**
- hdg = **1.78 ± 0.05°**
- durée = **416 ± 4 min** (wall-clock, train+predict+evaluate sur 1 GPU)

Notation : `Δm` = écart de la moyenne vs baseline (− = mieux). `Δs` = écart de l'écart-type vs baseline (− = plus stable).

## Comparaison baselines run 1 → run 2

| métrique     | run 1 baseline (bs=128, w=on, silu) | run 2 baseline (bs=64, w=off, relu) | Δm        | Δs       |
|--------------|--------------------------------------|--------------------------------------|-----------|----------|
| mae_alt      | 278.8 ± 21.3                         | 254.9 ± 21.7                         | **−23.9** | +0.4     |
| mae_tas      | 2.940 ± 0.038                        | 2.953 ± 0.047                        | +0.013    | +0.009   |
| mae_gamma    | 0.2867 ± 0.0053                      | 0.2846 ± 0.0056                      | −0.0021   | +0.0003  |
| mae_heading  | 1.93 ± 0.32                          | 1.78 ± 0.05                          | **−0.15** | **−0.27** |
| durée (min)  | 287 ± 1                              | 416 ± 4                              | +129      | +3       |

→ Les 3 free-wins de run 1 **gagnent en moyenne sur alt et hdg** comme prévu. La **stabilité hdg est confirmée** (std ÷6).
→ ⚠️ L'effet stabilisateur de `weighting=off` seul (std=1.2 dans run 1) **n'a pas tenu** avec bs=64+relu : alt_std remonte à 21.7. Une interaction négative existe, mais elle ne dégrade pas la moyenne — seulement la dispersion.
→ Le coût a augmenté (+129 min) à cause de bs=64.

## `batch_size` (baseline=64)

| bs  | alt mean±std       | Δm_alt | Δs_alt   | tas mean±std       | Δm_tas | Δs_tas | γ mean±std         | Δm_γ    | Δs_γ    | hdg mean±std    | Δm_hdg | Δs_hdg | durée (min) | Δ_wc  |
|-----|--------------------|--------|----------|--------------------|--------|--------|--------------------|---------|---------|-----------------|--------|--------|-------------|-------|
| 32  | 244.9 ± 4.2        | **−10.1** | **−17.5** | 3.01 ± 0.05      | +0.06  | +0.00  | 0.2781 ± 0.0024    | −0.0065 | −0.0032 | 1.58 ± 0.05     | **−0.20** | +0.00 | 689 ± 6     | **+273** |
| 64  | *254.9 ± 21.7*     | base   | base     | 2.95 ± 0.05        | base   | base   | *0.2846 ± 0.0056*  | base    | base    | 1.78 ± 0.05     | base   | base   | *416 ± 4*   | base  |
| 128 | 258.9 ± 11.6       | +3.9   | −10.1    | 3.08 ± 0.02        | +0.12  | −0.02  | 0.2858 ± 0.0036    | +0.0012 | −0.0020 | 1.94 ± 0.30     | +0.16  | +0.25  | 265 ± 1     | −151  |

→ **bs=32 confirme run 1** : meilleure moyenne (−10 m) **et** divise alt_std par 5 (4.2 vs 21.7). Coût ×1.7.
→ bs=128 dégrade peu sur l'altitude mais déstabilise heading (std ×6 : 0.05 → 0.30). Pour ~150 min économisés, mauvais trade-off.
→ **Cohérent run 1** : bs=32/64 dominent, bs=128/256 perdent. Le sweet spot est à **32**.

## `lr` (baseline=1e-3)

| lr     | alt mean±std       | Δm_alt | Δs_alt   | tas mean±std       | Δm_tas | Δs_tas | γ mean±std         | Δm_γ    | Δs_γ    | hdg mean±std    | Δm_hdg | Δs_hdg | durée (min) | Δ_wc |
|--------|--------------------|--------|----------|--------------------|--------|--------|--------------------|---------|---------|-----------------|--------|--------|-------------|------|
| 8.5e-4 | 250.1 ± 16.3       | −4.8   | −5.4     | 2.94 ± 0.05        | −0.02  | +0.01  | 0.2806 ± 0.0072    | −0.0040 | +0.0016 | 1.64 ± 0.09     | −0.15  | +0.04  | 419 ± 4     | +3   |
| 1e-3   | *254.9 ± 21.7*     | base   | base     | 2.95 ± 0.05        | base   | base   | *0.2846 ± 0.0056*  | base    | base    | 1.78 ± 0.05     | base   | base   | *416 ± 4*   | base |
| 1.5e-3 | 265.7 ± 7.7        | +10.7  | −14.0    | 3.00 ± 0.06        | +0.05  | +0.01  | 0.2846 ± 0.0033    | 0.0000  | −0.0023 | 1.62 ± 0.15     | −0.16  | +0.10  | 425 ± 2     | +9   |

→ lr=8.5e-4 marginal (gain dans le bruit). lr=1.5e-3 dégrade alt (+11 m) mais améliore hdg.
→ **Cohérent run 1** : 1e-3 reste le sweet spot ; la fenêtre étroite (8.5e-4 → 1.5e-3) confirme qu'aucune large amélioration n'est cachée. Pas de besoin d'explorer plus.

## `seq_len` (baseline=60)

| seq_len | alt mean±std        | Δm_alt | Δs_alt   | tas mean±std       | Δm_tas    | Δs_tas    | γ mean±std         | Δm_γ    | Δs_γ    | hdg mean±std    | Δm_hdg | Δs_hdg | durée (min) | Δ_wc |
|---------|---------------------|--------|----------|--------------------|-----------|-----------|--------------------|---------|---------|-----------------|--------|--------|-------------|------|
| **30**  | **237.7 ± 7.1**     | **−17.2** | **−14.6** | 3.35 ± 0.05    | **+0.39** | +0.01     | 0.2782 ± 0.0041    | −0.0064 | −0.0015 | 1.59 ± 0.16     | −0.19  | +0.10  | 414 ± 3     | −2   |
| 60      | *254.9 ± 21.7*      | base   | base     | 2.95 ± 0.05        | base      | base      | *0.2846 ± 0.0056*  | base    | base    | 1.78 ± 0.05     | base   | base   | *416 ± 4*   | base |
| 120     | 259.9 ± **23.0**    | +5.0   | +1.3     | 3.01 ± **0.14**    | +0.06     | **+0.10** | 0.2865 ± 0.0045    | +0.0019 | −0.0011 | 1.86 ± 0.26     | +0.08  | +0.21  | 399 ± 1     | −17  |

→ **seq_len=30 = plus gros gain alt du sweep** (−17 m) avec std réduit ×3.
→ ⚠️ **Trade-off TAS réel** : seq_30 dégrade tas de +0.39 m/s (13 %). Heading et γ tous deux améliorés.
→ seq_120 sans intérêt : pire moyenne **et** variance.
→ **Nouveauté run 2** (seq_len n'avait pas été testé dans run 1).

## `activation` (baseline=relu)

| act  | alt mean±std       | Δm_alt | Δs_alt   | tas mean±std       | Δm_tas | Δs_tas | γ mean±std         | Δm_γ    | Δs_γ    | hdg mean±std    | Δm_hdg | Δs_hdg | durée (min) | Δ_wc |
|------|--------------------|--------|----------|--------------------|--------|--------|--------------------|---------|---------|-----------------|--------|--------|-------------|------|
| relu | *254.9 ± 21.7*     | base   | base     | 2.95 ± 0.05        | base   | base   | *0.2846 ± 0.0056*  | base    | base    | 1.78 ± 0.05     | base   | base   | *416 ± 4*   | base |
| gelu | 251.0 ± 4.7        | −3.9   | **−17.0** | 3.06 ± 0.09       | +0.11  | +0.04  | 0.2805 ± 0.0009    | −0.0041 | **−0.0047** | 1.87 ± 0.14   | +0.09  | +0.10  | 411 ± 4     | −5   |

→ gelu : moyenne **équivalente** à relu (−4 m dans le bruit) mais **std÷5 sur alt** et **÷6 sur γ**.
→ gelu coûte +0.11 m/s sur tas et un poil sur hdg.
→ **À considérer comme nouvelle baseline** si la stabilité prime. Run 1 n'avait pas testé gelu.

## `hidden_width` (baseline=48)

| hidden | alt mean±std       | Δm_alt | Δs_alt   | tas mean±std       | Δm_tas | Δs_tas | γ mean±std         | Δm_γ    | Δs_γ    | hdg mean±std    | Δm_hdg | Δs_hdg | durée (min) | Δ_wc |
|--------|--------------------|--------|----------|--------------------|--------|--------|--------------------|---------|---------|-----------------|--------|--------|-------------|------|
| 24     | 260.6 ± 13.2       | +5.7   | −8.4     | 3.07 ± 0.04        | +0.11  | −0.01  | 0.2851 ± 0.0007    | +0.0005 | **−0.0049** | 1.80 ± 0.23   | +0.02  | +0.18  | 411 ± 4     | −5   |
| 48     | *254.9 ± 21.7*     | base   | base     | 2.95 ± 0.05        | base   | base   | *0.2846 ± 0.0056*  | base    | base    | 1.78 ± 0.05     | base   | base   | *416 ± 4*   | base |
| 96     | 268.5 ± **35.9**   | +13.6  | **+14.2** | 3.02 ± 0.04       | +0.07  | −0.01  | 0.2778 ± 0.0036    | −0.0068 | −0.0020 | 2.05 ± 0.39     | +0.27  | +0.34  | 418 ± 5     | +2   |

→ **hidden=24** : alt légèrement pire (+6 m), mais γ_std divisé par 8 (0.0007).
→ **hidden=96** : pire moyenne (+14 m) **et** std×1.7. La capacité supplémentaire **déstabilise**.
→ **Conclusion** : le modèle n'est pas sous-paramétré ; 48 reste le sweet spot. Aller au-dessus est néfaste.

## `backbone_depth` (baseline=3)

| backbone | alt mean±std       | Δm_alt | Δs_alt   | tas mean±std       | Δm_tas | Δs_tas | γ mean±std         | Δm_γ    | Δs_γ    | hdg mean±std    | Δm_hdg | Δs_hdg | durée (min) | Δ_wc |
|----------|--------------------|--------|----------|--------------------|--------|--------|--------------------|---------|---------|-----------------|--------|--------|-------------|------|
| 2        | 254.6 ± 6.7        | −0.3   | **−15.0** | 2.99 ± 0.07       | +0.03  | +0.02  | 0.2839 ± 0.0017    | −0.0007 | −0.0039 | 1.68 ± 0.18     | −0.10  | +0.12  | 414 ± 2     | −2   |
| 3        | *254.9 ± 21.7*     | base   | base     | 2.95 ± 0.05        | base   | base   | *0.2846 ± 0.0056*  | base    | base    | 1.78 ± 0.05     | base   | base   | *416 ± 4*   | base |
| 4        | 267.0 ± **35.3**   | +12.1  | **+13.6** | 3.03 ± 0.07       | +0.08  | +0.03  | 0.2866 ± 0.0084    | +0.0020 | +0.0028 | 2.03 ± 0.34     | +0.25  | +0.29  | 359 ± 25    | −57  |

→ **backbone=2** : moyenne **identique** à la baseline (Δm = −0.3 m) mais **alt_std÷3** (6.7 vs 21.7).
→ backbone=4 : pire moyenne (+12 m) **et** std×1.6. Comme `hidden=96`, **plus profond = plus instable**.
→ **Conclusion** : on peut **simplifier l'archi** sans rien perdre. backbone=2 est dominé sur la moyenne, mais c'est le seul axe qui garde la même mae moyenne **et** stabilise comme `bs=32`.

---

## Synthèse stabilité (alt_std)

Configs qui **réduisent fortement le bruit inter-seed** (baseline run 2 std = 21.7) :

| config             | alt_std | facteur vs baseline |
|--------------------|---------|---------------------|
| bs=32              | 4.2     | **÷5.2**            |
| activation=gelu    | 4.7     | **÷4.6**            |
| backbone_depth=2   | 6.7     | **÷3.2**            |
| seq_len=30         | 7.1     | **÷3.1**            |
| lr=1.5e-3          | 7.7     | **÷2.8**            |
| bs=128             | 11.6    | ÷1.9                |
| hidden_width=24    | 13.2    | ÷1.6                |
| lr=8.5e-4          | 16.3    | ÷1.3                |

Configs qui **font exploser la variance** :

| config             | alt_std | facteur vs baseline |
|--------------------|---------|---------------------|
| hidden_width=96    | 35.9    | ×1.7                |
| backbone_depth=4   | 35.3    | ×1.6                |
| seq_len=120        | 23.0    | ×1.1                |

→ **Pattern clair** : monter en capacité (largeur OU profondeur) déstabilise systématiquement. Réduire l'archi (backbone=2) ou la fenêtre temporelle (seq=30) stabilise.

## Coût compute (wall-clock par run)

| config              | durée (min) | facteur vs baseline | qualité (Δm_alt) |
|---------------------|-------------|---------------------|------------------|
| bs=128              | 265         | ×0.64               | +3.9 m           |
| backbone=4          | 359         | ×0.86               | +12.1 m (pire)   |
| seq_120             | 399         | ×0.96               | +5.0 m           |
| activation=gelu     | 411         | ×0.99               | −3.9 m           |
| hidden=24           | 411         | ×0.99               | +5.7 m           |
| seq_30              | 414         | ×1.00               | **−17.2 m** (best) |
| backbone=2          | 414         | ×1.00               | ~0 m             |
| **baseline** (bs=64) | **416**    | ×1.00               | base             |
| hidden=96           | 418         | ×1.00               | +13.6 m          |
| lr=8.5e-4 / 1.5e-3  | 419 / 425   | ×1.01–1.02          | ±5 à +11 m       |
| bs=32               | **689**     | **×1.66**           | −10.1 m          |

→ **seq_30 est le meilleur ROI absolu** : meilleur gain (−17 m) à coût **identique**.
→ **bs=32 reste cher** mais paie en stabilité (alt_std÷5).
→ **backbone=2 est presque gratuit** (~0 m, mais stabilité ÷3) → simplifie l'archi sans coût.

## Comparaison conclusions run 1 → run 2

| acquis run 1                             | confirmé run 2 ?                                                       |
|------------------------------------------|------------------------------------------------------------------------|
| `weighting=off` améliore mae_alt         | Cumulé dans la baseline → −24 m global vs baseline run 1 ✓             |
| `weighting=off` divise alt_std par 18    | **NON cumulé** : baseline run 2 std=21.7 (idem run 1 avec weighting=on) ⚠️ |
| `activation=relu` meilleur que silu      | Cumulé. gelu fait aussi bien ou mieux → relu pas dominant absolu       |
| `bs=64` meilleur compromis qualité/coût  | bs=32 est **meilleur** sur run 2 (std÷5). bs=64 = compromis seulement  |
| `bs=256` dégrade fortement               | bs=128 le fait déjà (run 2) → confirme tendance                        |
| `lr ∈ (1e-3, 3e-3)` à explorer           | Exploré. Rien à gagner : 1e-3 reste le sweet spot                      |
| TAS plafonné aux hyperparams testés      | **Falsifié** : seq_len=30 dégrade TAS de +0.39 → seq_len a un effet réel sur TAS |
| epochs=100 overfit                       | Non re-testé (drop justifié)                                           |

## Nouveautés run 2

- **seq_len est le levier le plus puissant** non testé en run 1 : seq=30 gagne 17 m d'alt sans coût compute, mais dégrade TAS.
- **L'archi est trop grande** : `backbone=2` ou `hidden=24` montrent qu'on peut réduire ; aller au-dessus (backbone=4, hidden=96) **déstabilise**.
- **gelu compétitif avec relu** : même perf moyenne, ÷5 sur alt_std, ÷6 sur γ_std. Candidat sérieux pour stabiliser sans coût.

## Recommandations pour run 3

### Combinaisons jamais testées (priorité haute)

Aucun run du sweep n'a combiné les **3 stabilisateurs** : `bs=32` × `seq=30` × `backbone=2` (× `gelu` ?). Estimation d'effets indépendants → alt ~210 m, std ~2 m. Mais on a vu run 1→2 que les effets ne sont pas additifs, donc à vérifier empiriquement.

### Proposition matrice run 3

**Nouvelle baseline** : `bs=32, lr=1e-3, weighting=off, relu, seq_len=30, hidden=48, backbone=2` (cumule seq=30 et backbone=2).

| axe          | variants                | configs |
|--------------|-------------------------|---------|
| baseline     | —                       | 1       |
| `activation` | gelu                    | 1       |
| `seq_len`    | 15, 60                  | 2       |
| `bs`         | 16, 64                  | 2       |
| `backbone`   | 3                       | 1       |
| `hidden`     | 32, 64                  | 2       |

→ 9 configs × 3 seeds = **27 runs**.

Vu les durées run 2 (bs=32 ~690 min, seq_30 ~414 min, on s'attend à ~500 min/run en moyenne), coût estimé **~225 h sur 1 GPU, ~57 h sur 4 slots**.

### Trade-off TAS à clarifier avant run 3

Si l'objectif inclut la **vitesse**, seq=30 pénalise tas de 13 %. Soit on accepte (seq=30 acquis), soit on remet une boucle TAS-sensible (seq=60 fixé, exploration sur d'autres axes).

### Hypothèse à valider hors sweep

L'**explosion de variance alt sur baseline run 2** (21.7 vs attendu 1.2) suggère une **interaction négative** entre `relu` et `bs=64` qui annule le stabilisateur de `weighting=off`. À investiguer : refaire 3 seeds de `(bs=128, w=off, relu)` pour mesurer si la variance vient de bs=64 ou de l'interaction multi-acquis.

## Note artefacts

- `queue.json` : 36 completed, 0 failed.
- `training_losses.csv` désormais copié dans chaque `{run_id}/` (nouveau pour run 2 — permet d'analyser train/val loss vs mae intégré sans aller chercher dans `data/models/`).
- Agréger en `summary.parquet` :
  ```bash
  uv run python -m scripts.sweep_runner aggregate --results-dir results_run2
  ```
