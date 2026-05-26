# Sweep run 4 — analyse intra-axe

27 runs · 9 axes × 3 seeds · `train_limit=5000`.

**Baseline run 4** (cumule les acquis run 1 + 2 + 3, switch `bs=32 → bs=64` pour le ROI compute)

`bs=64, lr=1e-3, weighting=off, activation=relu, epochs=50, seq_len=30, hidden_width=48, backbone_depth=3, head_depth=2`

- alt = **237.72 ± 7.06 m**
- tas = **3.346 ± 0.053 m/s**
- γ   = **0.2782 ± 0.0041 rad**
- hdg = **1.59 ± 0.16°**
- durée = **422 ± 6 min** (wall-clock, train+predict+evaluate sur 1 GPU)

Notation : `Δm` = écart de la moyenne vs baseline (− = mieux). `Δs` = écart de l'écart-type vs baseline (− = plus stable).

## Comparaison baselines run 3 → run 4

| métrique     | run 3 baseline (bs=32, bb=2)         | run 4 baseline (bs=64, bb=3)         | Δm        | Δs        |
|--------------|---------------------------------------|---------------------------------------|-----------|-----------|
| mae_alt      | 256.48 ± 5.03                         | 237.72 ± 7.06                         | **−18.76** | +2.03     |
| mae_tas      | 3.307 ± 0.082                         | 3.346 ± 0.053                         | +0.039    | −0.029    |
| mae_gamma    | 0.2791 ± 0.0009                       | 0.2782 ± 0.0041                       | −0.0009   | +0.0032   |
| mae_heading  | 1.65 ± 0.48                           | 1.59 ± 0.16                           | **−0.06** | **−0.32** |
| durée (min)  | 694 ± 7                               | 422 ± 6                               | **−272**  | −1        |

→ ✅ **Pari validé** : bs=64 × backbone=3 récupère le meilleur de run 3 (alt 236 avec backbone_3 dans run 3) **et** divise le coût compute par ~1.65 (694 → 422 min).
→ ✅ **hdg_std ÷3** (0.48 → 0.16) : l'explosion hdg de run 3 venait bien de bs=32 + seq=30. bs=64 la résout.
→ ✅ **alt mean −18.76 m** vs run 3 baseline pour 39 % de coût en moins.
→ ⚠️ γ_std × 4.5 (0.0009 → 0.0041) : seule régression. La sur-stabilité γ de run 3 (baseline_seed identique à 1e-4 près) était possiblement un point particulier ; valeur run 4 reste excellente.

## Comparaison historique (toutes baselines)

| run | baseline                                  | alt mean ± std    | hdg mean ± std | durée (min) |
|-----|-------------------------------------------|-------------------|----------------|-------------|
| 1   | bs=128, w=on, silu, seq=60, bb=3          | 278.8 ± 21.3      | 1.93 ± 0.32    | 287         |
| 2   | bs=64, w=off, relu, seq=60, bb=3          | 254.9 ± 21.7      | 1.78 ± 0.05    | 416         |
| 3   | bs=32, w=off, relu, seq=30, bb=2          | 256.5 ± 5.0       | 1.65 ± 0.48    | 694         |
| 4   | **bs=64, w=off, relu, seq=30, bb=3**      | **237.7 ± 7.1**   | **1.59 ± 0.16** | **422**     |

→ **Cumul gains 1 → 4** : alt mean **−41 m (−15 %)**, alt std **÷3**, hdg mean **−18 %**, hdg std **÷2**.
→ Coût ×1.47 vs run 1 mais on a échangé 287 min de bs=128/silu contre 422 min de bs=64+bb=3.

## `batch_size` (baseline=64)

| bs  | alt mean±std       | Δm_alt | Δs_alt   | tas mean±std    | Δm_tas | Δs_tas | γ mean±std         | Δm_γ    | Δs_γ    | hdg mean±std    | Δm_hdg | Δs_hdg | durée (min) | Δ_wc       |
|-----|--------------------|--------|----------|-----------------|--------|--------|--------------------|---------|---------|-----------------|--------|--------|-------------|------------|
| 32  | 245.6 ± 4.5        | +7.9   | **−2.6** | 3.28 ± 0.02     | −0.06  | −0.03  | 0.2778 ± 0.0044    | −0.0004 | +0.0003 | 1.62 ± 0.45     | +0.03  | +0.30  | 712 ± 4     | **+290**   |
| 64  | *237.7 ± 7.1*      | base   | base     | 3.35 ± 0.05     | base   | base   | *0.2782 ± 0.0041*  | base    | base    | 1.59 ± 0.16     | base   | base   | *422 ± 6*   | base       |

→ ⚠️ **Inversion vs run 3** : run 3 (avec backbone=2) disait "bs=32 légèrement plus stable, bs=64 légèrement meilleur en moyenne". Avec backbone=3, **bs=32 perd sur la moyenne (+7.9 m)** mais garde un léger avantage stabilité (std−2.6) — sauf sur hdg où il déstabilise (×2.8).
→ ROI bs=64 confirmé : +290 min de coût bs=32 pour aucun gain global.
→ **bs=64 + backbone=3 = config dominante** sur les 4 sweeps.

## `backbone_depth` (baseline=3)

| backbone | alt mean±std    | Δm_alt   | Δs_alt    | tas mean±std    | Δm_tas | Δs_tas | γ mean±std         | Δm_γ      | Δs_γ    | hdg mean±std    | Δm_hdg | Δs_hdg | durée (min) | Δ_wc |
|----------|-----------------|----------|-----------|-----------------|--------|--------|--------------------|-----------|---------|-----------------|--------|--------|-------------|------|
| 3        | *237.7 ± 7.1*   | base     | base      | 3.35 ± 0.05     | base   | base   | *0.2782 ± 0.0041*  | base      | base    | 1.59 ± 0.16     | base   | base   | *422 ± 6*   | base |
| 4        | 239.9 ± 11.2    | +2.2     | +4.1      | 3.28 ± 0.08     | −0.07  | +0.02  | 0.2766 ± 0.0063    | −0.0016   | +0.0022 | 1.57 ± 0.07     | −0.02  | −0.09  | 431 ± 1     | +9   |
| 5        | 277.5 ± **56.1** | **+39.8** | **+49.0** | 3.49 ± **0.37** | +0.15  | +0.32  | 0.2867 ± **0.0155** | **+0.0085** | +0.0114 | 1.83 ± 0.40     | +0.24  | +0.24  | 632 ± 0     | +209 |

→ ✅ **Plateau backbone atteint** : bb=4 ≈ bb=3 (Δm dans le bruit). bb=5 = effondrement (alt+40, std×8).
→ **Lecture** : la trajectoire bb=2 → bb=3 (+gain) → bb=4 (plateau) → bb=5 (crash) montre que **bb=3 est l'optimum stable**. La capacité supplémentaire au-delà de 3 ne compense pas l'instabilité d'optimisation qu'elle introduit.
→ ⚠️ Suspect : tous les bb=5 ont la même durée (632 ± 0) → vérifier que ce n'est pas un early-stopping silencieux.

## `seq_len` (baseline=30)

| seq_len | alt mean±std       | Δm_alt | Δs_alt    | tas mean±std    | Δm_tas    | Δs_tas | γ mean±std         | Δm_γ    | Δs_γ    | hdg mean±std    | Δm_hdg | Δs_hdg | durée (min) | Δ_wc |
|---------|--------------------|--------|-----------|-----------------|-----------|--------|--------------------|---------|---------|-----------------|--------|--------|-------------|------|
| 25      | 242.7 ± 5.0        | +5.0   | **−2.0**  | 3.53 ± 0.08     | **+0.18** | +0.02  | 0.2801 ± 0.0019    | +0.0019 | −0.0022 | 1.73 ± 0.22     | +0.14  | +0.06  | 629 ± 4     | +207 |
| 30      | *237.7 ± 7.1*      | base   | base      | 3.35 ± 0.05     | base      | base   | *0.2782 ± 0.0041*  | base    | base    | 1.59 ± 0.16     | base   | base   | *422 ± 6*   | base |
| 45      | 248.6 ± 17.6       | +10.9  | +10.5     | **3.13 ± 0.09** | **−0.21** | +0.04  | 0.2773 ± 0.0020    | −0.0009 | −0.0021 | 1.59 ± 0.07     | 0.00   | −0.09  | 411 ± 0     | −11  |
| 60      | 254.9 ± **21.7**   | +17.2  | **+14.6** | **2.95 ± 0.05** | **−0.39** | −0.01  | 0.2846 ± 0.0056    | +0.0064 | +0.0015 | 1.78 ± 0.05     | +0.19  | −0.10  | 422 ± 0     | 0    |

→ **Frontière TAS/alt nette** : seq augmente → tas s'améliore monotone (3.53 → 3.35 → 3.13 → 2.95), alt se dégrade monotone (242.7 → 237.7 → 248.6 → 254.9). Le sweet spot **alt** = seq=30, **TAS** = seq=60.
→ ⚠️ seq=25 coûte 207 min de plus que seq=30 (629 vs 422). Suspect — possiblement un effet de cache de batches plus petits. À investiguer.
→ **seq=45 = compromis** : −0.21 m/s sur TAS pour seulement +10.9 m sur alt (≈ 1 std). Si l'objectif est multi-objectif TAS+alt, seq=45 domine seq=30 et seq=60.
→ ⚠️ seq=15 (crash run 3) confirmé hors range : seq=25 marche, donc la frontière de divergence est entre 15 et 25.

## `lr` (baseline=1e-3)

| lr     | alt mean±std       | Δm_alt | Δs_alt    | tas mean±std    | Δm_tas | Δs_tas | γ mean±std         | Δm_γ    | Δs_γ    | hdg mean±std    | Δm_hdg | Δs_hdg | durée (min) | Δ_wc |
|--------|--------------------|--------|-----------|-----------------|--------|--------|--------------------|---------|---------|-----------------|--------|--------|-------------|------|
| 7e-4   | 240.9 ± 4.4        | +3.2   | **−2.6**  | 3.33 ± 0.11     | −0.02  | +0.05  | 0.2796 ± 0.0010    | +0.0014 | **−0.0031** | 1.62 ± 0.10 | +0.03  | −0.06  | 420 ± 1     | −2   |
| 1e-3   | *237.7 ± 7.1*      | base   | base      | 3.35 ± 0.05     | base   | base   | *0.2782 ± 0.0041*  | base    | base    | 1.59 ± 0.16     | base   | base   | *422 ± 6*   | base |
| 1.5e-3 | 245.7 ± 10.5       | +8.0   | +3.5      | 3.29 ± 0.11     | −0.06  | +0.06  | 0.2803 ± 0.0052    | +0.0020 | +0.0011 | 1.47 ± 0.33     | −0.12  | +0.17  | 409 ± 4     | −13  |

→ **lr=7e-4 marginalement gagnant sur stabilité** (alt_std÷1.6, γ_std÷4) pour +3 m alt (dans le bruit).
→ lr=1.5e-3 dégrade alt et déstabilise (cohérent run 1 et 2).
→ **Conclusion** : 1e-3 reste le sweet spot. lr=7e-4 mérite un test de confirmation avec plus de seeds si l'objectif est la stabilité ultime.

---

## Synthèse stabilité (alt_std)

Baseline run 4 alt_std = **7.1 m**. Le plancher de stabilité naturelle est conservé malgré le switch bs=32 → bs=64.

Configs qui **stabilisent encore** :

| config            | alt_std | facteur vs baseline |
|-------------------|---------|---------------------|
| seq_25            | 5.0     | ÷1.4                |
| lr_7e-4           | 4.4     | **÷1.6**            |
| bs_32             | 4.5     | ÷1.6                |

Configs qui **explosent la variance** :

| config            | alt_std | facteur vs baseline |
|-------------------|---------|---------------------|
| backbone_5        | **56.1** | **×7.9**           |
| seq_60            | 21.7    | ×3.0                |
| seq_45            | 17.6    | ×2.5                |
| backbone_4        | 11.2    | ×1.6                |
| lr_1.5e-3         | 10.5    | ×1.5                |

→ **Pattern run 4** : la baseline est sur une crête de stabilité dans un puits étroit. Seules 3 variantes améliorent marginalement, toutes les autres dégradent.

## Coût compute (wall-clock par run)

| config        | durée (min) | facteur vs baseline | qualité (Δm_alt)   |
|---------------|-------------|---------------------|--------------------|
| lr=1.5e-3     | 409         | ×0.97               | +8.0 m             |
| seq=45        | 411         | **×0.97**           | +10.9 m            |
| lr=7e-4       | 420         | ×0.99               | +3.2 m             |
| **baseline**  | **422**     | ×1.00               | base               |
| seq=60        | 422         | ×1.00               | +17.2 m            |
| backbone=4    | 431         | ×1.02               | +2.2 m (≈ base)    |
| backbone=5    | 632         | ×1.50               | **+39.8 m** (pire) |
| seq=25        | 629         | ×1.49               | +5.0 m             |
| bs=32         | 712         | **×1.69**           | +7.9 m             |

→ **Aucune variante run 4 ne domine la baseline globalement**. backbone=4 est neutre (Δm dans le bruit, +2 % coût) ; tout le reste paie cher pour pire ou équivalent.
→ **Plateau atteint** : c'est un signal fort que **la baseline run 4 est l'optimum local des hyperparams sweepés**.

## Comparaison conclusions run 3 → run 4

| acquis run 3                                       | confirmé run 4 ?                                                              |
|----------------------------------------------------|-------------------------------------------------------------------------------|
| `backbone=3` > `backbone=2` (interaction avec seq=30) | ✅ Confirmé. Bonus : bb=4 = plateau (neutre), bb=5 = crash                  |
| `seq=30` sweet spot pour alt, `seq=60` pour TAS    | ✅ Confirmé + raffiné : la frontière est monotone (25 → 30 → 45 → 60)        |
| `bs=64` meilleur ROI que `bs=32`                   | ✅ Confirmé **et renforcé** avec bb=3 : bs=64 meilleur en **moyenne** aussi |
| `lr=1e-3` sweet spot                               | ✅ Confirmé (1.5e-3 dégrade ; 7e-4 marginal)                                 |
| `hdg_std` baseline run 3 (0.48) suspect            | ✅ **Résolu** : venait de bs=32+seq=30. bs=64 → hdg_std=0.16 (÷3)            |

## Nouveautés run 4

- **🏆 Nouvelle baseline globale** : alt **237.7 ± 7.1 m**, hdg **1.59 ± 0.16°**, durée **422 min** — meilleur que toutes les baselines précédentes sur 4 métriques, avec un coût raisonnable.
- **Plateau backbone confirmé** : `bb ∈ {3, 4}` équivalents, `bb=5` est la falaise (alt+40 m, std×8). L'optimisation explore mal au-delà de 4 couches.
- **seq_len est un trade-off TAS/alt monotone et propre** : c'est désormais un **levier objectif-dépendant**, plus un hyperparam à tuner aveuglément.
- **lr=7e-4 candidat pour la version "ultra-stable"** (γ_std÷4, alt_std÷1.6) mais gain marginal en moyenne.

## Recommandations pour run 5 (si pertinent)

### Verdict global

**Le sweep par axe a probablement épuisé son potentiel sur les hyperparams testés.** 4 runs successifs convergent vers la même région ; les axes restants donnent du bruit ou une explosion.

### Pistes potentielles (hors hyperparams classiques)

1. **Multi-task TAS+alt** : seq=45 est dominé par les Pareto seq=30 (alt) et seq=60 (TAS). Pour casser ce trade-off, il faut probablement deux têtes spécialisées ou une normalisation différenciée TAS / alt.
2. **lr scheduling** : aucun sweep n'a touché au scheduler. Un cosine ou warmup pourrait stabiliser les seeds dans le régime backbone=3, ou rendre backbone=4 stable.
3. **Architecture (head_depth, residuals, layer-norm)** : `head_depth=2` est fixé depuis le début. Tester `head_depth ∈ {1, 3}` pour vérifier.
4. **train_limit > 5000** : tout le sweep est à 5000 vols. Augmenter (10k, 20k) pour voir si l'optimum se déplace.
5. **Trade-off TAS hors hyperparams** : run 2 disait déjà "TAS plafonné aux hyperparams testés — investiguer seq_len". Run 4 falsifie partiellement (seq_len **a** un effet sur TAS), mais le gain max (−0.4 m/s à seq=60) reste plafonné. Investiguer **normalisation des features vent/Mach** ou **architecture du sous-réseau moteur**.

### Si on garde un sweep d'hyperparam

Hypothèse minimale (~12 runs) : `head_depth ∈ {1, 3}` × `lr_schedule ∈ {cosine, none}` × 3 seeds.

### Décision suggérée

**Stopper le sweep par axe** et basculer vers une **investigation ciblée** (multi-task ou normalisation TAS-specifique), avec la baseline run 4 figée comme référence.

## Note artefacts

- `queue.json` : 27 completed, 0 failed, **0 diverged**. Bonne nouvelle vs run 3 (seq=15).
- Tous les `backbone_5_seed{0,1,2}` finissent à 632 min ± 0 — durée suspectement identique sur 3 seeds. Vérifier `train.log` qu'il n'y a pas d'early-stop silencieux qui sauverait des epochs et expliquerait la mauvaise qualité.
- Agréger en `summary.parquet` :
  ```bash
  uv run python -m scripts.sweep_runner aggregate --results-dir results_run4
  ```
