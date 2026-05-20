# Sweep run 3 — analyse intra-axe

27 runs · 9 axes × 3 seeds · `train_limit=5000`.

**Baseline run 3** (cumule les acquis run 2 figés : `bs=32`, `seq_len=30`, `backbone_depth=2`)

`bs=32, lr=1e-3, weighting=off, activation=relu, epochs=50, seq_len=30, hidden_width=48, backbone_depth=2, head_depth=2`

- alt = **256.48 ± 5.03 m**
- tas = **3.307 ± 0.082 m/s**
- γ   = **0.2791 ± 0.0009 rad**
- hdg = **1.65 ± 0.48°**
- durée = **694 ± 7 min** (wall-clock, train+predict+evaluate sur 1 GPU)

Notation : `Δm` = écart de la moyenne vs baseline (− = mieux). `Δs` = écart de l'écart-type vs baseline (− = plus stable).

## Comparaison baselines run 2 → run 3

| métrique     | run 2 baseline (bs=64, seq=60, bb=3) | run 3 baseline (bs=32, seq=30, bb=2) | Δm        | Δs           |
|--------------|---------------------------------------|---------------------------------------|-----------|--------------|
| mae_alt      | 254.92 ± 21.69                        | 256.48 ± 5.03                         | +1.6      | **−16.7**    |
| mae_tas      | 2.953 ± 0.047                         | 3.307 ± 0.082                         | **+0.354** | +0.035       |
| mae_gamma    | 0.2846 ± 0.0056                       | 0.2791 ± 0.0009                       | **−0.0055** | **−0.0047** |
| mae_heading  | 1.78 ± 0.05                           | 1.65 ± 0.48                           | −0.13     | **+0.43**    |
| durée (min)  | 416 ± 4                               | 694 ± 7                               | +278      | +3           |

→ ✅ **Stabilité alt confirmée** : les 3 stabilisateurs (bs=32, seq=30, backbone=2) cumulés divisent alt_std par **4.3** (21.7 → 5.0), proche de l'estimation indépendante (~2 m).
→ ✅ **γ doublement gagnant** : meilleur en moyenne (−0.0055) **et** std÷6 (0.0056 → 0.0009).
→ ⚠️ **TAS confirmé pénalisé par seq=30** (+0.35 m/s, +12 %). Le coût TAS de seq=30 vu en run 2 (+0.39 m/s) tient.
→ ⚠️ **hdg_std explose** (×10 : 0.05 → 0.48). La stabilité hdg quasi-parfaite de run 2 ne tient pas avec bs=32.
→ Coût wall-clock : +278 min, dominé par bs=32 (×1.67 vs bs=64).

## `batch_size` (baseline=32)

| bs  | alt mean±std       | Δm_alt | Δs_alt   | tas mean±std    | Δm_tas | Δs_tas | γ mean±std         | Δm_γ    | Δs_γ    | hdg mean±std    | Δm_hdg | Δs_hdg | durée (min) | Δ_wc       |
|-----|--------------------|--------|----------|-----------------|--------|--------|--------------------|---------|---------|-----------------|--------|--------|-------------|------------|
| 16  | 262.1 ± 22.6       | +5.6   | **+17.6** | 3.46 ± 0.13   | +0.16  | +0.04  | 0.2818 ± 0.0029    | +0.0027 | +0.0020 | 1.38 ± 0.14     | −0.27  | −0.34  | 1268 ± 4    | **+574**   |
| 32  | *256.5 ± 5.0*      | base   | base     | 3.31 ± 0.08     | base   | base   | *0.2791 ± 0.0009*  | base    | base    | 1.65 ± 0.48     | base   | base   | *694 ± 7*   | base       |
| 64  | **252.4 ± 8.4**    | **−4.0** | +3.4   | 3.33 ± 0.03     | +0.02  | −0.06  | 0.2801 ± 0.0084    | +0.0010 | +0.0075 | 1.46 ± 0.16     | −0.19  | **−0.32** | 411 ± 1     | **−283**   |

→ **bs=64 redevient compétitif** sur run 3 : alt mean meilleur (−4 m), hdg meilleur (−0.19), coût ÷1.7 (411 vs 694 min). Mais alt_std×1.7 (5.0 → 8.4) et γ_std×9.3.
→ **bs=16 catastrophique** : pire qu'avant sur alt, std×4.5, et +574 min de coût. À éliminer.
→ **Inversion vs run 2** : run 2 disait "bs=32 domine bs=64 sur la stabilité". Avec seq=30+bb=2, bs=32 garde le titre stabilité, mais l'écart se réduit fortement (5 vs 8.4 au lieu de 4.2 vs 21.7). Le **vrai compromis ROI** devient bs=64.

## `seq_len` (baseline=30)

| seq_len | alt mean±std     | Δm_alt | Δs_alt    | tas mean±std    | Δm_tas    | Δs_tas | γ mean±std         | Δm_γ    | Δs_γ    | hdg mean±std    | Δm_hdg | Δs_hdg | durée (min) | Δ_wc |
|---------|------------------|--------|-----------|-----------------|-----------|--------|--------------------|---------|---------|-----------------|--------|--------|-------------|------|
| **15**  | ⚠️ **8468 ± 0**  | +8212  | n/a       | ⚠️ 198.7 ± 0    | +195.4    | n/a    | ⚠️ 15.34 ± 0       | +15.06  | n/a     | ⚠️ 71.1 ± 0     | +69.4  | n/a    | 672 ± 4     | −22  |
| 30      | *256.5 ± 5.0*    | base   | base      | 3.31 ± 0.08     | base      | base   | *0.2791 ± 0.0009*  | base    | base    | 1.65 ± 0.48     | base   | base   | *694 ± 7*   | base |
| 60      | 256.8 ± 17.1     | +0.3   | **+12.1** | **2.95 ± 0.10** | **−0.35** | +0.02  | 0.2795 ± 0.0074    | +0.0004 | +0.0065 | 1.55 ± 0.11     | −0.11  | **−0.37** | 679 ± 1     | −15  |

→ ⚠️ **seq_15 = échec total**. Les 3 seeds divergent en `nan` durant le training (`nan_or_inf_loss` warnings, `best_val_loss=inf`, prédictions constantes identiques sur les 3 seeds : alt=8468, tas=199). **15 timesteps est trop court** pour que le NODE apprenne la dynamique. À éliminer définitivement.
→ **seq_60 récupère le TAS** (−0.35 m/s, équivalent à la baseline run 2) sans dégrader alt (+0.3 m, dans le bruit), au prix de alt_std×3.4 et γ_std×8.
→ Confirme l'**arbitrage seq=30 vs seq=60** vu en run 2 : seq=30 trade TAS contre stabilité, seq=60 inverse.

## `backbone_depth` (baseline=2)

| backbone | alt mean±std    | Δm_alt   | Δs_alt   | tas mean±std    | Δm_tas | Δs_tas | γ mean±std         | Δm_γ      | Δs_γ    | hdg mean±std    | Δm_hdg | Δs_hdg | durée (min) | Δ_wc |
|----------|-----------------|----------|----------|-----------------|--------|--------|--------------------|-----------|---------|-----------------|--------|--------|-------------|------|
| 2        | *256.5 ± 5.0*   | base     | base     | 3.31 ± 0.08     | base   | base   | *0.2791 ± 0.0009*  | base      | base    | 1.65 ± 0.48     | base   | base   | *694 ± 7*   | base |
| 3        | **236.1 ± 3.5** | **−20.4** | **−1.5** | 3.29 ± 0.12     | −0.02  | +0.03  | **0.2763 ± 0.0063** | **−0.0028** | +0.0054 | 1.48 ± 0.14     | **−0.17** | **−0.34** | 735 ± 1     | +42  |

→ **🏆 Plus gros gain alt du sweep** : **−20 m** vs baseline, alt_std encore **mieux** (3.5 vs 5.0), hdg amélioré, γ amélioré, coût marginal (+42 min).
→ ⚠️ **Inversion totale vs run 2** : run 2 disait "backbone=2 est dominé sur la moyenne, backbone=3 déstabilise (std×1.6)". Avec seq=30+bs=32, **backbone=3 est dominant sur tout sauf le coût**.
→ **Lecture** : la profondeur backbone interagit avec seq_len. Sur sequences longues (seq=60), backbone=2 suffit ; sur sequences courtes (seq=30), backbone=3 récupère la capacité perdue.

## `hidden_width` (baseline=48)

| hidden | alt mean±std       | Δm_alt | Δs_alt    | tas mean±std    | Δm_tas | Δs_tas | γ mean±std         | Δm_γ    | Δs_γ    | hdg mean±std    | Δm_hdg | Δs_hdg | durée (min) | Δ_wc |
|--------|--------------------|--------|-----------|-----------------|--------|--------|--------------------|---------|---------|-----------------|--------|--------|-------------|------|
| 32     | 259.1 ± 21.9       | +2.7   | **+16.9** | 3.38 ± 0.09     | +0.07  | +0.01  | 0.2844 ± 0.0041    | +0.0053 | +0.0032 | 1.67 ± 0.24     | +0.01  | −0.24  | 689 ± 3     | −5   |
| 48     | *256.5 ± 5.0*      | base   | base      | 3.31 ± 0.08     | base   | base   | *0.2791 ± 0.0009*  | base    | base    | 1.65 ± 0.48     | base   | base   | *694 ± 7*   | base |
| 64     | 259.1 ± 17.2       | +2.6   | **+12.2** | 3.45 ± 0.09     | +0.15  | +0.00  | 0.2832 ± 0.0046    | +0.0041 | +0.0037 | 1.63 ± 0.03     | −0.03  | **−0.45** | 672 ± 10    | −22  |

→ **48 reste optimal** : descendre OU monter dégrade alt (+2.6 m) **et** déstabilise (std×3.4 à ×4.4).
→ Cohérent avec run 2 (hidden=24 dégrade alt, hidden=96 explose). La sweet spot 48 est confirmée sur 2 sweeps.
→ hidden=64 stabilise hdg (std÷16) sans contrepartie sur la moyenne → utile **uniquement** si l'objectif est purement latéral.

## `activation` (baseline=relu)

| act  | alt mean±std    | Δm_alt | Δs_alt   | tas mean±std    | Δm_tas | Δs_tas | γ mean±std         | Δm_γ    | Δs_γ    | hdg mean±std    | Δm_hdg | Δs_hdg | durée (min) | Δ_wc |
|------|-----------------|--------|----------|-----------------|--------|--------|--------------------|---------|---------|-----------------|--------|--------|-------------|------|
| relu | *256.5 ± 5.0*   | base   | base     | 3.31 ± 0.08     | base   | base   | *0.2791 ± 0.0009*  | base    | base    | 1.65 ± 0.48     | base   | base   | *694 ± 7*   | base |
| gelu | 263.7 ± 10.4    | +7.2   | +5.4     | 3.53 ± 0.18     | +0.22  | +0.10  | 0.2849 ± 0.0055    | +0.0058 | +0.0046 | 1.49 ± 0.07     | **−0.16** | **−0.41** | 702 ± 6     | +8   |

→ **Inversion vs run 2** : run 2 avait `gelu` ~équivalent à `relu` en moyenne, ÷5 sur std. Run 3 : `gelu` est **pire** sur alt (+7.2 m), tas (+0.22), γ (+0.0058) et alt_std×2. Seul gain : hdg (std×0.15).
→ Avec seq=30+bs=32+bb=2 (régime déjà stable), gelu n'apporte plus le bénéfice stabilisateur observé en run 2. relu reste le bon choix.

---

## Synthèse stabilité (alt_std)

Baseline run 3 alt_std = **5.0 m** — le plancher de stabilité naturelle est quasi atteint avec les 3 stabilisateurs cumulés.

Configs qui **stabilisent encore plus** :

| config            | alt_std | facteur vs baseline |
|-------------------|---------|---------------------|
| backbone_depth=3  | **3.5** | **÷1.4**            |

→ **Un seul axe** sur 8 améliore encore alt_std. Le régime baseline (5 m) est proche du plancher inter-seed.

Configs qui **font remonter la variance** (la majorité) :

| config            | alt_std | facteur vs baseline |
|-------------------|---------|---------------------|
| bs=16             | 22.6    | ×4.5                |
| hidden_width=32   | 21.9    | ×4.4                |
| hidden_width=64   | 17.2    | ×3.4                |
| seq_len=60        | 17.1    | ×3.4                |
| act=gelu          | 10.4    | ×2.1                |
| bs=64             | 8.4     | ×1.7                |

→ **Pattern run 3** : presque tout déstabilise depuis ce point d'équilibre. La baseline run 3 est sur une crête.

## Coût compute (wall-clock par run)

| config        | durée (min) | facteur vs baseline | qualité (Δm_alt)   |
|---------------|-------------|---------------------|--------------------|
| bs=64         | 411         | **×0.59**           | **−4.0 m**         |
| seq=60        | 679         | ×0.98               | +0.3 m             |
| hidden=64     | 672         | ×0.97               | +2.6 m             |
| seq=15        | 672         | ×0.97               | ⚠️ diverged        |
| hidden=32     | 689         | ×0.99               | +2.7 m             |
| **baseline**  | **694**     | ×1.00               | base               |
| act=gelu      | 702         | ×1.01               | +7.2 m             |
| backbone=3    | 735         | ×1.06               | **−20.4 m** (best) |
| bs=16         | **1268**    | **×1.83**           | +5.6 m             |

→ **🏆 backbone=3 = ROI du run 3** : meilleur gain qualité (−20 m) **et** stabilité (std÷1.4) pour +6 % de coût (+42 min).
→ **bs=64 = quasi-gratuit** : qualité ~équivalente (−4 m, dans le bruit), 41 % du coût en moins (−283 min).
→ **bs=16 doublement perdant** : pire qualité **et** ×1.83 coût.

## Comparaison conclusions run 2 → run 3

| acquis run 2                                       | confirmé run 3 ?                                                              |
|----------------------------------------------------|-------------------------------------------------------------------------------|
| `bs=32` + `seq=30` + `backbone=2` cumule la stabilité | ✅ **Vérifié** : alt_std passe de 21.7 → 5.0 (÷4.3, prédiction d'effets indépendants tenue) |
| `seq=30` dégrade TAS (+0.39 m/s)                   | ✅ Confirmé : seq=30 vs seq=60 dans run 3 = +0.35 m/s                          |
| `gelu` ≈ `relu` en moyenne, ÷5 sur std             | ❌ **Falsifié** sur run 3 : gelu pire mean (+7 m) **et** moins stable (×2)     |
| `backbone=2` dominé en mean, gagnant en stabilité  | ❌ **Inversé** : avec seq=30, backbone=3 gagne **et** stabilise               |
| `bs=32` domine bs=64                               | ⚠️ Nuancé : bs=32 garde la stabilité, bs=64 redevient compétitif sur la moyenne et économise 283 min |
| `hidden=48` sweet spot                             | ✅ Reconfirmé (24, 32, 64, 96 tous dégradent)                                  |
| `lr=1e-3` sweet spot                               | ✅ Non re-testé (drop justifié, sweet spot consolidé sur run 1+2)              |

## Nouveautés run 3

- **🏆 backbone=3 + seq=30 + bs=32 = meilleur config global** (alt 236 ± 3.5). C'est −43 m vs run 1 baseline (278.8), −19 m vs run 2 baseline (254.9), avec un std divisé par **6** par rapport à run 1.
- **L'interaction `seq_len × backbone_depth`** existe et a un signe : sequences courtes (seq=30) **bénéficient** d'un backbone plus profond pour compenser la perte d'information temporelle.
- **seq_len=15 est un cliff** : passer de 30 à 15 timesteps fait diverger le training (`nan_or_inf_loss`). La frontière inférieure de seq_len est entre 15 et 30.
- **gelu n'est pas universellement stabilisateur** : son bénéfice run 2 venait d'une interaction avec bs=64+bb=3+seq=60 ; il disparaît sur la baseline run 3.

## Recommandations pour run 4

### Nouvelle baseline retenue (run 4)

`bs=64, lr=1e-3, weighting=off, relu, seq_len=30, hidden=48, backbone=3, head_depth=2`
- Choix `bs=64` (vs `bs=32`) : qualité ~équivalente (−4 m alt sur run 3, dans le bruit) pour −40 % de coût compute. `bs=32` est gardé comme axe de contrôle de stabilité.
- Estimé : alt ~240 ± 8 m (extrapolation `backbone=3` × `bs=64`), durée ~430 min/run.

### Axes prioritaires à explorer

1. **`backbone_depth` ∈ {4, 5}** — l'inversion run 2 → run 3 sur backbone suggère qu'on n'a pas atteint le plateau. backbone=3 gagne encore vs 2 ; tester 4 et 5 avant de conclure.
2. **Interaction `seq=60 × backbone=3`** — récupérer le TAS sans perdre la profondeur. Sur run 3, seq=60+bb=2 ramène TAS à 2.95 mais perd alt_std. Avec bb=3 (capacité supplémentaire), peut-on garder les deux ?
3. **`seq_len` ∈ {20, 25, 45}** — affiner la frontière utile entre seq=15 (crash) et seq=30 (TAS dégradé). seq=20 ou 25 pourrait diverger encore ; seq=45 pourrait être le compromis TAS/alt.
4. **`bs=64` + nouvelle baseline (backbone=3)** — si bs=64 garde son ROI (−4 m, −283 min) avec backbone=3, c'est la **vraie nouvelle baseline**.
5. **`lr` à re-tester sur backbone=3** — la fenêtre étroite confirmée sur run 1+2 (1e-3 optimal) a été établie sur backbone=2/3 ; backbone plus profond peut décaler le sweet spot lr.

### Axes à abandonner définitivement

| axe abandonné       | raison                                                                  |
|---------------------|-------------------------------------------------------------------------|
| `bs=16`             | pire qualité **et** ×1.83 coût (run 3)                                  |
| `bs=128`, `bs=256`  | dominés depuis run 1                                                    |
| `seq=15`            | divergence NaN (run 3)                                                  |
| `seq=120`           | dominé (run 2)                                                          |
| `hidden ∈ {24, 32, 64, 96}` | sweet spot 48 confirmé sur 2 sweeps                              |
| `act=gelu`          | inversion run 3 — pas universel                                         |
| `lr ∈ {5e-4, 3e-3}` | dominé / divergent (run 1)                                              |
| `α ∈ {0.3, 0.7}`, `weighting=on` | dominés par weighting=off depuis run 1                    |
| `epochs=100`        | overfit (run 1)                                                         |

### Matrice run 4 (figée)

| axe          | variants     | configs |
|--------------|--------------|---------|
| baseline (bs=64, bb=3, seq=30) | — | 1       |
| `bs`         | 32           | 1       |
| `backbone`   | 4, 5         | 2       |
| `seq_len`    | 25, 45, 60   | 3       |
| `lr`         | 7e-4, 1.5e-3 | 2       |

→ 9 configs × 3 seeds = **27 runs**.

Vu durée extrapolée (~430 min/run baseline bs=64+bb=3, ~700 min sur bs_32), coût estimé **~210 h sur 1 GPU, ~53 h sur 4 slots**.

Implémentation : `scripts.sweep_matrix.run4_matrix()`. Lancement : `uv run python -m scripts.sweep_runner run --run4 --gpus 0,1 --per-gpu 2`.

### Hypothèse à valider hors sweep

L'**explosion de hdg_std** sur la baseline run 3 (0.48 vs 0.05 en run 2) est suspecte : bs=32 + seq=30 amplifient le bruit hdg inter-seed. À investiguer : refaire 3 seeds de `(bs=64, seq=30, bb=2)` pour isoler si hdg_std vient de bs=32, de seq=30, ou de l'interaction des deux.

## Note artefacts

- `queue.json` : 27 completed, 0 failed (seq_15 a "completed" malgré le NaN — le pipeline ne détecte pas la divergence).
- `seq_15_seed{0,1,2}/train.log` : warnings `nan_or_inf_loss` pendant tout l'entraînement, `best_val_loss=inf`. À ajouter au sweep_runner : flag "diverged" si `best_val_loss=inf` ou `nan` dans metrics.
- Agréger en `summary.parquet` :
  ```bash
  uv run python -m scripts.sweep_runner aggregate --results-dir results_run3
  ```
