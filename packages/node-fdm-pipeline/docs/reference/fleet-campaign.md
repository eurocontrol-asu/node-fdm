# Campagnes de flotte

Le point d’entrée public `run_fleet_campaign` centralise la préparation, le contrôle et
l’exécution d’une campagne de flotte à partir d’un fichier de configuration.

```python
from pathlib import Path

from node_fdm_pipeline import run_fleet_campaign

report = run_fleet_campaign(Path("campaign.yaml"), "plan")
```

## Modes

`run_fleet_campaign(config, mode)` accepte cinq modes :

- `plan` renvoie directement le `CampaignPlan` produit par `campaign_plan(config)` ;
- `status` renvoie directement le `CampaignReport` produit par
  `campaign_status(config)` ;
- `validate` renvoie directement le `CampaignReport` produit par
  `campaign_validate(config)` ;
- `run` vérifie l’identité durable, l’enregistre pour un nouvel état, puis exécute les
  jours du plan ;
- `resume` vérifie la compatibilité de l’identité et ne reprogramme que les jours dont
  une étape reste incomplète.

Les modes `run` et `resume` renvoient un `CampaignRunReport`. Une valeur de mode
inconnue lève `UnknownCampaignMode`.

## Préflight et état durable

Avant toute exécution active, le point d’entrée charge la sélection enregistrée et appelle
`preflight_campaign`. La configuration `fleet_run` doit fournir au minimum :

- `lease_path` et `min_free_gib` pour les garde-fous d’exécution ;
- `recorded_source` pour la sélection locale et reproductible ;
- `acquisition_journal` et `acquisition_receipt_dir` pour l’état durable.

Une nouvelle exécution contre une identité incompatible lève
`CampaignStateIncompatible` avant toute écriture dans le journal. Une reprise exige une
identité existante et compatible. L’identité de campagne est conservée à côté du fichier de
configuration ; les chemins relatifs du journal, des reçus et de la sélection sont résolus
depuis ce même répertoire.

## Reprise

La reprise reconstruit l’avancement à partir du journal. Un jour portant un événement
`cleanup_completed` est considéré terminé ; `next_incomplete_step` détermine si le jour
doit être reprogrammé. Ainsi, un jour déjà terminé ne reçoit pas de second événement
`day_started`.
