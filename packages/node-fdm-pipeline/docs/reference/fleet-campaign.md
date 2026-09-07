# Campagnes de flotte

Le point d’entrée public `run_fleet_campaign` centralise la préparation, le contrôle et
l’exécution d’une campagne de flotte à partir d’un fichier de configuration.

```python
from pathlib import Path

from node_fdm_pipeline import run_fleet_campaign

report = run_fleet_campaign(
    Path("campaign.yaml"),
    "run",
    only_steps=["decode"],
)
```

## Modes

`run_fleet_campaign(config, mode, *, only_steps=None)` accepte cinq modes :

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

Pour une exécution active, `only_steps` limite le traitement aux étapes nommées. Les
valeurs acceptées sont `download`, `decode` et `enrich`; l’ordre fourni est conservé et
les doublons sont éliminés. Une valeur inconnue lève l’exception publique
`UnknownCampaignStep`, dont le message rappelle la liste acceptée. Le rapport expose les
étapes retenues et le journal écrit un événement `campaign_step_completed` pour chacune.

## Préflight et état durable

Avant toute exécution active, le point d’entrée charge la sélection enregistrée et appelle
`preflight_campaign`. La configuration `fleet_run` doit fournir au minimum :

- `lease_path` et `min_free_gib` pour les garde-fous d’exécution (`0` désactive
  explicitement le seuil d’espace libre) ;
- `recorded_source` pour la sélection locale et reproductible ;
- `acquisition_journal` et `acquisition_receipt_dir` pour l’état durable.

Une nouvelle exécution contre une identité incompatible lève
`CampaignStateIncompatible` avant toute écriture dans le journal. Une reprise exige une
identité existante et compatible. L’identité de campagne est conservée à côté du fichier de
configuration ; les chemins relatifs du journal, des reçus et de la sélection sont résolus
depuis ce même répertoire.

## Commandes historiques

`fdm download-fleet`, `fdm decode-fleet` et `fdm enrich-fleet` restent disponibles pendant
la période de migration. Chaque invocation écrit exactement un avis de dépréciation sur
stderr, renvoie vers `fdm fleet-campaign run`, puis utilise le même contrat central avec
respectivement `only_steps=["download"]`, `only_steps=["decode"]` ou
`only_steps=["enrich"]`. Les configurations historiques qui ne déclarent pas encore
`fleet_run` conservent temporairement leur comportement antérieur.

## Reprise

La reprise reconstruit l’avancement à partir du journal. Un jour portant un événement
`cleanup_completed` est considéré terminé ; `next_incomplete_step` détermine si le jour
doit être reprogrammé. Ainsi, un jour déjà terminé ne reçoit pas de second événement
`day_started`.
