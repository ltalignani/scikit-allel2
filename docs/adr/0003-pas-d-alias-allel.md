---
status: accepted
---

# Pas d'alias d'import `allel`

scikit-allel2 s'importe sous le nom `allel2`. Nous avons décidé (2026-10-26) de ne **pas** livrer de paquet `allel` qui réexporterait `allel2`, même si cela aurait rendu la migration transparente. Un tel paquet occuperait le même nom d'import que scikit-allel. Or les deux coexistent souvent dans un même environnement, par exemple via malariagen_data, qui dépend de scikit-allel. Selon l'ordre d'installation, `import allel` chargerait alors l'un ou l'autre sans avertissement, et pip ne détecte pas ce conflit entre deux distributions différentes. La promesse de remplaçant direct tient en une ligne : seul l'import change, `import allel2 as allel`.

## Considered Options

- **Alias `allel` livré avec scikit-allel2** : rejeté. La migration était sans effort, mais au prix d'un écrasement silencieux de scikit-allel et de résultats qui dépendent de l'ordre d'installation.
- **Alias en option (extra `scikit-allel2[alias]`)** : rejeté. Le risque reste le même pour qui active l'extra, et cela ajoute une variante à maintenir et à documenter.

## Consequences

- Les utilisateurs remplacent `import allel` par `import allel2 as allel`. La page « Migrer depuis scikit-allel » et les notes de version de la 2.0.0 le montrent en premier.
- scikit-allel et scikit-allel2 peuvent être installés ensemble sans conflit.
- Revenir sur cette décision plus tard reste possible, mais introduirait un conflit chez les utilisateurs qui ont les deux paquets installés.

Détail : [Quelle stratégie de release pour la première version PyPI ?](https://github.com/ltalignani/scikit-allel2/issues/15)
