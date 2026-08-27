# Darwin HOR 3

Évolution darwinienne de **profils de pale d’éolienne à axe horizontal**.

Réécriture moderne de `Darwin_HORV2.sci` (Stéphane Roddier, IUT GIM Saint-Denis) : plus de Scilab, plus de `xfoil.exe` Windows.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=sroddier/darwin-hor&branch=main&mainModule=app.py)

## Utiliser en ligne (sans installer)

1. Ouvre [ce lien de déploiement Streamlit Cloud](https://share.streamlit.io/deploy?repository=sroddier/darwin-hor&branch=main&mainModule=app.py) (compte GitHub, gratuit).
2. Choisis **Python 3.11** ou **3.12** (pas 3.13 : CasADi n’a pas toujours de roue).
3. Clique **Deploy**. L’URL publique ressemble à `https://darwin-hor.streamlit.app`.

Ensuite n’importe qui ouvre l’URL : aucun Scilab, aucun `xfoil.exe`.

Code source : [github.com/sroddier/darwin-hor](https://github.com/sroddier/darwin-hor)

- Aéro : [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil) (réseau entraîné sur des millions de polaires XFOIL)
- Géométrie : NACA 4 chiffres + mutations type XFOIL GDES + CST (Kulfan)
- Interface : Streamlit
- Export : `.dat` Selig, `.sldcrv` SolidWorks (corde 100 mm comme la V2), DXF, SVG, pale STL (BEM Schmitz)

Licence MIT — utilisation libre et ouverte.

## Lancer

Windows (double-clic ou invite de commandes) :

```bat
run.bat
```

À la main, partout (Windows / Mac / Linux) :

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -e ".[dev]"
streamlit run app.py
```

Évolution en console, sans navigateur :

```bash
python -m darwin_hor --cli --generations 10 --pop 8 --seed 42
```

## Ce qui est repris de la V2

Chaque génération :

1. injecte des **NACA 4 chiffres aléatoires** (M, P, xx ∈ 10–21)
2. mute chaque survivant avec un opérateur tiré au hasard
3. évalue une **polaire visqueuse** (Re = 3×10⁵, Mach = 0.2, α = 0…11° par défaut)
4. tue les profils trop épais, et ceux sous la médiane (5 % des faibles survivent, 1 % des bons meurent)

Opérateurs, équivalents XFOIL :

| V2 | V3 |
|---|---|
| `INTER` (40–60 %) | interpolation réelle des géométries |
| `TGAP` | épaisseur de bord de fuite |
| `LERA` | poids de bord d’attaque CST |
| `FLAP` | volet de bord de fuite |
| — | mutation gaussienne CST (nouveau) |

### Correctifs par rapport à `Darwin_HORV2.sci`

1. **Croisement cassé** : `mfprintf(..., "%d", a)` avec `a ∈ [0.4, 0.6]` écrivait `0` dans le script XFOIL. Le mélange 40–60 % est maintenant un vrai flottant.
2. **Fitness** : la V2 faisait `sum(|Cl/Cd|)`, ce qui favorise les polaires avec plus de points convergents. Le mode par défaut est la **moyenne Cl/Cd pondérée par la confiance NeuralFoil**. Le mode `classique` reproduit la somme.
3. **Épaisseur** : le dialogue V2 demandait `22`, puis comparait `max(y)` à `11` sur une corde unitaire — le filtre ne s’activait jamais. Ici **22 % = 22 % de corde**.
4. **Plus de hangs XFOIL** (`killxfoil.bat`) : NeuralFoil répond toujours, avec un indice de confiance.

## Pale 3D

À partir du champion, une pale est loftée avec les formules de **Schmitz** (cours IUT) :

- φ(r) = (2/3) arctan(1/λ_r)
- c(r) = 8π r (1 − cos φ) / (B Cl)
- β(r) = φ − α_design

Export STL prêt à visualiser / découper ; ce n’est pas un CFD 3D.

## Structure

```
darwin-hor/
  app.py                 interface Streamlit
  darwin_hor/            géométrie, solver, évolution, export, pale
  tests/
  run.bat
```

Comparer un ancien profil V2 : onglet **Comparer**, importer un `.g1` ou `Le meilleurNACA7513.dat`.

## Tests

```bash
pytest
```

## Limites

NeuralFoil approxime XFOIL, ce n’est pas un calcul Navier–Stokes. Pour un podium de concours, revalider le champion dans XFOIL, XFLR5 ou un CFD. La pale Schmitz ignore le décrochage 3D, les pertes de bout et la structure.
