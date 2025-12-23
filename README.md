# League of Legends - Anàlisi Predictiu i Data Mining

**Projecte d'Aprenentatge Computacional**  
Universitat de Barcelona - Curs 2024/2025

[English](#english) | [Español](#español)

---

## 📋 Descripció del Projecte

Aquest projecte explora la predicció de resultats en partides de League of Legends utilitzant tècniques de Machine Learning i Data Mining. Més enllà de la predicció en si, l'objectiu principal és demostrar la importància de la **selecció d'atributs** i l'anàlisi de dades en el rendiment dels models.

El dataset utilitzat prové de Kaggle i conté informació detallada sobre partides classificatòries de League of Legends, incloent-hi estadístiques d'objectius, eliminacions, or acumulat i altres mètriques de joc.

## 🎯 Objectius

1. **Data Mining**: Identificar quins atributs són realment rellevants per la predicció
2. **Anàlisi de Correlació**: Estudiar com la correlació entre variables afecta el rendiment del model
3. **Optimització de Features**: Demostrar que menys dades, però ben seleccionades, poden superar models amb totes les variables
4. **Comparació de Models**: Avaluar diferents algoritmes de classificació sobre el mateix dataset

## 📂 Estructura del Repositori

```
├── LOL - Dataset/           # Dataset original de Kaggle
├── Anàlisi_de_dades/        # Notebooks d'exploració i visualització
├── Selecció de Model/       # Entrenament i comparació de models
└── Descripció Dades.txt     # Documentació dels atributs
```

## 🔍 Procés de Data Mining

### 1. Exploració Inicial
El dataset original conté més de 60 atributs, incloent-hi:
- Estadístiques d'objectius (torres, dracs, barons)
- Mètriques individuals (or, eliminacions, assists)
- Informació temporal (duració de partida)
- Esdeveniments clau (primer sang, primera torre)

### 2. Anàlisi de Correlació
Mitjançant matrius de correlació, s'identifiquen:
- **Atributs altament correlacionats** amb el resultat (winner)
- **Redundància entre variables**: atributs que aporten informació similar
- **Soroll**: variables amb alta desviació estàndard i baixa correlació

**Descoberta clau**: Atributs com `towerKills` i objectius d'equip mostren les correlacions més fortes, mentre que mètriques individuals solen ser menys predictives.

### 3. Selecció d'Atributs
Del conjunt original de 60+ atributs, es redueix a aproximadament **11-15 features clau**:
- Eliminacions de torres (team1TowerKills, team2TowerKills)
- Objectius majors (dracs, barons, heralds)
- Avantatge d'or acumulat
- Esdeveniments crítics (firstBlood, firstTower)

**Resultat**: Els models amb features seleccionades aconsegueixen millor precisió i generalització que models amb tots els atributs.

### 4. Comparació de Models
S'han entrenat i comparat diversos algoritmes:
- Decision Trees
- Random Forest
- Support Vector Machines (SVM)
- Regressió Logística

## 📊 Resultats Principals

- La **selecció estratègica d'atributs** millora tant la precisió com l'eficiència computacional
- Les **torres destruïdes** són el millor predictor individual del resultat
- La **correlació entre variables** pot introduir soroll: menys pot ser més
- Els models entrenen més ràpid i generalitzen millor amb features curades

## 🛠️ Tecnologies Utilitzades

- **Python 3.x**
- **Pandas** & **NumPy**: manipulació de dades
- **Scikit-learn**: models de ML i mètriques
- **Matplotlib** & **Seaborn**: visualització
- **Jupyter Notebook**: entorn de desenvolupament

## 📈 Com Executar el Projecte

1. Clona el repositori:
```bash
git clone https://github.com/NIU1710545/Kaggle-Code.git
cd Kaggle-Code
```

2. Instal·la les dependències:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

3. Obre els notebooks:
```bash
jupyter notebook
```

4. Navega a `Anàlisi_de_dades/` per explorar el procés de data mining

## 💡 Lliçons Apreses

- **Qualitat sobre quantitat**: Un conjunt curat d'atributs supera un conjunt complet
- **Context del domini**: Entendre el joc (LoL) ajuda a identificar features rellevants
- **Iteració**: El data mining és un procés iteratiu d'anàlisi i refinament
- **Generalització**: Models més simples amb bones features eviten overfitting

## 📚 Referències

- Dataset original: [Kaggle - League of Legends Ranked Games](https://www.kaggle.com/)
- Inspiració: Diversos notebooks de la comunitat de Kaggle sobre predicció en LoL

## 👤 Autor

**Laia Lishuang Orús Vázquez**  
NIU: 1710545  
Universitat de Barcelona

---

*Aquest projecte ha estat desenvolupat com a part de l'assignatura d'Aprenentatge Computacional. L'objectiu principal és demostrar la importància del data mining i la selecció d'atributs en el desenvolupament de models predictius.*

---
---

<a name="english"></a>
# League of Legends - Predictive Analysis and Data Mining

**Computational Learning Project**  
University of Barcelona - Academic Year 2024/2025

[Català](#league-of-legends---anàlisi-predictiu-i-data-mining) | [Español](#español)

---

## 📋 Project Description

This project explores outcome prediction in League of Legends matches using Machine Learning and Data Mining techniques. Beyond prediction itself, the main objective is to demonstrate the importance of **feature selection** and data analysis in model performance.

The dataset comes from Kaggle and contains detailed information about ranked League of Legends matches, including objective statistics, kills, accumulated gold, and other game metrics.

## 🎯 Objectives

1. **Data Mining**: Identify which attributes are truly relevant for prediction
2. **Correlation Analysis**: Study how correlation between variables affects model performance
3. **Feature Optimization**: Demonstrate that less data, but well-selected, can outperform models with all variables
4. **Model Comparison**: Evaluate different classification algorithms on the same dataset

## 📂 Repository Structure

```
├── LOL - Dataset/           # Original Kaggle dataset
├── Anàlisi_de_dades/        # Exploration and visualization notebooks
├── Selecció de Model/       # Model training and comparison
└── Descripció Dades.txt     # Attribute documentation
```

## 🔍 Data Mining Process

### 1. Initial Exploration
The original dataset contains over 60 attributes, including:
- Objective statistics (towers, dragons, barons)
- Individual metrics (gold, kills, assists)
- Temporal information (match duration)
- Key events (first blood, first tower)

### 2. Correlation Analysis
Through correlation matrices, we identify:
- **Highly correlated attributes** with the outcome (winner)
- **Redundancy between variables**: attributes that provide similar information
- **Noise**: variables with high standard deviation and low correlation

**Key finding**: Attributes like `towerKills` and team objectives show the strongest correlations, while individual metrics tend to be less predictive.

### 3. Feature Selection
From the original set of 60+ attributes, we reduce to approximately **11-15 key features**:
- Tower kills (team1TowerKills, team2TowerKills)
- Major objectives (dragons, barons, heralds)
- Accumulated gold advantage
- Critical events (firstBlood, firstTower)

**Result**: Models with selected features achieve better accuracy and generalization than models with all attributes.

### 4. Model Comparison
Several algorithms have been trained and compared:
- Decision Trees
- Random Forest
- Support Vector Machines (SVM)
- Logistic Regression

## 📊 Main Results

- **Strategic feature selection** improves both accuracy and computational efficiency
- **Destroyed towers** are the best individual predictor of the outcome
- **Correlation between variables** can introduce noise: less can be more
- Models train faster and generalize better with curated features

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas** & **NumPy**: data manipulation
- **Scikit-learn**: ML models and metrics
- **Matplotlib** & **Seaborn**: visualization
- **Jupyter Notebook**: development environment

## 📈 How to Run the Project

1. Clone the repository:
```bash
git clone https://github.com/NIU1710545/Kaggle-Code.git
cd Kaggle-Code
```

2. Install dependencies:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

3. Open the notebooks:
```bash
jupyter notebook
```

4. Navigate to `Anàlisi_de_dades/` to explore the data mining process

## 💡 Lessons Learned

- **Quality over quantity**: A curated set of attributes outperforms a complete set
- **Domain context**: Understanding the game (LoL) helps identify relevant features
- **Iteration**: Data mining is an iterative process of analysis and refinement
- **Generalization**: Simpler models with good features avoid overfitting

## 📚 References

- Original dataset: [Kaggle - League of Legends Ranked Games](https://www.kaggle.com/)
- Inspiration: Various Kaggle community notebooks on LoL prediction

## 👤 Author

**Laia Lishuang Orús Vázquez**  
NIU: 1710545  
University of Barcelona

---

*This project was developed as part of the Computational Learning course. The main objective is to demonstrate the importance of data mining and feature selection in developing predictive models.*

---
---

<a name="español"></a>
# League of Legends - Análisis Predictivo y Data Mining

**Proyecto de Aprendizaje Computacional**  
Universidad de Barcelona - Curso 2024/2025

[Català](#league-of-legends---anàlisi-predictiu-i-data-mining) | [English](#english)

---

## 📋 Descripción del Proyecto

Este proyecto explora la predicción de resultados en partidas de League of Legends utilizando técnicas de Machine Learning y Data Mining. Más allá de la predicción en sí, el objetivo principal es demostrar la importancia de la **selección de atributos** y el análisis de datos en el rendimiento de los modelos.

El dataset utilizado proviene de Kaggle y contiene información detallada sobre partidas clasificatorias de League of Legends, incluyendo estadísticas de objetivos, eliminaciones, oro acumulado y otras métricas de juego.

## 🎯 Objetivos

1. **Data Mining**: Identificar qué atributos son realmente relevantes para la predicción
2. **Análisis de Correlación**: Estudiar cómo la correlación entre variables afecta el rendimiento del modelo
3. **Optimización de Features**: Demostrar que menos datos, pero bien seleccionados, pueden superar modelos con todas las variables
4. **Comparación de Modelos**: Evaluar diferentes algoritmos de clasificación sobre el mismo dataset

## 📂 Estructura del Repositorio

```
├── LOL - Dataset/           # Dataset original de Kaggle
├── Anàlisi_de_dades/        # Notebooks de exploración y visualización
├── Selecció de Model/       # Entrenamiento y comparación de modelos
└── Descripció Dades.txt     # Documentación de los atributos
```

## 🔍 Proceso de Data Mining

### 1. Exploración Inicial
El dataset original contiene más de 60 atributos, incluyendo:
- Estadísticas de objetivos (torres, dragones, barones)
- Métricas individuales (oro, eliminaciones, asistencias)
- Información temporal (duración de partida)
- Eventos clave (primera sangre, primera torre)

### 2. Análisis de Correlación
Mediante matrices de correlación, se identifican:
- **Atributos altamente correlacionados** con el resultado (winner)
- **Redundancia entre variables**: atributos que aportan información similar
- **Ruido**: variables con alta desviación estándar y baja correlación

**Descubrimiento clave**: Atributos como `towerKills` y objetivos de equipo muestran las correlaciones más fuertes, mientras que métricas individuales suelen ser menos predictivas.

### 3. Selección de Atributos
Del conjunto original de 60+ atributos, se reduce a aproximadamente **11-15 features clave**:
- Eliminaciones de torres (team1TowerKills, team2TowerKills)
- Objetivos mayores (dragones, barones, heralds)
- Ventaja de oro acumulado
- Eventos críticos (firstBlood, firstTower)

**Resultado**: Los modelos con features seleccionadas consiguen mejor precisión y generalización que modelos con todos los atributos.

### 4. Comparación de Modelos
Se han entrenado y comparado varios algoritmos:
- Decision Trees
- Random Forest
- Support Vector Machines (SVM)
- Regresión Logística

## 📊 Resultados Principales

- La **selección estratégica de atributos** mejora tanto la precisión como la eficiencia computacional
- Las **torres destruidas** son el mejor predictor individual del resultado
- La **correlación entre variables** puede introducir ruido: menos puede ser más
- Los modelos entrenan más rápido y generalizan mejor con features curadas

## 🛠️ Tecnologías Utilizadas

- **Python 3.x**
- **Pandas** & **NumPy**: manipulación de datos
- **Scikit-learn**: modelos de ML y métricas
- **Matplotlib** & **Seaborn**: visualización
- **Jupyter Notebook**: entorno de desarrollo

## 📈 Cómo Ejecutar el Proyecto

1. Clona el repositorio:
```bash
git clone https://github.com/NIU1710545/Kaggle-Code.git
cd Kaggle-Code
```

2. Instala las dependencias:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

3. Abre los notebooks:
```bash
jupyter notebook
```

4. Navega a `Anàlisi_de_dades/` para explorar el proceso de data mining

## 💡 Lecciones Aprendidas

- **Calidad sobre cantidad**: Un conjunto curado de atributos supera un conjunto completo
- **Contexto del dominio**: Entender el juego (LoL) ayuda a identificar features relevantes
- **Iteración**: El data mining es un proceso iterativo de análisis y refinamiento
- **Generalización**: Modelos más simples con buenas features evitan overfitting

## 📚 Referencias

- Dataset original: [Kaggle - League of Legends Ranked Games](https://www.kaggle.com/)
- Inspiración: Diversos notebooks de la comunidad de Kaggle sobre predicción en LoL

## 👤 Autor

**Laia Lishuang Orús Vázquez**  
NIU: 1710545  
Universidad de Barcelona

---

*Este proyecto ha sido desarrollado como parte de la asignatura de Aprendizaje Computacional. El objetivo principal es demostrar la importancia del data mining y la selección de atributos en el desarrollo de modelos predictivos.*
