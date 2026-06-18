# Система оцінювання стану здоров'я рослин

Десктоп-застосунок (Tkinter) для класифікації стану здоров'я рослин за біосенсорними показниками за допомогою нейронної мережі (MLP), з порівнянням проти класичних ML-моделей (RandomForest, SVM, GradientBoosting).

## Що робить проєкт

Система приймає 8 біосенсорних показників рослини та класифікує її стан на одну з 4 категорій:

- 🔴 Критичний стрес
- 🟠 Помірний стрес
- 🟢 Легкий стрес
- 🟢 Здорова рослина

Вхідні ознаки:

| Ознака | Опис |
|---|---|
| `fluorescence_intensity` | Флуоресценція |
| `colorimetric_index` | Колориметричний індекс |
| `spr_signal_strength` | SPR-сигнал |
| `leaf_temperature` | Температура листа (°C) |
| `chlorophyll_content` | Вміст хлорофілу |
| `moisture_level` | Рівень вологості |
| `light_absorption_ratio` | Поглинання світла |
| `volatile_organic_compounds` | Леткі органічні сполуки (VOC) |

## Структура проєкту

```
plant_health_nn/
├── app.py                          # Tkinter GUI: вкладки Інфо / Прогноз / Навчання / Пакетна обробка
├── analysis_logic.py               # Бізнес-логіка класифікації, обчислення норм та діапазонів
├── config.yaml                     # Конфігурація даних, моделі та навчання
├── split_dataset.py                # Розбиття CSV на train/test (80/20, stratify)
├── test_train.py                   # Швидке тестове навчання MLP (3 епохи) для перевірки пайплайну
├── plant_health_biosensor_dataset.csv  # Датасет біосенсорних вимірювань
├── run_app.bat                     # Запуск застосунку на Windows
├── requirements.txt
│
├── preprocessing/
│   ├── loader.py                   # Завантаження CSV, авто-визначення цільової колонки та ознак
│   ├── scaler.py                   # StandardScaler / MinMaxScaler
│   ├── splitter.py                 # train_test_split за конфігом (stratify)
│   └── augmentor.py                # SMOTE для балансування класів
│
├── models/
│   ├── mlp_model.py                # MLP: Dense(128→64→32) + BN + Dropout + L2
│   ├── cnn_model.py                 # 1D-CNN альтернатива (Conv1D → BN → GAP → Dense)
│   ├── baseline_ml.py               # RandomForest, SVM, GradientBoosting
│   ├── trainer.py                   # Тренування з class_weight, EarlyStopping, ReduceLROnPlateau
│   └── predictor.py                 # Інференс збереженої моделі
│
├── evaluation/
│   ├── metrics.py                   # accuracy, f1 (weighted/macro/micro), precision/recall, ROC-AUC
│   ├── confusion.py                 # Матриця помилок (абс. + нормалізована)
│   ├── cross_val.py                 # 5-fold StratifiedKFold крос-валідація
│   ├── arch_comparison.py           # Порівняння архітектур (MLP vs CNN vs baselines)
│   ├── error_analysis.py            # Аналіз помилкових передбачень
│   └── reporter.py                  # Збір усіх метрик у results/report.json
│
├── visualization/
│   ├── plots.py                     # Графіки навчання (loss/accuracy)
│   ├── roc_curves.py                # ROC-криві по класах
│   └── feature_imp.py               # Важливість ознак
│
├── utils/
│   └── seed.py                      # Фіксація seed (Python/NumPy/TensorFlow) для відтворюваності
│
├── saved_models/                    # Навчені моделі (mlp_best.keras, scaler.pkl, *.pkl baselines)
└── results/                         # Графіки, report.json, історія передбачень
```

## Встановлення

Потрібен Python 3.11.

```bash
pip install -r requirements.txt
```

Залежності: `tensorflow`, `scikit-learn`, `imbalanced-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `pyyaml`, `joblib`.

## Запуск

**GUI-застосунок:**

```bash
python app.py
```

На Windows можна також просто запустити `run_app.bat` — він сам встановить залежності та відкриє застосунок.

Застосунок має вкладки:
- **Інфо** — нормальні діапазони показників, метрики з останнього звіту, поточна конфігурація
- **Прогноз** — введення значень показників (вручну або з пресетів класів) і отримання класифікації з ймовірностями та аналізом чутливості
- **Навчання** — запуск навчання MLP з вибором CSV, логом у реальному часі та графіком прогресу
- **Пакетна обробка** — класифікація CSV-файлу з кількома зразками одночасно та експорт результатів

**Швидка перевірка пайплайну навчання (3 епохи):**

```bash
python test_train.py
```

**Розбиття датасету на train/test CSV:**

```bash
python split_dataset.py
```

## Конфігурація (`config.yaml`)

```yaml
data:
  path: plant_health_biosensor_dataset.csv
  test_size: 0.2
  random_state: 42
model:
  type: mlp
  hidden_layers: [128, 64, 32]
  dropout: 0.3
  activation: relu
preprocessing:
  scaler: standard
  apply_smote: true
training:
  epochs: 20
  batch_size: 32
  learning_rate: 0.001
  validation_split: 0.2
  patience: 7
```

Розбиття даних: 80/20 (train/test) через `train_test_split` зі стратифікацією; валідаційна крива на графіках навчання будується через `validation_split=0.2` у Keras.

## Підхід до моделювання

- **Балансування класів**: SMOTE застосовується лише до тренувальної вибірки (`smote_only_to_train: true`), плюс додаткове зважування класів (`class_weight`) під час навчання MLP.
- **Архітектура MLP**: три приховані шари (128→64→32) з BatchNormalization, Dropout (0.3) та L2-регуляризацією (λ=1e-4).
- **Регуляризація навчання**: EarlyStopping (patience=7) та ReduceLROnPlateau відстежують `val_loss`.
- **Порівняння моделей**: MLP порівнюється з 1D-CNN та класичними алгоритмами (RandomForest, SVM, GradientBoosting) — результати в `results/arch_comparison.png` та `results/report.json`.
- **Відтворюваність**: фіксований seed (42) для Python, NumPy та TensorFlow (`utils/seed.py`).

5-fold крос-валідація MLP: accuracy 0.888 ± 0.013.

### Порівняння архітектур: MLP vs CNN

`evaluation/arch_comparison.py` тренує MLP та 1D-CNN (`models/cnn_model.py`) на однакових даних і порівнює їх за accuracy, F1, ROC-AUC, часом навчання та кількістю параметрів. Результат збережено в `results/arch_comparison.png` та секції `arch_comparison` у `results/report.json`:

| Архітектура | Accuracy | F1 (weighted) | F1 (macro) | ROC-AUC | Час навчання | Параметри | Епох |
|---|---|---|---|---|---|---|---|
| MLP | 0.825 | 0.836 | 0.738 | 0.963 | 7.4 с | 12 516 | 20 |
| CNN (1D) | 0.733 | 0.739 | 0.602 | 0.891 | 7.6 с | 34 244 | 20 |

На цих даних MLP перевершує 1D-CNN за всіма метриками якості, маючи при цьому втричі менше параметрів і приблизно той самий час навчання. Це очікувано: ознаки тут — це 8 незалежних табличних біосенсорних показників без природного просторового чи часового порядку, тож згорткові шари CNN (розраховані на виявлення локальних залежностей між сусідніми ознаками) не дають переваги, а лише додають складності моделі. Саме тому фінальна модель, що зберігається й використовується в GUI (`saved_models/mlp_best.keras`), — це MLP, а CNN залишається лише як референс для порівняння архітектур.

## Артефакти результатів

У `results/` зберігаються згенеровані графіки та звіти:
- `training_history_mlp.png` — крива навчання MLP
- `confusion_matrix.png` — матриця помилок
- `roc_curves.png` — ROC-криві по класах
- `arch_comparison.png` — порівняння архітектур
- `error_analysis.png` — аналіз помилкових прогнозів
- `smote_comparison.png` — вплив SMOTE на розподіл класів
- `correlation.png` — кореляція ознак
- `report.json` — повний звіт з метриками всіх моделей
- `prediction_history.json` — історія передбачень із GUI
