import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, cohen_kappa_score, matthews_corrcoef,
                             classification_report, confusion_matrix)
from sklearn.calibration import calibration_curve
import joblib

# Попробуем импортировать top_k_accuracy_score
try:
    from sklearn.metrics import top_k_accuracy_score

    TOP_K_AVAILABLE = True
except ImportError:
    TOP_K_AVAILABLE = False
    print("Предупреждение: top_k_accuracy_score не доступна (обновите scikit-learn).")

# Загрузка и предобработка данных
df = pd.read_csv('intestinal_infections_treatment_dataset.csv')

target_cols = ['t_hospitalization', 't_iv_fluids', 't_oral_rehydration',
               't_antibiotics', 't_probiotics', 't_antidiarrheals']

feature_cols = [c for c in df.columns if c not in target_cols + ['patient_id']]
X = df[feature_cols].copy()
y = df[target_cols].copy()

# Сохраняем медианы числовых признаков для замены пропусков позже
numeric_medians = X.select_dtypes(include=[np.number]).median().to_dict()

# Заполнение пропусков
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].fillna('unknown')
    else:
        X[col] = X[col].fillna(X[col].median())

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y.iloc[:, 0])

# Модель лечения (multi‑output)
model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# Модель для предсказания возбудителя
pathogen_encoder = LabelEncoder()
y_pathogen = pathogen_encoder.fit_transform(df['stool_pathogen'])

# Разделяем с тем же random_state, что и основные данные
_, _, y_pathogen_train, y_pathogen_test = train_test_split(
    X, y_pathogen, test_size=0.2, random_state=42, stratify=y_pathogen
)
pathogen_model = RandomForestClassifier(random_state=42)
pathogen_model.fit(X_train, y_pathogen_train)

# Оценка качества моделей
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

target_names = ['hospitalization', 'iv_fluids', 'oral_rehydration',
                'antibiotics', 'probiotics', 'antidiarrheals']

metrics_results = {}

print("Метрики для каждого исхода\n")
for i, name in enumerate(target_names):
    y_true = y_test.iloc[:, i].values
    y_pred_class = y_pred[:, i]

    acc = accuracy_score(y_true, y_pred_class)
    prec = precision_score(y_true, y_pred_class, zero_division=0)
    rec = recall_score(y_true, y_pred_class, zero_division=0)
    f1 = f1_score(y_true, y_pred_class, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred_class)
    mcc = matthews_corrcoef(y_true, y_pred_class)
    auc = roc_auc_score(y_true, y_pred_proba[i][:, 1])

    metrics_results[name] = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'kappa': kappa,
        'mcc': mcc,
        'roc_auc': auc
    }

    print(f"{name.upper()}:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  F1-score: {f1:.4f}")
    print(f"  Cohen's Kappa: {kappa:.4f}")
    print(f"  Matthews CC: {mcc:.4f}")
    print(f"  ROC-AUC: {auc:.4f}\n")

# Усреднённые метрики
print("Усредненные метрики:\n")
macro_acc = np.mean([metrics_results[t]['accuracy'] for t in target_names])
macro_prec = np.mean([metrics_results[t]['precision'] for t in target_names])
macro_rec = np.mean([metrics_results[t]['recall'] for t in target_names])
macro_f1 = np.mean([metrics_results[t]['f1'] for t in target_names])
macro_kappa = np.mean([metrics_results[t]['kappa'] for t in target_names])
macro_mcc = np.mean([metrics_results[t]['mcc'] for t in target_names])
macro_auc = np.mean([metrics_results[t]['roc_auc'] for t in target_names])

print(f"Macro Accuracy: {macro_acc:.4f}")
print(f"Macro Precision: {macro_prec:.4f}")
print(f"Macro Recall: {macro_rec:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")
print(f"Macro Kappa: {macro_kappa:.4f}")
print(f"Macro MCC: {macro_mcc:.4f}")
print(f"Macro ROC-AUC: {macro_auc:.4f}\n")

# Классификация возбудителя
y_path_true = y_pathogen_test
y_path_pred = pathogen_model.predict(X_test)

print("Классификация возбудителя\n")
print(classification_report(y_path_true, y_path_pred,
                            target_names=pathogen_encoder.classes_, zero_division=0))

# Top-3 accuracy для возбудителя
if TOP_K_AVAILABLE:
    try:
        y_path_proba = pathogen_model.predict_proba(X_test)
        top3_acc = top_k_accuracy_score(y_path_true, y_path_proba, k=3)
        print(f"Top-3 accuracy для возбудителя: {top3_acc:.4f}\n")
        metrics_results['pathogen_top3_acc'] = top3_acc
    except Exception as e:
        print(f"Top-3 accuracy не вычислена: {e}\n")
else:
    print("Top-3 accuracy не поддерживается (обновите scikit-learn)\n")

# Калибровочная кривая для исхода 'antibiotics' (пример)
idx_ab = target_names.index('antibiotics')
y_true_ab = y_test.iloc[:, idx_ab].values
y_proba_ab = y_pred_proba[idx_ab][:, 1]

prob_true, prob_pred = calibration_curve(y_true_ab, y_proba_ab, n_bins=5)
plt.figure(figsize=(6, 6))
plt.plot(prob_pred, prob_true, marker='o', label='Модель')
plt.plot([0, 1], [0, 1], linestyle='--', label='Идеальная калибровка')
plt.xlabel('Предсказанная вероятность')
plt.ylabel('Истинная частота')
plt.title('Калибровочная кривая: назначение антибиотиков')
plt.legend()
plt.savefig('calibration_curve.png', dpi=150)
plt.close()
print("Калибровочная кривая сохранена как calibration_curve.png\n")

# Сохраняем все компоненты модели
joblib.dump(model, 'treatment_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_dict, 'label_encoders.pkl')
joblib.dump(numeric_medians, 'numeric_medians.pkl')
joblib.dump(num_cols, 'num_cols.pkl')
joblib.dump(cat_cols, 'cat_cols.pkl')
joblib.dump(pathogen_model, 'pathogen_model.pkl')
joblib.dump(pathogen_encoder, 'pathogen_encoder.pkl')

# Сохраняем метрики в JSON
with open('model_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics_results, f, ensure_ascii=False, indent=2)