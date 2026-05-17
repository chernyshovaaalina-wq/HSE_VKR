from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json
import os
import re

# RAG dependencies
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# Импорт класса для поиска аналогов
from medicine import EsklpAnalogFinder

app = Flask(__name__)
CORS(app)

# Загрузка компонентов модели лечения
model = joblib.load('treatment_model.pkl')
scaler = joblib.load('scaler.pkl')
le_dict = joblib.load('label_encoders.pkl')
numeric_medians = joblib.load('numeric_medians.pkl')
num_cols = joblib.load('num_cols.pkl')
cat_cols = joblib.load('cat_cols.pkl')
pathogen_model = joblib.load('pathogen_model.pkl')
pathogen_encoder = joblib.load('pathogen_encoder.pkl')


# Загрузка правил из JSON
def load_json_file(filename, default=None):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Предупреждение: файл {filename} не найден.")
        return default if default is not None else {}

clinical_rules = load_json_file('clinical_rules.json', {"red_flags": []})
antibiotic_alternatives_config = load_json_file('antibiotic_alternatives.json', {})

# Загрузка классификатора МКБ-10
mkb_data = []
def load_mkb():
    global mkb_data
    try:
        df = pd.read_excel('mkb_10.xls', sheet_name='МКБ_10', dtype=str)
        df = df.dropna(subset=['Код', 'Наименование'])
        df['Код'] = df['Код'].str.strip()
        df['Наименование'] = df['Наименование'].str.strip()
        mkb_data = df[['Код', 'Наименование']].to_dict('records')
        print(f"Загружено {len(mkb_data)} записей МКБ-10")
    except Exception as e:
        print(f"Ошибка загрузки МКБ: {e}")
        mkb_data = []
load_mkb()

@app.route('/api/mkb')
def get_mkb():
    return jsonify(mkb_data)

# Загрузка клинических критериев (для дорожной карты)
treatment_criteria = []
def load_treatment_criteria():
    global treatment_criteria
    try:
        with open('enriched_criteria_dataset_validated.json', 'r', encoding='utf-8') as f:
            treatment_criteria = json.load(f)
        print(f"Загружено {len(treatment_criteria)} критериев лечения")
        icd_codes = set(c.get('icd10_code', 'unknown') for c in treatment_criteria)
        print(f"Уникальные коды МКБ в файле: {icd_codes}")
    except FileNotFoundError:
        print("Предупреждение: файл enriched_criteria_dataset_validated.json не найден.")
        treatment_criteria = []
    except Exception as e:
        print(f"Ошибка загрузки критериев: {e}")
        treatment_criteria = []
load_treatment_criteria()

def matches_patient(criterion, patient_icd, care_level, patient_group):
    if criterion.get('care_level') != care_level:
        return False
    if criterion.get('patient_group') != patient_group:
        return False
    icd_criterion = criterion.get('icd10_code', '')
    if not icd_criterion or icd_criterion == 'unknown':
        return True
    patient_icd_clean = patient_icd.strip().upper()
    codes = [c.strip().upper() for c in icd_criterion.split(',')]
    for code in codes:
        if '-' in code:
            start, end = code.split('-')
            if patient_icd_clean >= start and patient_icd_clean <= end:
                return True
        else:
            if patient_icd_clean == code:
                return True
            if patient_icd_clean.startswith(code + '.'):
                return True
    return False

def check_conditions(conditions, patient_data):
    return True

def build_treatment_plan(patient_icd, care_level, patient_group, patient_data=None):
    plan = {'diagnostic': [], 'therapeutic': [], 'consultation': [], 'monitoring': [], 'other': []}
    if not treatment_criteria:
        return plan
    sorted_criteria = sorted(treatment_criteria, key=lambda x: (x.get('section_id', ''), x.get('row_number', 0)))
    for crit in sorted_criteria:
        if not matches_patient(crit, patient_icd, care_level, patient_group):
            continue
        if not check_conditions(crit.get('conditions', []), patient_data):
            continue
        text = crit.get('full_criterion') or crit.get('description', '')
        if not text:
            continue
        category = crit.get('category', 'other')
        timeframe = crit.get('timeframe')
        if timeframe:
            text += f" (срок: {timeframe})"
        plan[category].append(text)
    for cat in plan:
        plan[cat] = list(dict.fromkeys(plan[cat]))
    return plan

# AI-анализ (модель лечения кишечных инфекций)
def get_diagnosis(pred_pathogen, has_blood=False, has_mucus=False):
    if pred_pathogen == 'salmonella':
        return "Сальмонеллёз (A02)"
    elif pred_pathogen == 'shigella':
        return "Шигеллёз (дизентерия) (A03)"
    elif pred_pathogen == 'e_coli':
        return "Эшерихиоз (A04.0-A04.4)"
    elif pred_pathogen == 'campylobacter':
        return "Кампилобактериоз (A04.5)"
    elif pred_pathogen == 'viral':
        return "Острый вирусный гастроэнтерит (A08)"
    else:
        if has_blood:
            return "Инвазивная диарея неуточнённая (A04.9)"
        elif has_mucus:
            return "Колит неуточнённый (K52.9)"
        else:
            return "Острая кишечная инфекция неуточнённая (A09)"

def check_red_flags(patient):
    red_flags = []
    total_score = 0
    for rule in clinical_rules.get('red_flags', []):
        attr = patient.get(rule['attribute'])
        if attr is None:
            continue
        condition_met = False
        op = rule.get('operator', '==')
        value = rule.get('value')
        if op == '>':
            condition_met = attr > value
        elif op == '<':
            condition_met = attr < value
        elif op == '>=':
            condition_met = attr >= value
        elif op == '<=':
            condition_met = attr <= value
        elif op == '==':
            condition_met = (attr == value)
        if condition_met:
            red_flags.append({'warning': rule['description'], 'source': rule.get('source', 'Не указан'), 'action': rule.get('action', '')})
            total_score += rule.get('weight', 0)
    return red_flags, total_score

def get_alternatives_for_drug(drug_name, pathogen):
    path_conf = antibiotic_alternatives_config.get('pathogens', {}).get(pathogen.lower(), {})
    alternatives = []
    if 'alternatives' in path_conf:
        alternatives.extend(path_conf['alternatives'])
    if 'severe_allergy' in path_conf:
        alternatives.extend(path_conf['severe_allergy'])
    if 'resistance' in path_conf:
        alternatives.extend(path_conf['resistance'])
    uniq = {}
    for alt in alternatives:
        uniq[alt['name']] = alt
    return list(uniq.values())

def predict_treatment_plan(patient_data):
    expected_features = num_cols + cat_cols
    input_dict = {feat: [patient_data.get(feat, np.nan)] for feat in expected_features}
    input_df = pd.DataFrame(input_dict)
    for col in input_df.columns:
        if input_df[col].isnull().any():
            if col in cat_cols:
                input_df[col].fillna('unknown', inplace=True)
            else:
                median_val = numeric_medians.get(col, 0)
                input_df[col].fillna(median_val, inplace=True)
    for col in cat_cols:
        le = le_dict.get(col)
        if le:
            input_df[col] = input_df[col].map(lambda x: x if x in le.classes_ else 'unknown')
            input_df[col] = le.transform(input_df[col])
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    pred = model.predict(input_df)[0]
    pred_pathogen_idx = pathogen_model.predict(input_df)[0]
    pred_pathogen = pathogen_encoder.inverse_transform([pred_pathogen_idx])[0]
    has_blood = patient_data.get('bloody_stool', 0) == 1
    has_mucus = patient_data.get('mucus_in_stool', 0) == 1
    diagnosis = get_diagnosis(pred_pathogen, has_blood, has_mucus)
    default_ab = {
        'salmonella': [('Цефтриаксон', '2.0 г/сут'), ('Азитромицин', '10 мг/кг/сут')],
        'shigella': [('Ципрофлоксацин', '0.5 г 2 раза/сут'), ('Цефтриаксон', '2.0 г/сут')],
        'e_coli': [('Цефтриаксон', '2.0 г/сут')],
        'campylobacter': [('Азитромицин', '500 мг 1 раз/сут 3 дня')],
        'viral': []
    }
    antibiotics = []
    if pred[3]:
        path_key = pred_pathogen.lower()
        if path_key in default_ab:
            antibiotics = default_ab[path_key]
        else:
            antibiotics = [('Цефтриаксон', '2.0 г/сут'), ('Азитромицин', '500 мг/сут')]
    alternatives_by_drug = {}
    for drug, dose in antibiotics:
        alts = get_alternatives_for_drug(drug, pred_pathogen)
        if alts:
            alternatives_by_drug[drug] = alts
    return {
        'diagnosis': diagnosis,
        'hospitalization': bool(pred[0]),
        'iv_fluids': bool(pred[1]),
        'oral_rehydration': bool(pred[2]),
        'antibiotics_needed': bool(pred[3]),
        'probiotics': bool(pred[4]),
        'antidiarrheals': bool(pred[5]),
        'suspected_pathogen': pred_pathogen,
        'recommended_antibiotics': antibiotics,
        'alternatives': alternatives_by_drug,
        'reasoning': f"Модель: госпитализация {'ДА' if pred[0] else 'НЕТ'}, антибиотики {'ДА' if pred[3] else 'НЕТ'}. Возбудитель: {pred_pathogen}."
    }

# RAG-агент
PDF_DIR = "./kr_data"
EXCEL_PATH = "./kr_data/Список утвержденных клинических рекомендаций.xlsx"
VECTOR_DB_PATH = "./chroma_kr_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OLLAMA_MODEL = "qwen2.5:7b-instruct"
OLLAMA_BASE_URL = "http://localhost:11434"
DISCLAIMER = "\nДанная информация носит справочный характер и не заменяет очную консультацию врача."

vectorstore = None
rag_chain = None

def load_vector_db():
    if not os.path.exists(VECTOR_DB_PATH):
        return None
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)

def build_vector_db():
    df = pd.read_excel(EXCEL_PATH)
    df = df[df["Статус применения"].astype(str).str.strip() == "Применяется"].copy()
    documents = []
    def safe_str(val):
        return str(val).strip() if pd.notna(val) else "Не указано"
    for _, row in df.iterrows():
        kr_id = safe_str(row["ID"])
        pdf_path = os.path.join(PDF_DIR, f"КР{kr_id}.pdf")
        if not os.path.exists(pdf_path):
            print(f"Пропущен {pdf_path}")
            continue
        loader = PyPDFLoader(pdf_path)
        raw_docs = loader.load()
        meta = {
            "id": kr_id,
            "title": safe_str(row["Наименование"]),
            "mkb10": safe_str(row["МКБ-10"]),
            "age_group": safe_str(row["Возрастная категория"]),
            "status": safe_str(row["Статус применения"]),
            "date": str(pd.to_datetime(row["Дата размещения"])).split(" ")[0] if pd.notna(row["Дата размещения"]) else "Не указана",
            "developer": safe_str(row["Разработчик"])
        }
        for doc in raw_docs:
            doc.metadata.update(meta)
            documents.append(doc)
    if not documents:
        raise ValueError("Нет загруженных PDF")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=40)
    chunks = splitter.split_documents(documents)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 30]
    vs = Chroma.from_documents(chunks, embeddings, persist_directory=VECTOR_DB_PATH)
    print(f"Векторная БД создана, чанков: {len(chunks)}")
    return vs

def init_rag():
    global vectorstore, rag_chain
    print("Инициализация RAG-агента...")
    vectorstore = load_vector_db()
    if vectorstore is None:
        print("Готовая БД не найдена, собираем заново (это займёт время)...")
        vectorstore = build_vector_db()
    else:
        print("Векторная БД загружена из кэша")
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    prompt = ChatPromptTemplate.from_template("""
Ты — медицинский информационный ассистент для пациентов.
Отвечай ТОЛЬКО на основе КОНТЕКСТА из утверждённых клинических рекомендаций.
ПРАВИЛА:
1. Не ставь диагнозов. Не назначай препараты, дозировки или схемы лечения.
2. Если информации нет, ответь: "В утверждённых рекомендациях по вашему вопросу данных не найдено."
3. Укажи источник: [Название] | [Дата] | [Возрастная группа]

КОНТЕКСТ:
{context}

ВОПРОС:
{question}
""").partial(disclaimer=DISCLAIMER)

    def format_docs(docs):
        return "\n\n---\n\n".join([f"📄 [{d.metadata['title']}]\n{d.page_content}" for d in docs])

    rag_chain = (
        {"context": lambda x: format_docs(x["docs"]), "question": lambda x: x["question"]}
        | prompt | llm | StrOutputParser()
    )
    print("RAG-агент готов к работе")

def ask_patient(query: str, age_group: str = "Взрослые") -> str:
    if vectorstore is None or rag_chain is None:
        return "RAG-агент не инициализирован. Обратитесь к администратору."
    age_filter = {"$in": [age_group, f"{age_group}, дети", f"дети, {age_group}"]}
    search_filter = {"$and": [{"status": {"$eq": "Применяется"}}, {"age_group": age_filter}]}
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5, "filter": search_filter})
    docs = retriever.invoke(query)
    if not docs:
        return f"В действующих КР для '{age_group}' информация не найдена.{DISCLAIMER}"
    return rag_chain.invoke({"docs": docs, "question": query})

# Инициализация RAG (если есть файлы)
try:
    init_rag()
except Exception as e:
    print(f"RAG не инициализирован: {e}")

# Поиск аналогов лекарств (EsklpAnalogFinder)
PARQUET_FILE = "esklp_merged2.parquet"
analog_finder = None
try:
    if os.path.exists(PARQUET_FILE):
        analog_finder = EsklpAnalogFinder(PARQUET_FILE)
        print("Поиск аналогов лекарств инициализирован")
    else:
        print(f"Файл {PARQUET_FILE} не найден. Поиск аналогов недоступен.")
except Exception as e:
    print(f"Ошибка загрузки EsklpAnalogFinder: {e}")

# Маршруты Flask
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        result = predict_treatment_plan(data)
        red_flags, score = check_red_flags(data)
        result['red_flags'] = red_flags
        result['red_flags_score'] = score
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/treatment_plan', methods=['POST'])
def get_treatment_plan():
    data = request.json
    patient_icd = data.get('icd_code', '')
    care_level = data.get('care_level', 'primary')
    patient_group = data.get('patient_group', 'adult')
    patient_data = data.get('patient_data', {})
    if not patient_icd:
        return jsonify({'error': 'Не указан код МКБ'}), 400
    plan = build_treatment_plan(patient_icd, care_level, patient_group, patient_data)
    return jsonify(plan)

@app.route('/api/rag', methods=['POST'])
def rag_query():
    data = request.json
    question = data.get('question', '').strip()
    age_group = data.get('age_group', 'Взрослые')
    if not question:
        return jsonify({'error': 'Введите вопрос'}), 400
    try:
        answer = ask_patient(question, age_group)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analogs', methods=['POST'])
def analogs():
    if analog_finder is None:
        return jsonify({'error': 'Система поиска аналогов не инициализирована'}), 503
    data = request.json
    query = data.get('query', '').strip()
    form = data.get('form', None)
    dosage = data.get('dosage', None)
    if not query:
        return jsonify({'error': 'Введите торговое наименование'}), 400

    # Получаем доступные формы и дозировки
    forms_dosages = analog_finder.get_forms_dosages(query) if hasattr(analog_finder, 'get_forms_dosages') else []
    # Если метод отсутствует, реализуем через прямой вызов
    if not forms_dosages and hasattr(analog_finder, '_get_unique_forms_and_dosages'):
        forms_dosages = analog_finder._get_unique_forms_and_dosages(query.lower().strip())
    if not forms_dosages:
        return jsonify({'error': f'Препарат "{query}" не найден в справочнике'}), 404

    # Если форма и дозировка не указаны, возвращаем список возможных вариантов
    if form is None or dosage is None:
        return jsonify({
            'status': 'need_selection',
            'forms_dosages': [{'form': f, 'dosage': d} for f, d in forms_dosages]
        })

    # Ищем аналоги с указанными формой и дозировкой
    analogs = analog_finder.find_analogs(query, form, dosage)
    if not analogs:
        return jsonify({'status': 'no_analogs', 'message': 'Аналоги не найдены'})

    # Форматируем результат для фронтенда
    result = []
    for a in analogs:
        result.append({
            'trade_name': a.get('trade_name', ''),
            'mnn': a.get('mnn', ''),
            'dosage_form': a.get('dosage_form', ''),
            'dosage': a.get('dosage', ''),
            'reg_number': a.get('reg_number', ''),
            'owner': a.get('owner', ''),
            'smnn_code': a.get('smnn_code', '')
        })
    return jsonify({'status': 'ok', 'analogs': result})

if __name__ == '__main__':
    app.run(debug=True, port=5000)