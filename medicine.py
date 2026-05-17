import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class EsklpAnalogFinder:
    def __init__(self, parquet_path: str):
        self.df = pd.read_parquet(parquet_path)
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].astype(str)
        self.df.fillna('', inplace=True)
        self._map_columns()
        self._build_smnn_index()
        self._build_trade_index()

    def _map_columns(self):
        col_lower = {col: col.lower() for col in self.df.columns}
        patterns = {
            'smnn_code': ['смнн', 'smnn', 'узел'],
            'trade_name': ['торговое', 'торг', 'trade', 'наименование'],
            'mnn_std': ['мнн', 'mnn', 'стандартизованное'],
            'form_std': ['лекарственная форма', 'форма', 'form'],
            'dosage_std': ['дозировка', 'dosage', 'доза'],
            'reg_number': ['номер', 'регистрационное', 'reg_number', 'ру'],
            'owner': ['владелец', 'owner']
        }
        self.col = {}
        for key, keywords in patterns.items():
            found = None
            for col, low in col_lower.items():
                if any(kw in low for kw in keywords):
                    found = col
                    break
            self.col[key] = found
        if self.col['smnn_code'] is None:
            raise KeyError("Не найдена колонка, содержащая 'СМНН' или 'узел'.")
        if self.col['trade_name'] is None:
            raise KeyError("Не найдена колонка, содержащая 'торговое' или 'наименование'.")

    def _build_smnn_index(self):
        smnn_col = self.col['smnn_code']
        self.smnn_index = {}
        for idx, row in self.df.iterrows():
            key = row[smnn_col]
            if not key or key == 'nan':
                continue
            if key not in self.smnn_index:
                self.smnn_index[key] = []
            self.smnn_index[key].append(row)

    def _build_trade_index(self):
        trade_col = self.col['trade_name']
        self.trade_index = {}  # ключ: нормализованное торговое наименование -> список строк
        for idx, row in self.df.iterrows():
            trade = row[trade_col]
            if not trade or trade == 'nan':
                continue
            key = trade.strip().lower()
            if key not in self.trade_index:
                self.trade_index[key] = []
            self.trade_index[key].append(row)

    def _normalize_form(self, form: str) -> str:
        """Приводит форму к каноническому виду (удаляет лишние пробелы, приводит к нижнему регистру)."""
        return form.strip().lower()

    def _normalize_dosage(self, dosage: str) -> str:
        """Приводит дозировку к каноническому виду."""
        return dosage.strip().lower()

    def _get_unique_forms_and_dosages(self, trade_name: str) -> List[Tuple[str, str]]:
        """Возвращает список уникальных пар (лекарственная форма, дозировка) для данного торгового наименования."""
        if trade_name not in self.trade_index:
            return []
        pairs = set()
        for row in self.trade_index[trade_name]:
            form = row.get(self.col.get('form_std', ''), '')
            dosage = row.get(self.col.get('dosage_std', ''), '')
            if form and dosage:
                pairs.add((self._normalize_form(form), self._normalize_dosage(dosage)))
        return list(pairs)

    def get_forms_dosages(self, query: str) -> List[Tuple[str, str]]:
        """Публичный метод для получения доступных форм и дозировок препарата."""
        query_clean = query.strip().lower()
        if query_clean not in self.trade_index:
            return []
        return self._get_unique_forms_and_dosages(query_clean)

    def find_analogs(self, query: str, target_form: Optional[str] = None, target_dosage: Optional[str] = None) -> List[Dict]:
        """
        Ищет аналоги.
        Если target_form и target_dosage не указаны, но у торгового наименования есть несколько форм,
        возвращает пустой список (вызывающий код должен сначала запросить уточнение через get_forms_dosages).
        """
        query_clean = query.strip().lower()
        if query_clean not in self.trade_index:
            return []

        # Определяем возможные формы и дозировки для запроса
        forms_dosages = self._get_unique_forms_and_dosages(query_clean)
        if not forms_dosages:
            return []

        if target_form is None or target_dosage is None:
            # Если форм несколько, возвращаем пустой список (клиент должен уточнить)
            return []

        # Ищем все записи с таким же торговым наименованием и нужной формой/дозировкой,
        # чтобы получить узлы СМНН
        smnn_codes = set()
        for row in self.trade_index[query_clean]:
            form = self._normalize_form(row.get(self.col.get('form_std', ''), ''))
            dosage = self._normalize_dosage(row.get(self.col.get('dosage_std', ''), ''))
            if form == target_form and dosage == target_dosage:
                smnn = row[self.col['smnn_code']]
                if smnn and smnn != 'nan':
                    smnn_codes.add(smnn)

        if not smnn_codes:
            return []

        # Собираем все строки по узлам СМНН
        all_rows = []
        for code in smnn_codes:
            all_rows.extend(self.smnn_index.get(code, []))

        # Фильтруем только те строки, у которых форма и дозировка совпадают с целевыми
        filtered_rows = []
        for row in all_rows:
            form = self._normalize_form(row.get(self.col.get('form_std', ''), ''))
            dosage = self._normalize_dosage(row.get(self.col.get('dosage_std', ''), ''))
            if form == target_form and dosage == target_dosage:
                filtered_rows.append(row)

        # Дедуплицируем по торговому наименованию и номеру РУ (или по всем полям)
        unique = {}
        for row in filtered_rows:
            trade = row.get(self.col['trade_name'], '')
            reg = row.get(self.col.get('reg_number', ''), '')
            key = (trade, reg)
            if key not in unique:
                unique[key] = row

        analogs = [self._row_to_dict(row) for row in unique.values()]
        return analogs

    def _row_to_dict(self, row: pd.Series) -> Dict:
        def get(col_key):
            col_name = self.col.get(col_key)
            if col_name and col_name in row:
                val = row[col_name]
                return val if val != 'nan' else ''
            return ''

        mnn = get('mnn_std')
        if not mnn:
            for col in self.df.columns:
                if 'мнн' in col.lower() or 'mnn' in col.lower():
                    mnn = row[col] if row[col] != 'nan' else ''
                    break
        return {
            'trade_name': get('trade_name'),
            'mnn': mnn,
            'dosage_form': get('form_std'),
            'dosage': get('dosage_std'),
            'reg_number': get('reg_number'),
            'owner': get('owner'),
            'smnn_code': get('smnn_code')
        }

    def format_analog_card(self, analog: Dict) -> str:
        return (
            f"Торговое наименование: {analog['trade_name']}\n"
            f"МНН: {analog['mnn']}\n"
            f"Форма: {analog['dosage_form']} | Дозировка: {analog['dosage']}\n"
            f"РУ: {analog['reg_number']}\n"
            f"Владелец: {analog['owner']}\n"
            f"Узел СМНН: {analog['smnn_code']}\n"
            f"Основание: одинаковый узел СМНН"
        )

    def print_analogs(self, query: str):
        # Сначала проверим, есть ли несколько форм
        forms = self.get_forms_dosages(query)
        if len(forms) > 1:
            print(f"Уточните, пожалуйста, форму и дозировку для '{query}':")
            for i, (f, d) in enumerate(forms, 1):
                print(f"  {i}. {f}, {d}")
            return

        analogs = self.find_analogs(query, forms[0][0] if forms else None, forms[0][1] if forms else None)
        if not analogs:
            print(f"\nПрепарат '{query}' не найден в справочнике или нет аналогов.")
            return
        print(f"\nЗапрос: '{query}'")
        print(f"Найдено {len(analogs)} уникальных аналогов:\n")
        for idx, a in enumerate(analogs, 1):
            print(f"{idx}. {self.format_analog_card(a)}")

if __name__ == "__main__":
    PARQUET_FILE = "esklp_merged2.parquet"
    finder = EsklpAnalogFinder(PARQUET_FILE)
    test_queries = ["Нимесулид", "Нимулид", "Ибупрофен"]
    for q in test_queries:
        finder.print_analogs(q)