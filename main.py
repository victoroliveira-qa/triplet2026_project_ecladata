import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import textwrap  # Necessário para formatar o texto visualmente
import re

# Configuração estética dos gráficos
sns.set_theme(style="whitegrid")


class CorpusAnalyzer:
    def __init__(self, file_path, data_dir='data', csv_dir='csvs', plots_dir='graficos'):
        """Carrega o dataset e prepara as estruturas de dados iniciais."""
        # Criação dos diretórios de saída se não existirem
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        self.csv_dir = csv_dir
        self.plots_dir = plots_dir

        full_path = os.path.join(data_dir, file_path)  # Caminho completo para o arquivo de entrada
        print(f"Carregando arquivo: {file_path}...")
        with open(full_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.documents = self.data.get('documents', [])

        self.df_docs = None
        self.df_texts = None
        self.df_tables = None
        self.df_annotations = None

        self._process_data()

    # =====================================================
    # LOG (TELA + TXT)
    # =====================================================
    def _log(self, message, file=None):
        print(message)
        if file:
            file.write(message + "\n")

    def _process_data(self):
        """
        Transforma o JSON hierárquico em DataFrames planos para análise.
        Separa logicamente Textos, Tabelas e Anotações.
        """
        doc_rows = []
        text_rows = []
        table_rows = []
        annotation_rows = []

        for doc in self.documents:
            doc_id = doc.get('id')

            # --- Metadados de Extração (Textos e Tabelas) ---
            has_text = False
            has_table = False
            doc_text_len = 0
            doc_table_count = 0
            doc_cols_list = []

            if 'raw' in doc and '_source' in doc['raw']:
                meta_list = doc['raw']['_source'].get('extractionMetadata', [])
                for meta in meta_list:
                    # 1. Processar Textos
                    texts = meta.get('texts', [])
                    if texts:
                        has_text = True
                        for t in texts:
                            val = t.get('value', '')
                            length = len(val)
                            doc_text_len += length
                            text_rows.append({
                                'doc_id': doc_id,
                                'content': val,
                                'length': length,
                                'words': len(val.split())
                            })

                    # 2. Processar Tabelas
                    tables = meta.get('tables', [])
                    if tables:
                        has_table = True
                        doc_table_count += len(tables)
                        for tbl in tables:
                            t_data = tbl.get('tableData', [])
                            if t_data:
                                rows = len(t_data)
                                cols = len(t_data[0])  # Assume 1ª linha como largura
                                # Tenta pegar cabeçalhos da 1ª linha
                                headers = [str(h).strip() for h in t_data[0]]

                                doc_cols_list.append(cols)
                                table_rows.append({
                                    'doc_id': doc_id,
                                    'rows': rows,
                                    'cols': cols,
                                    'headers': headers
                                })

            # --- Anotações Semânticas ---
            annotations = doc.get('annotations', [])
            num_anns = len(annotations)

            def loc_code(x):
                x = (x or "").lower()
                if "table" in x:
                    return "TAB"
                if "text" in x:
                    return "T"
                return "UNK"

            for ann in annotations:
                source = 'unknown'
                if 'subject' in ann and 'annotationType' in ann['subject']:
                    source = ann['subject']['annotationType']

                subj = ann.get('subject', {})
                obj = ann.get('object', {})
                pred = ann.get('predicate', {})

                subj_loc = loc_code(subj.get("annotationType"))
                obj_loc = loc_code(obj.get("annotationType"))
                arg_origin = f"({subj_loc}, {obj_loc})"

                # Coleta os IDs que antes não estavam sendo pegos
                subj_id = subj.get('id', None)
                obj_id = obj.get('id', None)

                # Coleta dados detalhados do Predicado (Relação)
                pred_id = pred.get('id', None)
                pred_type = pred.get('entityDatatype', 'string')

                annotation_rows.append({
                    'doc_id': doc_id,

                    # --- DADOS DO PREDICADO (ATUALIZADO) ---
                    'relation': pred.get('entityLabel', 'Unknown'),
                    'relation_type': pred_type,  # Novo
                    'relation_id': pred_id,  # Novo

                    'subject': subj.get('entityLabel', 'Unknown'),
                    'subject_type': subj.get('entityDatatype', 'string'),
                    'subject_custom': subj.get('entityCustom', False),
                    'subject_id': subj.get('id', None),

                    'object': obj.get('entityLabel', 'Unknown'),
                    'object_type': obj.get('entityDatatype', 'string'),
                    'object_custom': obj.get('entityCustom', False),
                    'object_id': obj.get('id', None),

                    'source': source
                })

            # Resumo do Documento
            doc_rows.append({
                'id': doc_id,
                'has_text': has_text,
                'has_table': has_table,
                'text_length': doc_text_len,
                'num_tables': doc_table_count,
                'num_cols': doc_cols_list,  # Lista de colunas por tabela neste doc
                'num_annotations': num_anns
            })

        # Criação dos DataFrames
        self.df_docs = pd.DataFrame(doc_rows)
        self.df_texts = pd.DataFrame(text_rows)
        self.df_tables = pd.DataFrame(table_rows)
        self.df_annotations = pd.DataFrame(annotation_rows)

    def _calculate_document_origin(self, doc, texts, tables):
        """
        Calcula a origem das entidades (T, TAB) e a distância textual quando (T, T).
        Retorna um dicionário com contagens e distâncias.
        """
        origins = {}
        annotations = doc.get('annotations', [])

        # Divide textos em sentenças simples
        sentences = []
        for t in texts:
            sentences.extend([s.strip() for s in t.split('.') if s.strip()])

        for ann in annotations:
            subj = ann.get('subject', {}).get('entityLabel', '')
            obj = ann.get('object', {}).get('entityLabel', '')

            subj_in_text = any(subj in s for s in sentences)
            obj_in_text = any(obj in s for s in sentences)

            subj_in_table = any(subj in row for row in tables)
            obj_in_table = any(obj in row for row in tables)

            # Define origem
            def origin(x_text, x_tab):
                if x_tab:
                    return "TAB"
                if x_text:
                    return "T"
                return "UNK"

            o1 = origin(subj_in_text, subj_in_table)
            o2 = origin(obj_in_text, obj_in_table)

            key = (o1, o2)

            if key not in origins:
                origins[key] = {"count": 0, "distances": []}

            origins[key]["count"] += 1

            # Distância apenas para (T, T)
            if key == ("T", "T"):
                subj_pos = [i for i, s in enumerate(sentences) if subj in s]
                obj_pos = [i for i, s in enumerate(sentences) if obj in s]

                for sp in subj_pos:
                    for op in obj_pos:
                        origins[key]["distances"].append(abs(sp - op))

        return origins

    def _detect_entity_origin(self, label, texts, table_rows):
        in_text_positions = []
        for i, sent in enumerate(texts):
            if label and label in sent:
                in_text_positions.append(i)

        in_table = any(label in row for row in table_rows)

        if in_table:
            return "TAB", None
        if in_text_positions:
            return "T", in_text_positions
        return "UNK", None

    def _split_sentences(self, text: str):
        """
        Divide em sentenças usando ., !, ? e quebras de linha como delimitadores.
        Retorna lista de sentenças (sem vazios).
        """
        if not text:
            return []
        parts = re.split(r'(?<=[\.\!\?])\s+|\n+', text.strip())
        return [p.strip() for p in parts if p.strip()]

    def _count_in_table(self, label: str, table_data_blocks):
        """
        Conta ocorrências de 'label' em tableData (varrendo célula por célula).
        Retorna contagem total.
        """
        if not label:
            return 0
        needle = label.lower().strip()
        total = 0
        for tableData in table_data_blocks:
            for row in tableData:
                for cell in row:
                    s = str(cell).lower()
                    total += s.count(needle)
        return total

    # =========================================================================
    # TEMA 1: ESTRUTURAIS DO CORPUS
    # =========================================================================
    def analyze_structure(self, output_text=None):
        log_file = open(output_text, "a", encoding="utf-8") if output_text else None

        self._log("\n----------------------- 1. Análise Estrutural do Corpus --------------------------", log_file)

        total_docs = len(self.df_docs)

        text_only = self.df_docs[self.df_docs['has_text'] & ~self.df_docs['has_table']].shape[0]
        table_only = self.df_docs[~self.df_docs['has_text'] & self.df_docs['has_table']].shape[0]
        both = self.df_docs[self.df_docs['has_text'] & self.df_docs['has_table']].shape[0]

        docs_sem_ann = self.df_docs[self.df_docs['num_annotations'] == 0].shape[0]
        pct_sem_ann = (docs_sem_ann / total_docs) * 100
        avg_ann = self.df_docs['num_annotations'].mean()

        self._log(f"Quantos documentos existem no corpus?: {total_docs}", log_file)
        self._log(f"Quantos documentos possuem: apenas texto?: {text_only}", log_file)
        self._log(f"Quantos documentos possuem: apenas tabelas?: {table_only}", log_file)
        self._log(f"Quantos documentos possuem: texto e tabelas?: {both}", log_file)
        self._log(
            f"Quantos documentos possuem anotações semânticas (annotations)?: {total_docs - docs_sem_ann}",
            log_file
        )
        self._log(f"Qual a média de anotações por documento?: {avg_ann:.2f}", log_file)
        self._log(f"Qual o percentual de documentos sem nenhuma anotação?: {pct_sem_ann:.2f}%", log_file)

        if log_file:
            log_file.close()

    # =========================================================================
    # TEMA 2: SOBRE OS TEXTOS
    # =========================================================================
    def analyze_texts(self, output_text=None):
        def _log(message):
            print(message)
            if log_file:
                log_file.write(message + "\n")

        log_file = open(output_text, "a", encoding="utf-8") if output_text else None

        _log("\n----------------------- 2. Análise dos Textos --------------------------")

        if self.df_texts.empty:
            _log("Nenhum texto encontrado.")
            if log_file:
                log_file.close()
            return

        total_texts = len(self.df_texts)
        total_chars = self.df_texts['length'].sum()
        avg_chars = self.df_texts['length'].mean()
        avg_words = self.df_texts['words'].mean()
        max_len = self.df_texts['length'].max()
        min_len = self.df_texts['length'].min()

        text_per_doc = (
            self.df_texts
            .groupby('doc_id')['length']
            .sum()
            .reset_index()
        )

        merged = (
            pd.merge(
                self.df_docs,
                text_per_doc,
                left_on='id',
                right_on='doc_id',
                how='left'
            )
            .fillna(0)
        )

        corr = merged['length'].corr(merged['num_annotations'])

        _log(f"Quantos textos (texts) existem no total?: {total_texts}")
        _log(f"Qual o número total de caracteres nos textos?: {total_chars}")
        _log(f"Qual o tamanho médio dos textos (caracteres): {avg_chars:.2f}")
        _log(f"Qual o tamanho médio dos textos (palavras): {avg_words:.2f}")
        _log(f"Maior texto: {max_len} chars | Menor texto: {min_len} chars")
        _log(
            "Existe correlação entre: tamanho do texto e quantidade de entidades anotadas?: "
            f"{corr:.4f}"
        )

        if log_file:
            log_file.close()

    # =========================================================================
    # TEMA 3: SOBRE AS TABELAS
    # =========================================================================
    def analyze_tables(self, output_text=None):
        def _log(message):
            print(message)
            if log_file:
                log_file.write(message + "\n")

        log_file = open(output_text, "a", encoding="utf-8") if output_text else None

        _log("\n-------------------------- 3. Análise das Tabelas --------------------------")

        if self.df_tables.empty:
            _log("Nenhuma tabela encontrada.")
            if log_file:
                log_file.close()
            return

        total_tables = len(self.df_tables)
        avg_tables_doc = self.df_docs['num_tables'].mean()
        avg_rows = self.df_tables['rows'].mean()
        avg_cols = self.df_tables['cols'].mean()

        _log(f"Total de Tabelas: {total_tables}")
        _log(f"Média de tabelas por doc: {avg_tables_doc:.2f}")
        _log(f"Média de Linhas por tabela: {avg_rows:.2f}")
        _log(f"Média de Colunas por tabela: {avg_cols:.2f}")

        tables_header_01 = 0
        for h_list in self.df_tables['headers']:
            headers_norm = [str(h).strip() for h in h_list]
            if headers_norm == ["0", "1"]:
                tables_header_01 += 1

        _log(f"Quantidade de tabelas que começam com o header ['0','1']: {tables_header_01}")

        # =====================================================
        # HEADERS
        # =====================================================
        all_headers = []
        for h_list in self.df_tables['headers']:
            all_headers.extend(h_list)

        header_counts = Counter(all_headers)

        _log("\nTop 10 Nomes de Colunas mais frequentes:")
        for k, v in header_counts.most_common(10):
            _log(f"  - {k}: {v}")

        # =====================================================
        # CLASSIFICAÇÃO SEMÂNTICA SIMPLES
        # =====================================================
        def classify_header(h):
            h = str(h).lower()
            if any(x in h for x in ['name', 'title', 'company', 'ceo', 'director', 'artist']):
                return 'Entidade Nomeada (name, title, company, ceo, director, artist)'
            if any(x in h for x in ['date', 'year', 'revenue', 'budget', 'sales', 'score', 'count']):
                return 'Atributo Descritivo (date, year, revenue, budget, sales, score, count)'
            if any(x in h for x in ['by', 'of', 'with']):
                return 'Relação Implícita (by, of, with)'
            return 'Outros/Genérico'

        classified = [classify_header(h) for h in all_headers]
        type_counts = Counter(classified)

        _log("\nDistribuição Estimada de Tipos de Colunas:")
        total = len(classified)

        for k, v in type_counts.items():
            pct = (v / total * 100) if total > 0 else 0
            _log(f"  - {k}: {v} ({pct:.1f}%)")

        if log_file:
            log_file.close()

    # =========================================================================
    # TEMA 4: SOBRE ENTIDADES E RELAÇÕES
    # =========================================================================
    def analyze_er(self, output_text=None):
        def _log(message):
            print(message)
            if log_file:
                log_file.write(message + "\n")

        log_file = open(output_text, "a", encoding="utf-8") if output_text else None

        _log("\n-------------------------- 4. Análise de Entidades e Relações --------------------------")

        if self.df_annotations.empty:
            _log("Nenhuma anotação encontrada.")
            if log_file:
                log_file.close()
            return

        # =====================================================
        # ENTIDADES
        # =====================================================
        subjects = self.df_annotations['subject']
        objects = self.df_annotations['object']
        all_entities = pd.concat([subjects, objects])

        unique_ents = all_entities.nunique()
        _log(f"Entidades Únicas no Corpus: {unique_ents}")

        custom_count = self.df_annotations[self.df_annotations['subject_custom'] == True].shape[0]
        wiki_count = self.df_annotations[self.df_annotations['subject_custom'] == False].shape[0]

        _log(f"Ocorrências como Sujeito -> Wikidata: {wiki_count} | Custom: {custom_count}")

        _log("\nTop 10 Entidades mais frequentes:")
        _log(all_entities.value_counts().head(10).to_string())

        _log("\nLista Completa de Entidades e Frequências:")
        _log(all_entities.value_counts().head(2).to_string())

        _log("\n--- ORIGEM DOS ARGUMENTOS ---")
        if 'argument_origin' in self.df_annotations.columns:
            _log("Distribuição dos tipos de ORIGEM DOS ARGUMENTOS:")
            _log(self.df_annotations['argument_origin'].value_counts().to_string())
        else:
            _log("Coluna 'argument_origin' não encontrada. Verifique se o _process_data foi atualizado.")

        # =====================================================
        # RELAÇÕES
        # =====================================================
        _log("\n--- Relações ---")

        total_rels = len(self.df_annotations)
        distinct_rels = self.df_annotations['relation'].nunique()

        _log(f"Total de Relações: {total_rels}")
        _log(f"\nTipos de Relações Distintas: {distinct_rels}")

        _log("\nDistribuição dos Tipos de Relação:")
        _log(self.df_annotations['relation'].value_counts().head(10).to_string())

        _log("\nOrigem das Relações (Extraction Source):")
        _log(self.df_annotations['source'].value_counts().to_string())

        # Número médio de relações por documento (considerando docs com pelo menos 1 relação)
        rels_per_doc = self.df_annotations.groupby('doc_id').size()
        avg_rels = rels_per_doc.mean()

        _log(f"\nNúmero médio de relações por documento (docs com relações): {avg_rels:.2f}")

        # Relações reflexivas: subject == object
        reflexive = self.df_annotations[self.df_annotations['subject'] == self.df_annotations['object']].shape[0]
        _log(f"\nQuantidade de Relações Reflexivas (subject = object): {reflexive}")

        if log_file:
            log_file.close()

    # =========================================================================
    # TEMA 5: PERGUNTAS CRUZADAS
    # =========================================================================
    def analyze_cross(self, output_text=None):
        def _log(message):
            print(message)
            if log_file:
                log_file.write(message + "\n")

        log_file = open(output_text, "a", encoding="utf-8") if output_text else None

        _log("\n-------------------------- 5. Análises Cruzadas --------------------------")

        # =====================================================
        # DOCS COM TABELA x ENTIDADES
        # =====================================================
        docs_with_table = self.df_docs[self.df_docs['has_table']]
        avg_ent_table = docs_with_table['num_annotations'].mean()

        _log(f"Média de entidades em docs com tabela: {avg_ent_table:.2f}")

        # =====================================================
        # Nº DE COLUNAS x Nº DE ENTIDADES
        # =====================================================
        exploded = self.df_docs.explode('num_cols')
        exploded = exploded.dropna(subset=['num_cols'])

        if not exploded.empty:
            exploded['num_cols'] = exploded['num_cols'].astype(int)
            corr_cols_ent = exploded['num_cols'].corr(exploded['num_annotations'])
            _log(f"Correlação (Nº de Colunas vs Nº de Entidades): {corr_cols_ent:.4f}")
        else:
            _log("Não foi possível calcular correlação entre colunas e entidades.")

        # =====================================================
        # DENSIDADE RELACIONAL
        # =====================================================
        total_rels = len(self.df_annotations)
        total_chars = self.df_texts['length'].sum()

        density = (total_rels / total_chars) * 1000 if total_chars > 0 else 0
        _log(f"Densidade Relacional: {density:.2f} relações a cada 1.000 caracteres.")

        if log_file:
            log_file.close()

    # =========================================================================
    # TEMA 6: VISUALIZAÇÃO AMIGÁVEL
    # =========================================================================
    def visualize_samples(self, limit=2, output_text=None):
        import textwrap
        import re
        from collections import Counter

        log_file = open(output_text, "a", encoding="utf-8") if output_text else None

        def log(msg):
            print(msg)
            if log_file:
                log_file.write(msg + "\n")

        COL_WIDTH_TEXT = 30
        COL_WIDTH_TABLE = 150
        COL_WIDTH_ANN = 50

        def split_lines(text, width):
            return textwrap.wrap(str(text), width=width) or [""]

        def ann_type_to_code(t):
            t = (t or "").lower()
            if "table" in t:
                return "TAB"
            if "text" in t:
                return "TXT"
            return "UNK"

        def split_sentences_with_spans(full_text: str):
            """
            Divide full_text em sentenças e retorna lista de spans (start,end) e sentenças.
            Usa pontuação final (.!?), com fallback por quebras de linha.
            """
            if not full_text:
                return [], []

            spans = []
            sents = []

            # Captura sentenças terminadas em . ! ? ou última parte
            pattern = re.compile(r".+?(?:[\.!\?]+(?=\s)|$)", re.DOTALL)
            idx = 0
            for m in pattern.finditer(full_text):
                sent = m.group(0).strip()
                if not sent:
                    continue
                start = m.start()
                end = m.end()
                spans.append((start, end))
                sents.append(sent)
                idx = end

            # fallback simples se nada encontrado
            if not sents:
                parts = [p.strip() for p in re.split(r"\n+", full_text) if p.strip()]
                cur = 0
                for p in parts:
                    start = full_text.find(p, cur)
                    end = start + len(p)
                    spans.append((start, end))
                    sents.append(p)
                    cur = end

            return spans, sents

        def sentence_index_for_offset(spans, offset):
            """
            Retorna índice da sentença (1-based) onde offset cai.
            Se não encontrar, retorna None.
            """
            if offset is None:
                return None
            try:
                off = int(offset)
            except Exception:
                return None

            for i, (s, e) in enumerate(spans):
                if s <= off < e:
                    return i + 1  # 1-based
            return None

        def count_role_label_by_annotation(annotations, role, label):
            """
            Soma por annotation:
            quantas annotations possuem role.entityLabel==label e role.annotationType==text/table.
            """
            txt = 0
            tab = 0
            label = (label or "").strip()
            for a in annotations:
                block = a.get(role, {}) or {}
                if (block.get("entityLabel") or "").strip() == label:
                    code = ann_type_to_code(block.get("annotationType"))
                    if code == "TXT":
                        txt += 1
                    elif code == "TAB":
                        tab += 1
            return txt, tab

        log("\n--- 6. Visualização Amigável ---")

        count = 0
        for doc in self.documents:
            if count >= limit:
                break

            doc_id = doc.get("id", "N/A")
            metas = doc.get("raw", {}).get("_source", {}).get("extractionMetadata", [])

            # =====================================================
            # TEXT (concat)
            # =====================================================
            texts = []
            for m in metas:
                for t in m.get("texts", []):
                    if t.get("value"):
                        texts.append(t["value"])

            text_block = " ".join(texts) if texts else "Nenhum texto encontrado."
            text_lines = split_lines(text_block, COL_WIDTH_TEXT)

            # sentenças + spans para distância
            full_text = " ".join(texts) if texts else ""
            sent_spans, sentences = split_sentences_with_spans(full_text)

            # =====================================================
            # TABLE (layout horizontal)
            # =====================================================
            table_lines = []
            for m in metas:
                for tbl in m.get("tables", []):
                    data = tbl.get("tableData", [])
                    if len(data) < 2:
                        continue

                    headers = data[0]
                    header_line = " | ".join(map(str, headers))
                    table_lines.append(header_line)
                    table_lines.append("-" * len(header_line))

                    for row in data[1:]:
                        row_line = " | ".join(str(v) for v in row)
                        table_lines.append(row_line)

            if not table_lines:
                table_lines = ["Nenhuma tabela encontrada."]

            table_lines = sum([split_lines(l, COL_WIDTH_TABLE) for l in table_lines], [])

            # =====================================================
            # ANNOTATIONS (entityLabel + origem pelo annotationType)
            # =====================================================
            ann_lines = []
            annotations = doc.get('annotations', [])

            def type_code(t):
                t = (t or "").lower()
                if "text" in t:
                    return "T"
                if "table" in t:
                    return "TAB"
                return "UNK"

            def format_subject_or_object(block):
                label = block.get("entityLabel", "N/A")
                tcode = type_code(block.get("annotationType"))

                # Se for TEXT: usar (start,end)
                if tcode == "T":
                    start = block.get("annotationStart", "?")
                    end = block.get("annotationEnd", "?")
                    return f"{label} - {tcode}({start},{end})"

                # Se for TABLE: usar (row,cell)
                if tcode == "TAB":
                    row = block.get("annotationRow", "?")
                    cell = block.get("annotationCell", "?")
                    return f"{label} - {tcode}({row},{cell})"

                return f"{label} - {tcode}"

            def format_predicate(pred_block, subj_block, obj_block):
                pred_label = pred_block.get("entityLabel", "N/A")

                subj_code = type_code(subj_block.get("annotationType"))
                obj_code = type_code(obj_block.get("annotationType"))

                return f"{pred_label} - ({subj_code}, {obj_code})"

            if not annotations:
                ann_lines.append("Nenhuma anotação encontrada.")
            else:
                for idx, ann in enumerate(annotations, start=1):
                    subj_block = ann.get("subject", {}) or {}
                    pred_block = ann.get("predicate", {}) or {}
                    obj_block = ann.get("object", {}) or {}

                    ann_lines.append(f"ANNOTATION {idx}")
                    ann_lines.append(f"SUBJECT: {format_subject_or_object(subj_block)}")
                    ann_lines.append(f"PREDICATE: {format_predicate(pred_block, subj_block, obj_block)}")
                    ann_lines.append(f"OBJECT: {format_subject_or_object(obj_block)}")
                    ann_lines.append("-" * 30)

            ann_lines = sum(
                [split_lines(line, COL_WIDTH_ANN) for line in ann_lines],
                []
            )

            # =====================================================
            # GRID PRINCIPAL
            # =====================================================
            log(f"[INFO] Processando Document ID: {doc_id}")
            log("=" * (COL_WIDTH_TEXT + COL_WIDTH_TABLE + COL_WIDTH_ANN + 6))

            log(
                f"{'TEXT'.ljust(COL_WIDTH_TEXT)} | "
                f"{'TABLE'.ljust(COL_WIDTH_TABLE)} | "
                f"{'ANNOTATIONS'.ljust(COL_WIDTH_ANN)}"
            )
            log(
                "-" * COL_WIDTH_TEXT + "+-" +
                "-" * COL_WIDTH_TABLE + "+-" +
                "-" * COL_WIDTH_ANN
            )

            max_lines = max(len(text_lines), len(table_lines), len(ann_lines))
            for i in range(max_lines):
                log(
                    f"{(text_lines[i] if i < len(text_lines) else '').ljust(COL_WIDTH_TEXT)} | "
                    f"{(table_lines[i] if i < len(table_lines) else '').ljust(COL_WIDTH_TABLE)} | "
                    f"{(ann_lines[i] if i < len(ann_lines) else '').ljust(COL_WIDTH_ANN)}"
                )

            log("-" * (COL_WIDTH_TEXT + COL_WIDTH_TABLE + COL_WIDTH_ANN + 6))
            count += 1

        if log_file:
            log_file.close()

    # =========================================================================
    # TEMA 7: GERAÇÃO DOS GRÁFICOS
    # =========================================================================
    def generate_plots(self):
        print("\n--- 7. Gerando Gráficos... ---")

        # 1. Distribuição de Entidades por Documento
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df_docs['num_annotations'], bins=20, kde=True, color='skyblue')
        plt.title('Distribution of Entities by Document')
        plt.xlabel('Number of Entities')
        plt.ylabel('Document Count')
        plt.savefig(os.path.join(self.plots_dir, 'grafico_1_dist_entidades_doc.png'))
        plt.close()

        # 2. Distribuição de Relações por Tipo
        if not self.df_annotations.empty:
            plt.figure(figsize=(10, 6))
            top_rels = self.df_annotations['relation'].value_counts().head(10)
            sns.barplot(x=top_rels.values, y=top_rels.index, palette='viridis')
            plt.title('Distribution of Relationships by Type')
            plt.xlabel('Frequency')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'grafico_2_dist_relacoes.png'))
            plt.close()

        # 3. Número de Colunas por Tabela
        if not self.df_tables.empty:
            plt.figure(figsize=(10, 6))
            sns.countplot(data=self.df_tables, x='cols', palette='magma')
            plt.title('Distribution of Number of Columns by Table')
            plt.xlabel('Number of Columns')
            plt.ylabel('Table Count')
            plt.savefig(os.path.join(self.plots_dir, 'grafico_3_colunas_por_tabela.png'))
            plt.close()

        # 4. Textos: Caracteres vs Entidades (Scatterplot)
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=self.df_docs, x='text_length', y='num_annotations', alpha=0.6)
            plt.title('Correlation: Text Size vs. Number of Entities')
            plt.xlabel('Total Characters (Text)')
            plt.ylabel('Total Annotated Entities')
            plt.savefig(os.path.join(self.plots_dir, 'grafico_4_scatter_chars_ents.png'))
            plt.close()

        # 5. Proporção Texto vs Tabela vs Anotações (Barra Empilhada)
            counts = {
                'Possui Texto': self.df_docs['has_text'].sum(),
                'Possui Tabela': self.df_docs['has_table'].sum(),
                'Possui Anotações': (self.df_docs['num_annotations'] > 0).sum()
            }
            plt.figure(figsize=(8, 6))
            sns.barplot(x=list(counts.keys()), y=list(counts.values()), palette='muted')
            plt.title('Presence of Characteristics in Documents')
            plt.ylabel('Number of Documents')
            plt.savefig(os.path.join(self.plots_dir, 'grafico_5_features_doc.png'))
            plt.close()

        # 6. Número de Linhas por Tabela
        if not self.df_tables.empty:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=self.df_tables, x='rows', discrete=True, color='coral')
            plt.title('Distribution of the Number of Rows per Table')
            plt.xlabel('Number of Rows')
            plt.ylabel('Table Count')
            plt.savefig(os.path.join(self.plots_dir, 'grafico_6_linhas_por_tabela.png'))
            plt.close()

        # 7. NOVO: Totais Gerais - Entidades vs Relações
        if not self.df_annotations.empty:
            # Cálculos
            total_rels = len(self.df_annotations)
            # Une sujeitos e objetos para contar quantos únicos existem
            unique_ents = pd.concat([self.df_annotations['subject'], self.df_annotations['object']]).nunique()

            data_totals = {'Total Relações': total_rels, 'Entidades Únicas': unique_ents}

            # Plotagem
            plt.figure(figsize=(8, 6))
            ax = sns.barplot(x=list(data_totals.keys()), y=list(data_totals.values()), palette='Pastel1')
            plt.title('Absolute Totals of the Corpus: Relations vs. Entities')
            plt.ylabel('Total Quantity')

            # Adicionar o número exato em cima de cada barra para facilitar a leitura
            for i, v in enumerate(data_totals.values()):
                ax.text(i, v + (v * 0.01), str(v), ha='center', fontweight='bold')

            plt.savefig(os.path.join(self.plots_dir, 'grafico_7_totais_relacoes_entidades.png'))
            plt.close()

        # 8. Documentos Com vs Sem Anotações
            sem_anotacoes = self.df_docs[self.df_docs['num_annotations'] == 0].shape[0]
            com_anotacoes = self.df_docs[self.df_docs['num_annotations'] > 0].shape[0]

            data_presence = {'Com Anotações': com_anotacoes, 'Sem Anotações': sem_anotacoes}

            plt.figure(figsize=(8, 6))
            # Cores: Verde para sucesso (Com), Vermelho para alerta (Sem)
            ax = sns.barplot(x=list(data_presence.keys()), y=list(data_presence.values()),
                             palette=['#2ecc71', '#e74c3c'])
            plt.title('Document Count: With vs. Without Annotations')
            plt.ylabel('Number of Documents')

            # Adiciona o número total em cima de cada barra
            for i, v in enumerate(data_presence.values()):
                ax.text(i, v + (v * 0.01), str(v), ha='center', fontweight='bold', fontsize=12)

            plt.savefig(os.path.join(self.plots_dir, 'grafico_8_com_vs_sem_anotacoes.png'))
            plt.close()

        # 9. Top 10 Entidades Mais Frequentes
        if not self.df_annotations.empty:
            # Concatena sujeitos e objetos para considerar todas as aparições
            all_entities = pd.concat([self.df_annotations['subject'], self.df_annotations['object']])
            top_10_entities = all_entities.value_counts().head(10)

            plt.figure(figsize=(12, 6))
            # O palette 'magma' ou 'rocket' costuma ser bom para ranking
            sns.barplot(x=top_10_entities.values, y=top_10_entities.index, palette='magma')
            plt.title('Top 10 Most Frequent Entities in the Corpus')
            plt.xlabel('Frequency')
            plt.ylabel('Entity Name')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'grafico_9_top_10_entidades.png'))
            plt.close()

        # =====================================================
        # 10. Distribuição do número de relações por documento (bins de 5 em 5)
        # =====================================================
        if not self.df_docs.empty:
            rels_per_doc = (
                self.df_annotations
                .groupby('doc_id')
                .size()
                .reindex(self.df_docs['id'], fill_value=0)
            )

            # Bins no formato: 0-5, 6-10, 11-15, ...
            max_rels = int(rels_per_doc.max()) if len(rels_per_doc) else 0
            max_end = ((max_rels // 5) + 1) * 5  # arredonda pra cima em múltiplos de 5

            def bin_label(n):
                n = int(n)
                if n <= 5:
                    return "0-5"
                start = ((n - 1) // 5) * 5 + 1
                end = ((n - 1) // 5 + 1) * 5
                return f"{start}-{end}"

            binned = rels_per_doc.apply(bin_label)
            bin_counts = binned.value_counts()

            # Ordena bins numericamente
            def bin_sort_key(lbl):
                a, b = lbl.split("-")
                return int(a), int(b)

            bin_counts = bin_counts.sort_index(key=lambda idx: [bin_sort_key(x) for x in idx])

            plt.figure(figsize=(12, 6))
            sns.barplot(x=bin_counts.index, y=bin_counts.values)
            plt.title('Distribution of the Number of Relationships per Document (bins of 5)')
            plt.xlabel('Range of annotated relationships')
            plt.ylabel('Number of documents')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'grafico_10_dist_relacoes_por_documento_bins5.png'))
            plt.close()

        # =====================================================
        # 11. Distribuição dos textos por número de sentenças (1 em 1)
        # =====================================================
        if not self.df_texts.empty:
            # Conta sentenças por texto usando ., !, ?, e quebras de linha
            def count_sentences(text):
                if not isinstance(text, str) or not text.strip():
                    return 0
                # divide por final de sentença ou newline
                parts = re.split(r'(?<=[\.\!\?])\s+|\n+', text.strip())
                parts = [p.strip() for p in parts if p.strip()]
                return max(1, len(parts)) if parts else 0

            sent_counts = self.df_texts['content'].apply(count_sentences)
            sent_dist = sent_counts.value_counts().sort_index()

            plt.figure(figsize=(12, 6))
            sns.barplot(x=sent_dist.index.astype(str), y=sent_dist.values)
            plt.title('Distribution of Texts by Number of Sentences (1 out of 1)')
            plt.xlabel('Number of sentences in the text')
            plt.ylabel('Quantity of texts')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'grafico_11_dist_textos_por_num_sentencas.png'))
            plt.close()

        # 12. NOVO: Histogramas de Predicados (Ambas as Entidades com ID)
        preds_both_ids = []

        # Itera sobre os dados brutos para garantir acesso aos IDs
        for doc in self.documents:
            for ann in doc.get('annotations', []):
                subj = ann.get('subject', {})
                obj = ann.get('object', {})

                # Verifica se ambos possuem o campo 'id' preenchido
                if subj.get('id') and obj.get('id'):
                    pred_label = ann.get('predicate', {}).get('entityLabel', 'Unknown')
                    preds_both_ids.append(pred_label)

        if preds_both_ids:
            # Conta a frequência (Top 10)
            counts = pd.Series(preds_both_ids).value_counts().head(10)

            plt.figure(figsize=(10, 6))
            sns.barplot(x=counts.values, y=counts.index, palette='viridis')
            plt.title('Top 10 Predicados (Ambas Entidades com ID)')
            plt.xlabel('Frequência')
            plt.ylabel('Predicado')

            # Adiciona valores nas barras
            for i, v in enumerate(counts.values):
                plt.text(v, i, f" {v}", va='center')

            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'grafico_12_predicados_ambos_ids.png'))
            plt.close()
            print("Gráfico 11 salvo.")
        else:
            print("Aviso: Nenhuma anotação encontrada com IDs em ambas as entidades.")

        # 13. NOVO: Histogramas de Predicados (Apenas Uma Entidade com ID)
        preds_one_id = []

        # Itera sobre os dados brutos
        for doc in self.documents:
            for ann in doc.get('annotations', []):
                subj = ann.get('subject', {})
                obj = ann.get('object', {})

                # Verifica presença de ID
                has_s = bool(subj.get('id'))
                has_o = bool(obj.get('id'))

                # Lógica XOR: Verdadeiro apenas se um tiver ID e o outro não
                if has_s != has_o:
                    pred_label = ann.get('predicate', {}).get('entityLabel', 'Unknown')
                    preds_one_id.append(pred_label)

        if preds_one_id:
            counts = pd.Series(preds_one_id).value_counts().head(10)

            plt.figure(figsize=(10, 6))
            sns.barplot(x=counts.values, y=counts.index, palette='magma')  # Palette 'magma' para diferenciar
            plt.title('Top 10 Predicados (Apenas Uma Entidade com ID)')
            plt.xlabel('Frequência')
            plt.ylabel('Predicado')

            for i, v in enumerate(counts.values):
                plt.text(v, i, f" {v}", va='center')

            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'grafico_13_predicados_um_id.png'))
            plt.close()
            print("Gráfico 12 salvo.")
        else:
            print("Aviso: Nenhuma anotação encontrada onde apenas uma entidade possui ID.")

        # 14. NOVO: Histogramas de Predicados (Nenhuma Entidade com ID)
        preds_no_ids = []

        # Itera sobre os dados brutos
        for doc in self.documents:
            for ann in doc.get('annotations', []):
                subj = ann.get('subject', {})
                obj = ann.get('object', {})

                # Verifica presença de ID
                has_s = bool(subj.get('id'))
                has_o = bool(obj.get('id'))

                # Lógica: NENHUMA tem ID (Nem sujeito, nem objeto)
                if not has_s and not has_o:
                    pred_label = ann.get('predicate', {}).get('entityLabel', 'Unknown')
                    preds_no_ids.append(pred_label)

        if preds_no_ids:
            counts = pd.Series(preds_no_ids).value_counts().head(10)

            plt.figure(figsize=(10, 6))
            # Palette 'Reds' para indicar "atenção" ou ausência de dados
            sns.barplot(x=counts.values, y=counts.index, palette='Reds')
            plt.title('Top 10 Predicados (Nenhuma Entidade com ID)')
            plt.xlabel('Frequência')
            plt.ylabel('Predicado')

            for i, v in enumerate(counts.values):
                plt.text(v, i, f" {v}", va='center')

            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'grafico_14_predicados_nenhum_id.png'))
            plt.close()
            print("Gráfico 13 salvo.")
        else:
            print("Aviso: Nenhuma anotação encontrada onde nenhuma entidade possui ID.")

        # =================================================================
        # 15. Histogramas de Predicados: Com ID vs Sem ID
        # =================================================================
        preds_with_id = []
        preds_without_id = []

        for doc in self.documents:
            for ann in doc.get('annotations', []):
                pred = ann.get('predicate', {})
                label = pred.get('entityLabel', 'Unknown')

                # Verifica se o predicado tem ID
                if pred.get('id'):
                    preds_with_id.append(label)
                else:
                    preds_without_id.append(label)

        # Gráfico: Predicados COM ID
        if preds_with_id:
            counts = pd.Series(preds_with_id).value_counts().head(10)
            plt.figure(figsize=(10, 6))
            sns.barplot(x=counts.values, y=counts.index, palette='Greens')
            plt.title(f'Top 10 Predicados COM ID (Total: {len(preds_with_id)})')
            plt.xlabel('Frequência')

            for i, v in enumerate(counts.values):
                plt.text(v, i, f" {v}", va='center')

            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'grafico_15_predicados_com_id.png'))
            plt.close()
            print("Gráfico 14 salvo (Predicados Com ID).")
        else:
            print("Aviso: Nenhum predicado com ID encontrado.")

        # Gráfico: Predicados SEM ID
        if preds_without_id:
            counts = pd.Series(preds_without_id).value_counts().head(10)
            plt.figure(figsize=(10, 6))
            sns.barplot(x=counts.values, y=counts.index, palette='Greys')
            plt.title(f'Top 10 Predicados SEM ID (Total: {len(preds_without_id)})')
            plt.xlabel('Frequência')

            for i, v in enumerate(counts.values):
                plt.text(v, i, f" {v}", va='center')

            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'grafico_15_predicados_sem_id.png'))
            plt.close()
            print("Gráfico 15 salvo (Predicados Sem ID).")
        else:
            print("Aviso: Nenhum predicado sem ID encontrado.")

        # =================================================================
        # 16. NOVO: Distribuição de Frequência das Entidades (> 20, bins de 10)
        # =================================================================
        if not self.df_annotations.empty:
            # 1. Conta a frequência de todas as entidades (sujeito + objeto)
            all_entities = pd.concat([self.df_annotations['subject'], self.df_annotations['object']])
            entity_counts = all_entities.value_counts()

            # 2. Filtra apenas as que têm mais de 20 ocorrências
            filtered_counts = entity_counts[entity_counts > 20]

            if not filtered_counts.empty:
                # 3. Define os intervalos (bins) de 10 em 10
                # Começa em 20, vai até o máximo encontrado + margem
                max_val = filtered_counts.max()
                bins = range(20, int(max_val) + 20, 10)

                plt.figure(figsize=(12, 6))

                # O Histplot faz exatamente o agrupamento em classes (bins)
                sns.histplot(filtered_counts, bins=bins, color='rebeccapurple', edgecolor='black')

                plt.title('Distribuição das Entidades com Alta Frequência (>20 Ocorrências)')
                plt.xlabel('Faixa de Frequência (Agrupado de 10 em 10)')
                plt.ylabel('Quantidade de Entidades Únicas')

                # Ajusta o eixo X para mostrar os intervalos de 10 em 10
                plt.xticks(list(bins))

                plt.tight_layout()
                plt.savefig(os.path.join(self.plots_dir, 'grafico_16_distribuicao_freq_entidades.png'))
                plt.close()
                print("Gráfico 16 salvo (Distribuição de Frequência).")
            else:
                print("Aviso: Nenhuma entidade com mais de 20 ocorrências para gerar o gráfico 16.")

        # =================================================================
        # 17. NOVO: Distribuição de Frequência das Entidades (< 20, passo de 1)
        # =================================================================
        if not self.df_annotations.empty:
            # 1. Conta a frequência de todas as entidades
            all_entities = pd.concat([self.df_annotations['subject'], self.df_annotations['object']])
            entity_counts = all_entities.value_counts()

            # 2. Filtra apenas as que têm MENOS de 20 ocorrências
            filtered_counts_low = entity_counts[entity_counts < 20]

            if not filtered_counts_low.empty:
                plt.figure(figsize=(12, 6))

                # 'discrete=True' força uma barra para cada número inteiro
                sns.histplot(filtered_counts_low, binwidth=1, discrete=True, color='teal', edgecolor='black')

                plt.title('Distribuição das Entidades com Baixa Frequência (< 20 Ocorrências)')
                plt.xlabel('Frequência exata (Quantas vezes a entidade apareceu)')
                plt.ylabel('Quantidade de Entidades Únicas')

                # Força mostrar todos os números de 1 a 19 no eixo X
                plt.xticks(range(1, 20))

                plt.tight_layout()
                plt.savefig(os.path.join(self.plots_dir, 'grafico_17_distribuicao_freq_entidades_menor_20.png'))
                plt.close()
                print("Gráfico 17 salvo (Distribuição de Frequência < 20).")
            else:
                print("Aviso: Nenhuma entidade com menos de 20 ocorrências para o gráfico 17.")

        print("Gráficos salvos como arquivos .png no diretório atual.")

    # =========================================================================
    # TEMA 8: EXPORTAÇÃO DE CSVs
    # =========================================================================
    def export_csvs(self):
        print("\n--- 8. Exportando CSVs ---")

        # 1. Dataset de Triplas (O Principal solicitado)
        if not self.df_annotations.empty:
            filename_triples = os.path.join(self.csv_dir, 'dataset_triplas_semanticas.csv')
            cols_to_export = ['doc_id', 'subject', 'relation', 'object', 'subject_type', 'object_type', 'source']
            self.df_annotations[cols_to_export].to_csv(filename_triples, index=False, encoding='utf-8')
            print(f"-> [SUCESSO] '{filename_triples}' gerado com {len(self.df_annotations)} triplas.")
        else:
            print("-> [AVISO] Nenhuma anotação para exportar.")

        # 2. Relatório Geral por Documento
        if not self.df_docs.empty:
            filename_docs = os.path.join(self.csv_dir, 'relatorio_geral_documentos.csv')
            df_export_docs = self.df_docs.drop(columns=['num_cols'])
            df_export_docs.to_csv(filename_docs, index=False, encoding='utf-8')
            print(f"-> [SUCESSO] '{filename_docs}' gerado.")

        # 3. Inventário de Tabelas
        if not self.df_tables.empty:
            filename_tables = os.path.join(self.csv_dir, 'inventario_tabelas.csv')
            self.df_tables.to_csv(filename_tables, index=False, encoding='utf-8')
            print(f"-> [SUCESSO] '{filename_tables}' gerado.")

        # =====================================================
        # 4. ENTITIES (todas as entidades distintas)
        # Formato: entityLabel, entityValue, annotationType, annotationTag
        # Regras: manter aspas duplas APENAS no entityLabel
        # =====================================================
        entities_set = set()

        for doc in self.documents:
            annotations = doc.get("annotations", [])
            for ann in annotations:
                # Vamos coletar subject / predicate / object (se existirem)
                for role in ["subject", "predicate", "object"]:
                    block = ann.get(role, {}) or {}

                    entity_label = block.get("entityLabel")
                    entity_value = block.get("entityValue")
                    annotation_type = block.get("annotationType")
                    annotation_tag = block.get("annotationTag") or role  # fallback seguro

                    # Ignora entradas vazias/incompletas (ajuste se quiser permitir)
                    if not entity_label or not entity_value or not annotation_type:
                        continue

                    entities_set.add((
                        str(entity_label).strip(),
                        str(entity_value).strip(),
                        str(annotation_type).strip(),
                        str(annotation_tag).strip()
                    ))

        filename_entities = os.path.join(self.csv_dir, "Entities.csv")
        with open(filename_entities, "w", encoding="utf-8", newline="") as f:
            # Header CSV
            f.write("entityLabel,entityValue,annotationType,annotationTag\n")

            # Ordena para ficar estável/reprodutível
            for (label, value, atype, atag) in sorted(entities_set):
                # Regra: aspas duplas APENAS no entityLabel
                # E escapando aspas internas caso existam
                safe_label = label.replace('"', '""')
                f.write(f"\"{safe_label}\",{value},{atype},{atag}\n")

        print(f"-> [SUCESSO] '{filename_entities}' gerado com {len(entities_set)} entidades distintas.")

        # =====================================================
        # 5. PREDICATES (todas as instâncias distintas de predicados)
        # Formato: entityLabel, annotationValue, annotationType
        # Regra: aspas duplas APENAS no annotationValue
        # =====================================================
        predicates_set = set()

        for doc in self.documents:
            annotations = doc.get("annotations", [])
            for ann in annotations:
                pred = ann.get("predicate", {}) or {}

                entity_label = pred.get("entityLabel")
                annotation_value = pred.get("annotationValue")
                annotation_type = pred.get("annotationType")

                # Ignora entradas incompletas
                if entity_label is None or annotation_value is None or annotation_type is None:
                    continue

                predicates_set.add((
                    str(entity_label).strip(),
                    str(annotation_value).strip(),
                    str(annotation_type).strip()
                ))

        filename_predicates = os.path.join(self.csv_dir, "Predicates.csv")
        with open(filename_predicates, "w", encoding="utf-8", newline="") as f:
            # Header CSV
            f.write("entityLabel,annotationValue,annotationType\n")

            for (label, ann_value, atype) in sorted(predicates_set):
                # Aspas duplas apenas no annotationValue (com escape de aspas internas)
                safe_value = ann_value.replace('"', '""')
                f.write(f"{label},\"{safe_value}\",{atype}\n")

        print(f"-> [SUCESSO] '{filename_predicates}' gerado com {len(predicates_set)} predicados distintos.")

        # 4. NOVO: Lista Única de Entidades (Label, Tipo, ID, Frequência)
        if not self.df_annotations.empty:
            # Prepara dados dos SUJEITOS
            df_subj = self.df_annotations[['subject', 'subject_type', 'subject_id', 'subject_custom']].copy()
            df_subj.columns = ['entity_label', 'entity_type', 'entity_id', 'is_custom']

            # Prepara dados dos OBJETOS
            df_obj = self.df_annotations[['object', 'object_type', 'object_id', 'object_custom']].copy()
            df_obj.columns = ['entity_label', 'entity_type', 'entity_id', 'is_custom']

            # Junta tudo numa lista só
            df_entities_all = pd.concat([df_subj, df_obj])

            # Agrupa para remover duplicatas e contar a frequência
            df_entities_unique = df_entities_all.groupby('entity_label').agg({
                'entity_type': 'first',  # Mantém o tipo
                'entity_id': 'first',  # Mantém o ID
                'is_custom': 'first',  # Mantém se é customizado
                'entity_label': 'count'  # Conta quantas vezes apareceu
            }).rename(columns={'entity_label': 'frequency'}).reset_index()

            # Ordena: das mais frequentes para as menos frequentes
            df_entities_unique = df_entities_unique.sort_values(by='entity_label', ascending=True)

            # Salva o arquivo
            filename_entities = os.path.join(self.csv_dir, 'lista_entidades.csv')
            df_entities_unique.to_csv(filename_entities, index=False, encoding='utf-8')
            print(f"-> [SUCESSO] '{filename_entities}' gerado com {len(df_entities_unique)} entidades.")

        # 5. NOVO: Lista Única de Predicados (Label, Tipo, ID, Frequência)
        if not self.df_annotations.empty:
            # Seleciona colunas relevantes
            df_preds = self.df_annotations[['relation', 'relation_type', 'relation_id']].copy()
            df_preds.columns = ['predicate_label', 'predicate_type', 'predicate_id']

            # Agrupa para contar frequência e manter dados únicos
            df_preds_unique = df_preds.groupby('predicate_label').agg({
                'predicate_type': 'first',
                'predicate_id': 'first',
                'predicate_label': 'count'
            }).rename(columns={'predicate_label': 'frequency'}).reset_index()

            # Ordena
            df_preds_unique = df_preds_unique.sort_values(by='predicate_label', ascending=True)

            # Salva
            filename_preds = os.path.join(self.csv_dir, 'lista_predicados.csv')
            df_preds_unique.to_csv(filename_preds, index=False, encoding='utf-8')
            print(f"-> [SUCESSO] '{filename_preds}' gerado com {len(df_preds_unique)} predicados.")


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    DATA_DIR = 'data'
    ARQUIVO_JSON = 'Corpus_Business_IRIT_ISWC-Train_Joint_(nous)_(without_Pertinence)_OK.json'
    LOG_TXT = "saida_visualizacao.txt"

    open(LOG_TXT, "w", encoding="utf-8").close()

    if os.path.exists(os.path.join(DATA_DIR, ARQUIVO_JSON)):
        analyzer = CorpusAnalyzer(ARQUIVO_JSON, data_dir=DATA_DIR, csv_dir='csvs', plots_dir='graficos')

        # =====================================================
        # ANÁLISES (TELA + TXT)
        # =====================================================
        analyzer.analyze_structure(output_text=LOG_TXT)
        analyzer.analyze_texts(output_text=LOG_TXT)
        analyzer.analyze_tables(output_text=LOG_TXT)
        analyzer.analyze_er(output_text=LOG_TXT)
        analyzer.analyze_cross(output_text=LOG_TXT)

        # =====================================================
        # VISUALIZAÇÃO AMIGÁVEL
        # =====================================================
        analyzer.visualize_samples(limit=2000000, output_text=LOG_TXT)

        # =====================================================
        # EXPORTAÇÕES E GRÁFICOS
        # =====================================================
        analyzer.export_csvs()
        analyzer.generate_plots()

    else:
        print(f"Erro: Arquivo '{ARQUIVO_JSON}' não encontrado em '{DATA_DIR}'.")
