# json_to_mongodb.py
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def _safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _serialize_table(table_data: List[List[Any]]) -> Optional[Dict[str, Any]]:
    """
    Converte tableData em um formato serializável:
      { "header": [...], "lines": [[...], [...], ...] }
    """
    if not table_data or not isinstance(table_data, list) or len(table_data) == 0:
        return None
    header = table_data[0] if isinstance(table_data[0], list) else []
    lines = table_data[1:] if len(table_data) > 1 else []
    return {
        "header": [str(x) for x in header],
        "lines": [[str(x) for x in row] for row in lines if isinstance(row, list)],
    }

def _extract_doc_text_and_tables(doc: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    metas = _safe_get(doc, "raw", "_source", "extractionMetadata", default=[]) or []
    texts: List[str] = []
    tables_serialized: List[Dict[str, Any]] = []

    for meta in metas:
        for t in (meta.get("texts", []) or []):
            val = t.get("value")
            if val:
                texts.append(str(val))

        for tb in (meta.get("tables", []) or []):
            table_data = tb.get("tableData", [])
            serial = _serialize_table(table_data)
            if serial:
                tables_serialized.append(serial)

    doc_text = "\n\n".join(texts).strip()
    return doc_text, tables_serialized

def _entity_metadata(block: Dict[str, Any]) -> Dict[str, Any]:
    """
    Padroniza subject/predicate/object para o schema do PDF:
      text      -> entityLabel
      wkd_id    -> entityValue
      origin    -> annotationType
      start/end -> annotationStart/annotationEnd
      row/cell  -> annotationRow/annotationCell
      value     -> annotationValue
      tag       -> annotationTag
    """
    if not isinstance(block, dict):
        return {
            "text": None, "wkd_id": None, "origin": None,
            "start": None, "end": None, "row": None, "cell": None,
            "value": None, "tag": None
        }

    return {
        "text": block.get("entityLabel"),
        "wkd_id": block.get("entityValue"),
        "origin": block.get("annotationType"),
        "start": block.get("annotationStart"),
        "end": block.get("annotationEnd"),
        "row": block.get("annotationRow"),
        "cell": block.get("annotationCell"),
        "value": block.get("annotationValue"),
        "tag": block.get("annotationTag"),
    }

def build_tb_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    doc_id = doc.get("id")
    title = doc.get("title") or _safe_get(doc, "raw", "_source", "identificationMetadata", "title")
    doc_text, tables_serialized = _extract_doc_text_and_tables(doc)

    return {
        "_id": doc_id,                 # id_doc
        "id_doc": doc_id,
        "doc_title": title,
        "doc_text": doc_text,
        "doc_serial_table": tables_serialized,  # [ {header, lines}, ... ]
    }

def build_tb_annotations(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    doc_id = doc.get("id")
    annotations = doc.get("annotations", []) or []
    out: List[Dict[str, Any]] = []

    for ann in annotations:
        ann_id = ann.get("id")

        subj = ann.get("subject", {}) or {}
        pred = ann.get("predicate", {}) or {}
        obj = ann.get("object", {}) or {}

        subj_meta = _entity_metadata(subj)
        pred_meta = _entity_metadata(pred)
        obj_meta = _entity_metadata(obj)

        out.append({
            "_id": ann_id,           # id_annot
            "id_annot": ann_id,
            "id_doc": doc_id,        # FK -> TB_DOC.id_doc

            # ids (o PDF cita subj_id / pred_id / obj_id)
            "subj_id": subj.get("id"),
            "pred_id": pred.get("id"),
            "obj_id": obj.get("id"),

            # metadados completos
            "subj_metadata": subj_meta,
            "pred_metadata": pred_meta,
            "obj_metadata": obj_meta,

            # campos “de apoio” úteis para query
            "pred_type": pred.get("entityLabel"),   # ex.: "collaboration"
        })

    return out

def write_jsonl(path: Path, docs: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",default="data/Corpus_Business_IRIT_ISWC-Train_Joint_(nous)_(without_Pertinence)_OK.json",help="Caminho do JSON do corpus")
    parser.add_argument("--outdir", default="out_mongo", help="Diretório de saída (jsonl)")
    parser.add_argument("--mongo-uri", default=None, help="Se informado, insere direto no MongoDB")
    parser.add_argument("--db", default="TRIPLE_WORKSHOP", help="Nome do DB")
    parser.add_argument("--tb-doc", default="TB_DOC", help="Nome da collection TB_DOC")
    parser.add_argument("--tb-ann", default="TB_ANNOTATION", help="Nome da collection TB_ANNOTATION")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    documents = data.get("documents", []) or []

    tb_doc_docs: List[Dict[str, Any]] = []
    tb_ann_docs: List[Dict[str, Any]] = []

    for doc in documents:
        tb_doc_docs.append(build_tb_doc(doc))
        tb_ann_docs.extend(build_tb_annotations(doc))

    tb_doc_path = outdir / "tb_doc.jsonl"
    tb_ann_path = outdir / "tb_annotation.jsonl"
    write_jsonl(tb_doc_path, tb_doc_docs)
    write_jsonl(tb_ann_path, tb_ann_docs)

    print(f"[OK] Gerado: {tb_doc_path} ({len(tb_doc_docs)} docs)")
    print(f"[OK] Gerado: {tb_ann_path} ({len(tb_ann_docs)} annotations)")

    if args.mongo_uri:
        try:
            from pymongo import MongoClient
        except ImportError:
            raise SystemExit("Instale pymongo: pip install pymongo")

        client = MongoClient(args.mongo_uri)
        db = client[args.db]

        # upsert simples (evita duplicar)
        tb_doc_col = db[args.tb_doc]
        tb_ann_col = db[args.tb_ann]

        # TB_DOC
        for d in tb_doc_docs:
            tb_doc_col.replace_one({"_id": d["_id"]}, d, upsert=True)

        # TB_ANNOTATION
        for a in tb_ann_docs:
            tb_ann_col.replace_one({"_id": a["_id"]}, a, upsert=True)

        print(f"[OK] Inserido no MongoDB: {args.db}.{args.tb_doc} e {args.db}.{args.tb_ann}")

        # Sugestão de índices úteis
        tb_ann_col.create_index("id_doc")
        tb_ann_col.create_index("pred_type")
        tb_ann_col.create_index("subj_metadata.wkd_id")
        tb_ann_col.create_index("obj_metadata.wkd_id")
        print("[OK] Índices básicos criados em TB_ANNOTATION.")

if __name__ == "__main__":
    main()
