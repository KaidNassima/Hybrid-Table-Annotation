import os
import pandas as pd
import json
import pickle
from collections import defaultdict
import fonctionUtiles as fU
import Annotate

# === Charger le fichier CSV de ground truth ===
gt_path = "path to /Benchmark/CPA/test/CPA_test_gt.csv"
gt_df = pd.read_csv(gt_path)

# Obtenir main_column_index par table
main_column_dict = gt_df.drop_duplicates(subset=['table_name']) \
                        .set_index('table_name')['main_column_index'].to_dict()

# === Répertoire contenant les tables ===
tables_dir = "/mnt/data/nassima/Test"
table_files = [f for f in os.listdir(tables_dir) if f.endswith(".json.gz")]

# === Fichiers de sortie ===
cta_pkl_path = "./CTA_all_incremental.pkl"
cpa_kb_only_pkl_path = "./CPA_KB_only_incremental.pkl"
cpa_kb_only_types_pkl_path = "./CPA_KB_only_types_incremental.pkl"

cpa_log_json_path = "./CPA_log.jsonl"

# === Initialiser ou charger les fichiers existants ===
if os.path.exists(cta_pkl_path):
    with open(cta_pkl_path, "rb") as f:
        cta_all = pickle.load(f)
else:
    cta_all = {}

if os.path.exists(cpa_kb_only_pkl_path):
    with open(cpa_kb_only_pkl_path, "rb") as f:
        cpa_kb_only = pickle.load(f)
else:
    cpa_kb_only = {}

if os.path.exists(cpa_kb_only_types_pkl_path):
    with open(cpa_kb_only_types_pkl_path, "rb") as f:
        cpa_kb_only_types = pickle.load(f)
else:
    cpa_kb_only_types = {}

# === Pipeline pour chaque table ===
for table_file in table_files:
    if table_file in cta_all and table_file in cpa_kb_only:
        print(f"Déjà traité : {table_file}")
        continue

    table_path = os.path.join(tables_dir, table_file)
    try:
        df = pd.read_json(table_path, compression='gzip', lines=True)
        print(f"Traitement de {table_file}...")

        # === Annotations CTA ===
        KB_annotations = Annotate.extract_column_types_with_yago(df)
        LM_annotations = Annotate.predict_column_types_with_LM(df)
        types_final = Annotate.fusion_annotations(LM_annotations, KB_annotations)
        cta_all[table_file] = types_final

        # === Annotations CPA ===
        main_col_index = main_column_dict.get(table_file, 0)
        # LM_relationships = Annotate.discover_column_relationships_with_LM(df, main_col_index)
        KB_relationships = Annotate.extract_column_relations_with_KB_I(df, main_col_index)
        relation_entre_types = Annotate.get_relations_between_types_with_KB(types_final)
        # merged_relations = Annotate.merge_relations(df, LM_relationships, KB_relationships, relation_entre_types)
        cpa_kb_only[table_file] = KB_relationships
        cpa_kb_only_types[table_file] = relation_entre_types

        # print(merged_relations)

        
        # === Sauvegarde incrémentale en pkl ===
        with open(cta_pkl_path, "wb") as f_cta, open(cpa_kb_only_pkl_path, "wb") as f_cpa:
            pickle.dump(cta_all, f_cta)
            pickle.dump(cpa_kb_only, f_cpa)
        with open(cpa_kb_only_types_pkl_path, "wb") as f_cpa_types:
            pickle.dump(cpa_kb_only_types, f_cpa_types)

        print(f"*********** Table {table_file} annotée et sauvegardée.**************\n")

    except Exception as e:
        print(f" Erreur lors du traitement de {table_file} : {e}")
