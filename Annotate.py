import importlib
import fonctionUtiles as fU

import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from SPARQLWrapper import SPARQLWrapper, JSON
import argparse
import json

import pickle
import csv

from torch.utils.data import Dataset
import joblib
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch


# fonction CTA avec yago 
def extract_column_types_with_yago(df,top_k=3):
    """ 
    Pour chaque colonne, récupère tous les types YAGO possibles en interrogeant l'ensemble des valeurs uniques.
    Retourne également le nombre de valeurs valides et la fréquence des types extraits.
    """
    processed_data = fU.process_table(df)  # Utilise process_table pour filtrer et nettoyer les colonnes
    column_types = defaultdict(set)
    column_valid_counts = {}  # Stocke le nombre de valeurs valides par colonne
    column_top_types = {}  # Stocke les types les plus fréquents

    for col, values in processed_data.items():
        col_types = set()  # Ensemble des types extraits pour cette colonne
        type_counter = Counter()  # Compteur des types extraits
        valid_count = 0  # Nombre de valeurs valides

        for value in values:
            yago_types = fU.get_yago_types(value.replace(" ", "_"))  # Appliquer la requête SPARQL sur la valeur
            if yago_types:  # Si YAGO renvoie des types valides
                col_types.update(yago_types)
                type_counter.update(yago_types)
                valid_count += 1

        # Associe tous les types trouvés à la colonne
        column_types[col] = set(type_counter.keys())
        column_valid_counts[col] = valid_count
        column_top_types[col] = type_counter.most_common(top_k)  # Retourne les k types les plus fréquents

    return column_top_types


# fonction CTA avec DBPEDIA 

def extract_column_types_with_dbpedia(df,top_k=3):
    """ 
    Pour chaque colonne, récupère tous les types DBpedia possibles en interrogeant l'ensemble des valeurs uniques.
    Retourne également le nombre de valeurs valides et la fréquence des types extraits.
    """
    processed_data = fU.process_table(df)  # Utilise process_table pour filtrer et nettoyer les colonnes
    column_types = defaultdict(set)
    column_valid_counts = {}  # Stocke le nombre de valeurs valides par colonne
    column_top_types = {}  # Stocke les types les plus fréquents

    for col, values in processed_data.items():
        col_types = set()  # Ensemble des types extraits pour cette colonne
        type_counter = Counter()  # Compteur des types extraits
        valid_count = 0  # Nombre de valeurs valides

        for value in values:
            dbpedia_types = fU.get_dbpedia_types(value.replace(" ", "_"))  # Appliquer la requête SPARQL sur la valeur
            if dbpedia_types:  # Si DBpedia renvoie des types valides
                col_types.update(dbpedia_types)
                type_counter.update(dbpedia_types)
                valid_count += 1

        # Associe tous les types trouvés à la colonne
        column_types[col] = set(type_counter.keys())
        column_valid_counts[col] = valid_count
        column_top_types[col] = type_counter.most_common(top_k)  # Retourne les k types les plus fréquents

    return column_top_types


# fonction CTA avec LM 


def predict_column_types_with_LM(df,max_seq_len=512):
    
    # les essentiels de cette fonction : 
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model_path = './checkpoint-215500'  
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    model.eval() 

    with open('./label_to_id.json', 'r') as f:
        label_to_id = json.load(f)

    id_to_label = {v: k for k, v in label_to_id.items()}

    predictions = {}

    # Pour chaque colonne
    for col in df.columns:
        serialized = ' '.join(df[col].astype(str).values)
        tokens = tokenizer.encode(serialized, add_special_tokens=False, truncation=True, max_length=14)

        input_ids = [tokenizer.cls_token_id] + tokens + [tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        # Padding
        pad_len = max_seq_len - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len

        # Convertir en tenseurs
        input_tensor = torch.tensor([input_ids])
        mask_tensor = torch.tensor([attention_mask])

        with torch.no_grad():
            outputs = model(input_ids=input_tensor, attention_mask=mask_tensor)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            predicted_label = id_to_label[predicted_class]

        predictions[col]=predicted_label.split('/')[0]   #ajout de split pour les types du genre class/property comme sa on aura que class

    return predictions



# meme focntion de CTA avec LM mais LE BEST MODEL FINETUNE

def predict_column_types_with_Best_LM(df,max_seq_len=512):
    
    # les essentiels de cette fonction : 
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model_path = './best_model_CTA'  
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    model.eval() 

    with open('./label_to_id.json', 'r') as f:
        label_to_id = json.load(f)

    id_to_label = {v: k for k, v in label_to_id.items()}

    predictions = {}

    # Pour chaque colonne
    for col in df.columns:
        serialized = ' '.join(df[col].astype(str).values)
        tokens = tokenizer.encode(serialized, add_special_tokens=False, truncation=True, max_length=14)

        input_ids = [tokenizer.cls_token_id] + tokens + [tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        # Padding
        pad_len = max_seq_len - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len

        # Convertir en tenseurs
        input_tensor = torch.tensor([input_ids])
        mask_tensor = torch.tensor([attention_mask])

        with torch.no_grad():
            outputs = model(input_ids=input_tensor, attention_mask=mask_tensor)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            predicted_label = id_to_label[predicted_class]

        predictions[col]=predicted_label.split('/')[0]   #ajout de split pour les types du genre class/property comme sa on aura que class

    return predictions




#  fonction de fusion des annotations CTA 

### d'abord les fonctions elementaires nécéssaires : 
from SPARQLWrapper import SPARQLWrapper, JSON
# sparql = SPARQLWrapper("https://yago-knowledge.org/sparql/query") ==> modifier juste le endpoint pour dbpedia
sparql = SPARQLWrapper("https://dbpedia.org/sparql")  

def get_supertypes(entity_type):
        """Récupère tous les supertypes (directs et indirects) d'un type donné."""
        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?supertype WHERE {{
          <{entity_type}> rdfs:subClassOf* ?supertype .
          FILTER(?supertype != <http://www.w3.org/2002/07/owl#Thing>)
        }}
        """

        # changed the filter on supertype according to the kb used ( from  schema/thing to owl:Thing)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        return {res["supertype"]["value"].split("/")[-1].lower() for res in results["results"]["bindings"]}

# Fonction pour récupérer les supertypes d’un type donné (version sans URI)
def get_yago_supertypes(entity_type):
    """Récupère tous les supertypes (directs et indirects) d'un type donné."""
    query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?supertype WHERE {{
      ?subtype rdfs:label "{entity_type}"@en .
      ?subtype rdfs:subClassOf* ?supertype .
      FILTER(?supertype != <http://www.w3.org/2002/07/owl#Thing>)
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return {res["supertype"]["value"].split("/")[-1].lower() for res in results["results"]["bindings"]}


def normalize_type(type_str):
    """Extrait le nom du type depuis un URI ou label et le met en minuscule."""
    return type_str.strip().lower().split("/")[-1]



def fusion_annotations(colonnes_lm, colonnes_kb):
    fusion_table = {}


    for colonne, type_lm in colonnes_lm.items():
        lm_norm = normalize_type(type_lm)
        types_kb_info = colonnes_kb.get(colonne, [])
        types_kb = [t[0] for t in types_kb_info]  # on ne garde que le type, pas le score

        if not types_kb:
            print(f"[{colonne}] Aucun type KB → on garde le type LM : {type_lm}")
            fusion_table[colonne] = [type_lm]
            continue

        types_kb_norm = [normalize_type(t) for t in types_kb]
        
        fusion = []
        type_lm_retained = True  # on suppose qu’on garde LM, sauf si doublon ou plus spécifique trouvé

        for kb_type, kb_norm in zip(types_kb, types_kb_norm):
            if lm_norm == kb_norm:
                print(f"[{colonne}] Type LM ({lm_norm}) est identique à KB ({kb_norm}) → on garde KB")
                fusion.append(kb_type)
                type_lm_retained = False
            else:
                supertypes_lm = get_yago_supertypes(lm_norm)  # changed
                supertypes_kb = get_supertypes(kb_type)

                if kb_norm in supertypes_lm:
                    print(f"[{colonne}] Type LM ({lm_norm}) est un sous-type de KB ({kb_norm}) → on garde LM")
                    continue  # on garde LM, donc on n'ajoute pas ce KB
                elif lm_norm in supertypes_kb:
                    print(f"[{colonne}] Type KB ({kb_norm}) est un sous-type de LM ({lm_norm}) → on garde KB")
                    fusion.append(kb_type)
                    type_lm_retained = False
                else:
                    print(f"[{colonne}] Aucun lien hiérarchique entre LM ({lm_norm}) et KB ({kb_norm}) → on garde les deux")
                    fusion.append(kb_type)

        if type_lm_retained:
            print(f"[{colonne}] Type LM ({type_lm}) retenu dans la fusion")
            fusion.append(type_lm)

        fusion_table[colonne] = list(set(fusion))  # enlever les doublons éventuels

    # Ajouter les colonnes présentes uniquement dans KB
    for colonne, types_kb_info in colonnes_kb.items():
        if colonne not in fusion_table:
            print(f"[{colonne}] Présente uniquement dans KB → on garde tous les types KB")
            fusion_table[colonne] = [t[0] for t in types_kb_info]

    return fusion_table

# fonction CPA avec LM 



def discover_column_relationships_with_LM(table,main_column_index, max_length=512):

    # les essentiels de cette fonction : 
    label_encoder = joblib.load('./encoder.pkl')
    checkpoint_dir = './checkpoint_roberta'
    model = RobertaForSequenceClassification.from_pretrained(checkpoint_dir)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')



    model.eval()


    # Nombre de colonnes dans la table
    num_columns = table.shape[1]

    # Initialiser une liste pour stocker les résultats
    results = []

    # Comparer main colonne avec les autres colonnes
    for j in range(num_columns):
        if j== main_column_index:
            continue

        column_1_values = table.iloc[:, main_column_index].astype(str).tolist()[:20]
        column_2_values = table.iloc[:, j].astype(str).tolist()[:20]

        sequence = f"[Column 1] {' '.join(column_1_values)}[SEP] [Column 2] {' '.join(column_2_values)}"

            # Tokenisation
        encoding = tokenizer(sequence, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

            # Faire une prédiction
        with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                # predicted_label = torch.argmax(logits, dim=1).item()
                probabilities = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
                predicted_label = torch.argmax(logits, dim=1).item()
                predicted_prob = probabilities[predicted_label]

        # Décoder le label prédit
        decoded_label = label_encoder.inverse_transform([predicted_label])[0]

            # Convertir la prédiction en label humain
        results.append({
                'column_1_index': main_column_index,
                'column_2_index': j,
                'predicted_label': decoded_label,
                'predicted_probability': predicted_prob
            })

            # # Filtrer basé sur le seuil de probabilité
            # if predicted_prob >= threshold:
            #     results.append({
            #         'column_1_index': column_1_index,
            #         'column_2_index': column_2_index,
            #         'predicted_label': predicted_label,
            #         'predicted_probability': predicted_prob
            #     })

    return results

# fonction CPA avec yago version instances : 

from collections import defaultdict, Counter

def extract_column_relations_with_KB_I(df, main_col, top_k=None):   
    """ 
    Pour chaque colonne (hors main_col), essaie d'obtenir des relations YAGO 
    en interrogeant les paires de valeurs après nettoyage.
    Affiche les probas des relations et permet de récupérer le top-k des relations.
    """
    processed_data = fU.process_table(df)  # Nettoyage des colonnes avant interrogation
    column_relations = defaultdict(dict)
    column_relation_frequencies = defaultdict(Counter)  # Stocke la fréquence des relations
    column_valid_counts = {}  # Nombre de paires valides par colonne
    
    for col in df.columns:
        if col == main_col or col not in processed_data:
            continue  # Ignore la colonne principale et les colonnes non string
        
        pairs = [(fU.preprocessString(v1), fU.preprocessString(v2)) 
                 for v1, v2 in df[[main_col, col]].dropna().astype(str).drop_duplicates().values]
        
        col_relations = {}  # Stocke les relations trouvées pour cette colonne
        relation_counter = Counter()  # Compteur des relations
        valid_count = 0  # Nombre de paires valides
        
        for entity1, entity2 in pairs:
            relations = fU.get_yago_relations_between_entities(entity1.replace(" ", "_"), 
                                                       entity2.replace(" ", "_"))  
            if relations:
                
                relation_counter.update(relations)
                valid_count += 1
        
        # Stocker toutes les relations trouvées
        key= (main_col, col)
        

        # au lieu d'avoir des frequences, j'aurai des probs
        total_relations = sum(relation_counter.values())
        
        if total_relations > 0:
            # Calculer une distribution de probabilité
            probas = {rel: freq / total_relations for rel, freq in relation_counter.items()}
        else:
            probas = {}

        column_relations[key]=probas
        column_relation_frequencies[key] = probas
        column_valid_counts[key] = valid_count

        # Affichage des relations avec leur probabilité
        print(f"\nRelations pour la paire ('{main_col}','{col}') (paires valides: {valid_count}):")
        sorted_probas = sorted(probas.items(), key=lambda x: x[1], reverse=True)
        for rel, prob in sorted_probas:
            print(f"  {rel}: {prob:.3f}")

        # Filtrer pour garder uniquement le top-k si demandé
        if top_k:
            column_relations[key] = dict(sorted_probas[:top_k])

         
        # column_relation_frequencies[key] = dict(relation_counter)
        # column_valid_counts[key] = valid_count

        # # Affichage des relations et de leur fréquence
        # print(f"\nRelations pour la paire ('{main_col}','{col}' (paires valides: {valid_count}):")
        # sorted_relations = relation_counter.most_common()
        # for rel, freq in sorted_relations:
        #     print(f"  {rel}: {freq}")

        # Filtrer pour garder uniquement le top-k si demandé
        # if top_k:
        #     column_relation_frequencies[key] = dict(sorted_relations[:top_k]) 

    return column_relations

# fonction CPA avec yago version entre_types :

import time
from SPARQLWrapper import SPARQLWrapper, JSON

def get_relations_between_types_with_KB(types_dict):
    """
     Récupère les relations RDF entre les colonnes à partir des types fusionnés (LM + KB).
    Pour chaque paire de colonnes, on interroge YAGO avec toutes les combinaisons (type1, type2).
    On retourne pour chaque relation :
      - une probabilité (basée sur le nombre d’occurrences sur les paires de types)
      - la ou les paires de types qui ont permis de l’inférer
    """
    sparql = SPARQLWrapper("https://yago-knowledge.org/sparql/query")
    relations = {}
    columns = list(types_dict.keys())

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1, col2 = columns[i], columns[j]
            # relations[(col1, col2)] = set()
            relations[(col1, col2)] = {}

            list1 = types_dict[col1]
            list2 = types_dict[col2]

            for type1 in list1:
                for type2 in list2:

                    # Générer la bonne clause pour chaque type (URI ou label)
                    if type1.startswith("http"):
                        clause1 = f"?s1 a <{type1}> ."
                    else:
                        clause1 = f'?s1 a ?type1 . ?type1 rdfs:label "{type1}"@en .'

                    if type2.startswith("http"):
                        clause2 = f"?s2 a <{type2}> ."
                    else:
                        clause2 = f'?s2 a ?type2 . ?type2 rdfs:label "{type2}"@en .'



                    query = f"""
                    SELECT DISTINCT ?relation WHERE {{
                      {clause1}
                      {clause2}
                      ?s1 ?relation ?s2 .
                    }} LIMIT 50
                    """
                    try:
                        sparql.setQuery(query)
                        sparql.setReturnFormat(JSON)
                        print(f"Querying relation between [{type1}]and [{type2}]...")
                        results = sparql.query().convert()

                        for result in results["results"]["bindings"]:
                            # relations[(col1, col2)].add(result["relation"]["value"])
                            rel = result["relation"]["value"]
                            if rel not in relations[(col1,col2)]:
                                relations[(col1, col2)][rel]= {
                                    "count": 1,
                                    "type_pairs": set([(type1, type2)])
                                }
                            else:
                                relations[(col1, col2)][rel]["count"] += 1
                                relations[(col1, col2)][rel]["type_pairs"].add((type1, type2))


                    except Exception as e:
                        print(f"Erreur pour {type1} et {type2} : {e}")
                    
            # Calcul des probabilités pour chaque relation
            rel_info = relations[(col1, col2)]
            total_count = sum(info["count"] for info in rel_info.values())
            for rel, info in rel_info.items():
                info["proba"] = info["count"] / total_count if total_count > 0 else 0.0

            # Affichage (optionnel mais utile pour vérification)
            print(f"\nRelations trouvées pour ({col1}, {col2}) :")
            for rel, info in rel_info.items():
                types_str = "; ".join([f"{t1} ↔ {t2}" for t1, t2 in info["type_pairs"]])
                print(f"  {rel}: {info['proba']:.3f} (types: {types_str})")

    return relations

# fonction de fusion de relations avec info de provenance 

import numpy as np

def normalize_relation_name(relation):
    """Supprime les URI et met en forme canonique la relation (minuscule, sans espace, etc.)."""
    if isinstance(relation, str):
        return relation.strip().lower().split('/')[-1].split('#')[-1]
    return relation

def merge_relations(df, lm_relations, kb_instances, kb_entretypes):
    """Fusionne les relations extraites des modèles de langage (LM), des instances de la KB et des relations entre types de la KB."""
    
    # Créer un mappage des noms de colonnes vers leurs indices
    column_name_to_index = {col: idx for idx, col in enumerate(df.columns)}
    
    merged_relations = defaultdict(lambda: defaultdict(dict))  # Pour stocker les relations fusionnées
    
    # Fusionner les relations LM
    for lm_entry in lm_relations:
        col1_idx = lm_entry['column_1_index']
        col2_idx = lm_entry['column_2_index']
        relation_lm = lm_entry['predicted_label']
        prob_lm = lm_entry['predicted_probability']
        
        key = (col1_idx, col2_idx)
        norm_relation = normalize_relation_name(relation_lm)

        if norm_relation not in merged_relations[key]:
            merged_relations[key][norm_relation] = {
                'original_names': set(),
                'probabilities': [],
                'sources': []
            }
        
        merged_relations[key][norm_relation]['original_names'].add(relation_lm)
        merged_relations[key][norm_relation]['probabilities'].append(prob_lm)
        merged_relations[key][norm_relation]['sources'].append('LM')
    
    # Fusionner les relations KB instances
    for (col1, col2), kb_rels in kb_instances.items():
        col1_idx = column_name_to_index.get(col1, None)
        col2_idx = column_name_to_index.get(col2, None)

        if col1_idx is not None and col2_idx is not None:
            key = (col1_idx, col2_idx)

            for relation, prob in kb_rels.items():
                norm_relation = normalize_relation_name(relation)
                
                if norm_relation not in merged_relations[key]:
                    merged_relations[key][norm_relation] = {
                        'original_names': set(),
                        'probabilities': [],
                        'sources': []
                    }

                merged_relations[key][norm_relation]['original_names'].add(relation)
                merged_relations[key][norm_relation]['probabilities'].append(prob)
                merged_relations[key][norm_relation]['sources'].append('KB_instance')
    
    # Fusionner les relations KB entretypes
    for (col1, col2), relation_data in kb_entretypes.items():
        col1_idx = column_name_to_index.get(col1, None)
        col2_idx = column_name_to_index.get(col2, None)

        if col1_idx is not None and col2_idx is not None:
            key = (col1_idx, col2_idx)

            for relation, data in relation_data.items():
                prob = data['proba']
                for type_pair in data['type_pairs']:
                    relation_with_types = f"{relation} ({type_pair[0]} -> {type_pair[1]})"
                    norm_relation = normalize_relation_name(relation)

                    if norm_relation not in merged_relations[key]:
                        merged_relations[key][norm_relation] = {
                            'original_names': set(),
                            'probabilities': [],
                            'sources': []
                        }

                    merged_relations[key][norm_relation]['original_names'].add(relation_with_types)
                    merged_relations[key][norm_relation]['probabilities'].append(prob)
                    merged_relations[key][norm_relation]['sources'].append('KB_entretypes')

    # Calcul de la probabilité moyenne et structure finale
    final_relations = defaultdict(dict)
    for key, relations in merged_relations.items():
        for norm_relation, data in relations.items():
            probas = np.array(data['probabilities'])
            prob_combined = np.mean(probas)

            final_relations[key][norm_relation] = {
                'probability': prob_combined,
                'sources': list(set(data['sources'])),  # Pour éviter les doublons
                'individual_probabilities': probas.tolist(),
                'original_names': list(data['original_names'])
            }

    return final_relations
