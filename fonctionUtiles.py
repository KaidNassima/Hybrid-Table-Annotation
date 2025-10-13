from collections import defaultdict, Counter
from SPARQLWrapper import SPARQLWrapper, JSON
import argparse
import pandas as pd
import re
from collections import Counter
import numpy as np

def extract_type_name(uri):
    """ Extrait uniquement le nom du type depuis l'URI complète """
    return uri.split("/")[-1]  # Garde uniquement la dernière partie après "/"

# Fonction pour récupérer les types YAGO d'une ressource depuis l'endpoint distant
def get_yago_types(resource_name):
    sparql = SPARQLWrapper("https://yago-knowledge.org/sparql/query")  
    query = f"""
    PREFIX yago: <http://yago-knowledge.org/resource/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    SELECT ?type WHERE {{
      yago:{resource_name} rdf:type ?type .
      FILTER(?type NOT IN (
      <http://www.w3.org/2000/01/rdf-schema#Class>,
      <http://www.w3.org/2002/07/owl#Class>,
      <http://www.w3.org/1999/02/22-rdf-syntax-ns#Property>,
      <http://www.w3.org/2002/07/owl#ObjectProperty>,
      <http://www.w3.org/2002/07/owl#DatatypeProperty>,
      <http://www.w3.org/2002/07/owl#Thing>,
      <http://schema.org/Thing>
    ))
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        return {result["type"]["value"] for result in results["results"]["bindings"]}
    except Exception as e:
        print(f"Erreur SPARQL : {e}")
        return set()
    

def get_dbpedia_types(resource_name):
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")  
    query = f"""
    PREFIX dbr: <http://dbpedia.org/resource/>
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?type WHERE {{
      dbr:{resource_name} rdf:type ?type .
      
      # On filtre pour ne récupérer que les classes de l'ontologie DBpedia
      FILTER(STRSTARTS(STR(?type), "http://dbpedia.org/ontology/")) 

      # Optionnel : On s'assure que ce type est une sous-classe de quelque chose de pertinent
      
    }}
"""
   
    # query = f"""
    # PREFIX dbr: <http://dbpedia.org/resource/>
    # PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    # SELECT ?type WHERE {{
    #   dbr:{resource_name} rdf:type ?type .
    #   FILTER(
    #     STRSTARTS(STR(?type), "http://dbpedia.org/class/") ||
    #     STRSTARTS(STR(?type), "http://dbpedia.org/ontology/")
    #   )
    # }}
    # """

    # FILTER EXISTS {{ ?type rdfs:subClassOf ?supertype . 
    #                    FILTER(?supertype != <http://www.w3.org/2002/07/owl#Thing>) }}

    
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        return {result["type"]["value"] for result in results["results"]["bindings"]}
    except Exception as e:
        return set()
    
def get_column_type(attribute, column_threshold=0.5, entity_threshold=0.5):
    """Détermine si une colonne est textuelle ou numérique en fonction de la proportion de valeurs textuelles."""
    attribute = [item for item in attribute if str(item) != "nan"]
    if len(attribute) == 0:
        return False
    
    str_attribute = [item for item in attribute if isinstance(item, str)]
    str_att = [item for item in str_attribute if not item.isdigit()]
    
    for i in range(len(str_att) - 1, -1, -1):
        entity = str_att[i]
        num_count = sum(char.isdigit() for char in entity)
        if num_count / len(entity) > entity_threshold:
            del str_att[i]
    
    return len(str_att) / len(attribute) > column_threshold


def preprocessString(string):
    """Nettoie une chaîne de caractères en supprimant la ponctuation et en mettant la première lettre de chaque mot en majuscule."""
    string = re.sub(r'[^\w\s]', ' ', string)  # Supprime la ponctuation
    string = string.replace("nbsp", '')  # Supprime les espaces insécables
    string = " ".join(string.split())  # Supprime les espaces multiples
    string = string.title()  # Met en majuscule la première lettre de chaque mot
    return string


def process_table(df):
    """Filtre les colonnes de type string, sélectionne les valeurs uniques et nettoie les données."""
    processed_data = {}
    
    for col in df.columns:
        if df[col].dtype == 'object' and get_column_type(df[col].dropna().astype(str).tolist()):  # Vérifie si la colonne est textuelle
            unique_values = df[col].dropna().astype(str).tolist()
            
            # Compter la fréquence des valeurs
            value_counts = Counter(unique_values)
            sorted_values = [val for val, _ in value_counts.most_common(200)]
            
            # Nettoyage des valeurs
            cleaned_values = [preprocessString(val) for val in sorted_values]
            
            processed_data[col] = cleaned_values
    
    return processed_data

def get_yago_relations_between_types(types_dict):
    sparql = SPARQLWrapper("https://yago-knowledge.org/sparql/query")
    relations = {}
    
    columns = list(types_dict.keys())
    
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1, col2 = columns[i], columns[j]
            relations[(col1, col2)] = set()
            
            uri_list1 = [t[0] for t in types_dict[col1]]
            uri_list2 = [t[0] for t in types_dict[col2]]
            
            for uri1 in uri_list1:
                for uri2 in uri_list2:
                    query = f"""
                    SELECT DISTINCT ?relation WHERE {{
                      ?s1 a <{uri1}> .
                      ?s2 a <{uri2}> .
                      ?s1 ?relation ?s2 .
                    }}
                    """
                    
                    sparql.setQuery(query)
                    sparql.setReturnFormat(JSON)
                    results = sparql.query().convert()
                    
                    for result in results["results"]["bindings"]:
                        relations[(col1, col2)].add(result["relation"]["value"])
    
    return relations



def get_yago_common_supertype(types_dict):
    sparql = SPARQLWrapper("https://yago-knowledge.org/sparql/query")

    def get_yago_supertypes(entity_type):
        """Récupère tous les supertypes (directs et indirects) d'un type donné."""
        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?supertype WHERE {{
          <{entity_type}> rdfs:subClassOf* ?supertype .
          FILTER(?supertype != <http://schema.org/Thing>)
        }}
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        return {res["supertype"]["value"] for res in results["results"]["bindings"]}

    column_supertypes = {}

    for column, types_freq in types_dict.items():
        # Extraire uniquement les types (ignorer la fréquence)
        types = [t[0] for t in types_freq]
        
        # Récupérer les supertypes pour chaque type
        all_supertypes = [get_yago_supertypes(t) for t in types]

        # Trouver les supertypes communs
        common_supertypes = set.intersection(*all_supertypes) if all_supertypes else set()

        

        # Sélectionner le supertype le plus spécifique (le plus bas dans la hiérarchie)
        best_supertype = min(common_supertypes, key=len) if common_supertypes else None

        column_supertypes[column] = best_supertype

    return column_supertypes

from SPARQLWrapper import SPARQLWrapper, JSON

def get_dbpedia_common_supertype(types_dict):
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")

    def get_dbpedia_supertypes(entity_type):
        """Récupère tous les supertypes (directs et indirects) d'un type donné."""
        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?supertype WHERE {{
          <{entity_type}> rdfs:subClassOf* ?supertype .
          FILTER(?supertype != <http://www.w3.org/2002/07/owl#Thing>)
        }}
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        return {res["supertype"]["value"] for res in results["results"]["bindings"]}

    column_supertypes = {}

    for column, types_freq in types_dict.items():
        # Extraire uniquement les types (ignorer la fréquence)
        types = [t[0] for t in types_freq]
        
        # Récupérer les supertypes pour chaque type
        all_supertypes = [get_dbpedia_supertypes(t) for t in types]

        # Trouver les supertypes communs
        common_supertypes = set.intersection(*all_supertypes) if all_supertypes else set()

        

        # Sélectionner le supertype le plus spécifique (le plus bas dans la hiérarchie)
        best_supertype = min(common_supertypes, key=len) if common_supertypes else None

        column_supertypes[column] = best_supertype

    return column_supertypes

def get_specific_types(column_types,endpoint):
    """
    Vérifie les relations hiérarchiques entre les types DBpedia et retourne uniquement les types les plus spécifiques.
    """
    sparql = SPARQLWrapper(endpoint)
    filtered_types = {}

    for column, types in column_types.items():
        type_list = [t[0] for t in types]  # Extraire uniquement les types
        subclass_map = {t: set() for t in type_list}

        # Vérifier la relation hiérarchique entre les types
        for t1 in type_list:
            for t2 in type_list:
                if t1 != t2:
                    query = f"""
                    ASK WHERE {{
                        <{t1}> rdfs:subClassOf* <{t2}> .
                    }}
                    """
                    sparql.setQuery(query)
                    sparql.setReturnFormat(JSON)
                    try:
                        result = sparql.query().convert()
                        if result["boolean"]:  # t1 est un sous-type de t2
                            subclass_map[t2].add(t1)
                            print(f"{t1} est un sous-type de {t2}")
                    except Exception as e:
                        print(f"Erreur lors de la requête pour {t1} et {t2}: {e}")

        specific_types = {t for t in type_list if t not in subclass_map or len(subclass_map[t]) == 0}

        # Associer les fréquences aux types sélectionnés
        filtered_types[column] = [(t, freq) for t, freq in types if t in specific_types]

    return filtered_types



from collections import defaultdict

def get_yago_relations_between_entities(entity1, entity2):
    """
    Récupère les relations entre deux entités dans YAGO.
    """
    sparql = SPARQLWrapper("https://yago-knowledge.org/sparql/query")

    query = f"""
    PREFIX yago: <http://yago-knowledge.org/resource/>
    SELECT DISTINCT ?relation WHERE {{
        {{
            yago:{entity1} ?relation yago:{entity2} .
        }} UNION {{
            yago:{entity2} ?relation yago:{entity1} .
        }}
    }}
    """
    
    # query = f"""
    # PREFIX yago: <http://yago-knowledge.org/resource/>
    # SELECT ?relation WHERE {{
    #     yago:{entity1} ?relation yago:{entity2} .
    # }}
    # """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()
        return {result["relation"]["value"] for result in results["results"]["bindings"]}
    except Exception as e:
        # print(f"Erreur SPARQL : {e}")
        return set()


def get_dbpedia_relations_between_entities(entity1, entity2):
    """
    Récupère les relations entre deux entités dans DBpedia.
    """
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    query = f"""
    PREFIX dbr: <http://dbpedia.org/resource/>
    SELECT DISTINCT ?relation WHERE {{
        dbr:{entity1} ?relation dbr:{entity2} .
        UNION {{
            dbr:{entity2} ?relation dbr:{entity1} .
        }}
        # On filtre pour ne garder que les propriétés de l'ontologie DBpedia
        FILTER (STRSTARTS(STR(?relation), "http://dbpedia.org/property/"))
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()
        return {result["relation"]["value"] for result in results["results"]["bindings"]}
    except Exception as e:
        # print(f"Erreur SPARQL : {e}")
        return set()


def get_specific_relations(column_relations, endpoint):
    """
    Filtre les relations DBpedia pour ne garder que les plus spécifiques dans la hiérarchie.
    """
    sparql = SPARQLWrapper(endpoint)
    filtered_relations = defaultdict(set)

    for column, relations in column_relations.items():
        relation_list = list(relations)
        subclass_map = {r: set() for r in relation_list}

        # Vérifier la relation hiérarchique entre les relations
        for r1 in relation_list:
            for r2 in relation_list:
                if r1 != r2:
                    query = f"""
                    ASK WHERE {{
                        <{r1}> rdfs:subPropertyOf* <{r2}> .
                    }}
                    """
                    sparql.setQuery(query)
                    sparql.setReturnFormat(JSON)

                    try:
                        result = sparql.query().convert()
                        if result["boolean"]:  # r1 est une sous-relation de r2
                            subclass_map[r2].add(r1)
                            print(f" {r1} est une sous-relation de {r2}")
                    except Exception as e:
                        print(f" Erreur pour {r1} -> {r2}: {e}")

        #  Supprimer les relations trop génériques
        specific_relations = {r for r in relation_list if r not in subclass_map or len(subclass_map[r]) == 0}

        # Ajouter au dictionnaire filtré
        filtered_relations[column] = specific_relations

    return filtered_relations