# Note on Jun 19, 2025:
# The SABIO-RK website specifies that they have a RESTful API to query their
# database (http://sabio.h-its.org/layouts/content/docuRESTfulWeb/manual.gsp).
# This website states ~10% of SMILES, InChI, and InChI keys are incorrect in
# this database. Prioritize retreiving compound Name, ChEBl ID, PubChem CID 
# to verify whether 

import os
import time

import requests

from astra.constants import PROJECT_ROOT

BASE_URL_REST = "https://sabiork.h-its.org/sabioRestWebServices/"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "sabiork")
DELAY_SECONDS = 3

def make_request_and_save(endpoint, params, filename, method="POST", data=None):
    """Makes a request to the SABIO-RK API and saves the response."""
    full_url = BASE_URL_REST + endpoint
    print(f"Requesting data from: {endpoint} with params: {params}")

    try:
        if method.upper() == "POST":
            request = requests.post(full_url, params=params, data=data, timeout=300)
        elif method.upper() == "GET":
            request = requests.get(full_url, params=params, timeout=300)
        else:
            print(f"Unsupported method: {method}")
            return

        request.raise_for_status()

        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(request.text)
        print(f"Successfully saved data to {filepath}")

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error for {endpoint}: {e}")
        print(f"Response status code: {e.response.status_code}")
        print(f"Response text: {e.response.text[:500]}...")
    except requests.exceptions.RequestException as e:
        print(f"Error requesting {endpoint}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred with {endpoint}: {e}")
    finally:
        print(f"Waiting {DELAY_SECONDS} seconds before next request...")
        time.sleep(DELAY_SECONDS)

def download_all_kinetic_laws():
    """Downloads all kinetic laws using kineticlawsExportTsv (based on Script 2)."""
    endpoint = 'kineticlawsExportTsv'
    fields = [
        'EntryID', 'Organism', 'UniprotID', 'ECNumber', 'PubMedID', 'SabioReactionID',
        'Substrate', 'Product', 'Catalyst', 'Cofactor', 'Inhibitor', 'Activator',
        'Parameter','parameter.type', 'parameter.associatedSpecies', 'parameter.startValue', 'parameter.endValue', 'parameter.unit',
        'KineticMechanism', 'Pathway', 'Temperature', 'pH', 'Buffer', 'Medium',
        'EnzymeType', 'EnzymeVariant', 'Tissue', 'CellularLocation', 'ChebiID', 'ReactomeReactionID', 'InsertDate'
    ]
    # Attempting a wildcard query. This syntax might need verification from "Search Keyword Vocabulary".
    # An empty query string might also work for some APIs to mean "all".
    # If "*" doesn't work, you might need an empty query string or a very broad one.
    query_params = {
        'fields[]': fields,
        'q': '*' # Attempting wildcard, could also try q:'' or a very broad query like "ECNumber:*"
    }
    make_request_and_save(endpoint, query_params, "all_kinetic_laws.tsv", method="POST")

def download_all_compounds():
    """Downloads all compound details (based on Script 3)."""
    endpoint = 'searchCompoundDetails'
    fields = ["SabioCompoundID", "Name", "ChebiID", "PubChemID", "KeggCompoundID", "Smiles", "InChI"]
    query_params = {
        "SabioCompoundID": "*",
        "fields[]": fields
    }
    make_request_and_save(endpoint, query_params, "all_compounds.tsv", method="POST") # Script 3 uses POST

def download_all_reactions():
    """Downloads all reaction details (based on Script 4)."""
    endpoint = 'searchReactionDetails'
    fields = ["SabioReactionID", "KeggReactionID", "Enzymename", "ECNumber",
              "UniProtKB_AC", "ReactionEquation", "TransportReaction", "RheaReactionID"]
    query_params = {
        "SabioReactionID": "*",
        "fields[]": fields
    }
    make_request_and_save(endpoint, query_params, "all_reactions.tsv", method="GET") # Script 4 uses GET

def download_all_reaction_participants():
    """Downloads all reaction participant details (based on Script 5)."""
    endpoint = 'searchReactionParticipants'
    # 'SabioReactionID' is added by default for wildcard query according to docs
    fields = ["SabioCompoundID", "Name", "Role", "ChebiID", "PubChemID", "KeggCompoundID", "InChI", "Smiles"]
    query_params = {
        "SabioReactionID": "*",
        "fields[]": fields
    }
    make_request_and_save(endpoint, query_params, "all_reaction_participants.tsv", method="GET") # Script 5 uses GET

def download_all_reaction_modifiers():
    """Downloads all reaction modifier details (based on Script 6)."""
    endpoint = 'searchReactionModifiers'
    # 'SabioReactionID' is added by default for wildcard query
    fields = ["EntryID", "Name", "Role", "SabioCompoundID", "ChebiID", "PubChemID", "KeggCompoundID", "InChI", "Smiles"]
    query_params = {
        "SabioReactionID": "*",
        "fields[]": fields
    }
    make_request_and_save(endpoint, query_params, "all_reaction_modifiers.tsv", method="GET") # Script 6 uses GET

def download_all_compound_synonyms():
    """Downloads all compound synonyms (based on Script 7)."""
    endpoint = 'searchCompoundSynonyms'
    fields = ["SabioCompoundID", "Name", "NameType"]
    query_params = {
        "SabioCompoundID": "*",
        "fields[]": fields
    }
    make_request_and_save(endpoint, query_params, "all_compound_synonyms.tsv", method="POST") # Script 7 uses POST

def download_all_enzyme_synonyms():
    """Downloads all enzyme synonyms (based on Script 8)."""
    endpoint = 'searchEnzymeSynonyms'
    fields = ["ECNumber", "Name", "NameType"]
    query_params = {
        "ECNumber": "*",
        "fields[]": fields
    }
    make_request_and_save(endpoint, query_params, "all_enzyme_synonyms.tsv", method="POST") # Script 8 uses POST

def download_all_pathway_synonyms():
    """Downloads all pathway synonyms (based on Script 9)."""
    endpoint = 'searchPathwaySynonyms'
    fields = ["KeggPathwayID", "Name", "NameType"]
    query_params = {
        "PathwayName": "*",
        "fields[]": fields
    }
    make_request_and_save(endpoint, query_params, "all_pathway_synonyms.tsv", method="POST") # Script 9 uses POST

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    print("Starting SABIO-RK data download. This may take a very long time.")
    print(f"Data will be saved in '{OUTPUT_DIR}' directory.")
    print(f"Delay between requests: {DELAY_SECONDS} seconds.")
    print("Please be patient and respectful of the SABIO-RK servers.")
    print("-" * 30)

    # Call the download functions
    # You can comment out lines if you only want specific datasets or if one fails and you want to resume
    download_all_kinetic_laws()
    download_all_compounds()
    download_all_reactions()
    download_all_reaction_participants()
    download_all_reaction_modifiers()
    download_all_compound_synonyms()
    download_all_enzyme_synonyms()
    download_all_pathway_synonyms()

    print("-" * 30)
    print("All specified download tasks attempted.")
    print(f"Remember to check the '{OUTPUT_DIR}' directory for the downloaded files and any error messages above.")
    print("Consider the SMILES/InChI data quality warning from SABIO-RK when using this data.") 