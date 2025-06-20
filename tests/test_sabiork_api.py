# test_sabiork_api.py

import time

import pytest
import requests

BASE_URL = "https://sabiork.h-its.org/"
BASE_URL_REST = f"{BASE_URL}sabioRestWebServices/"
DELAY_SECONDS = 2

@pytest.fixture(scope="function", autouse=True)
def polite_delay():
    """
    A pytest fixture that automatically runs after each test function.
    It adds a delay to be respectful to the web server.
    'autouse=True' means it's applied to all tests in this file.
    """
    # Pass control to test function
    yield
    print(f"\nWaiting {DELAY_SECONDS} seconds...")
    time.sleep(DELAY_SECONDS)


def test_kinetic_laws_endpoint():
    """Tests the two-step kinetic law download process."""
    # Step 1: Get EntryIDs for a specific query
    url_1 = BASE_URL_REST + 'searchKineticLaws/entryIDs'
    params_1 = {'format': 'txt', 'q': 'ECNumber:"1.1.1.1"'}
    
    response_1 = requests.get(url_1, params=params_1, timeout=30)
    response_1.raise_for_status()  # Fails test on 4xx/5xx error
    
    entryIDs = [int(x) for x in response_1.text.strip().split('\n') if x.strip()]
    assert entryIDs, "Step 1 should return at least one EntryID for the query"
    
    # Step 2: Get parameters for one of the retrieved EntryIDs
    url_2 = BASE_URL + 'entry/exportToExcelCustomizable'
    params_2 = {'format': 'tsv', 'fields[]': ['EntryID', 'Organism', 'ECNumber']}
    data_2 = {'entryIDs[]': [entryIDs[0]]} # Use just the first ID for a quick test
    
    response_2 = requests.post(url_2, params=params_2, data=data_2, timeout=30)
    response_2.raise_for_status()
    assert "EntryID\tOrganism\tECNumber" in response_2.text, "Response header not found"


def test_compounds_endpoint():
    """Tests the searchCompoundDetails endpoint."""
    url = BASE_URL_REST + 'searchCompoundDetails'
    params = {"SabioCompoundID": "36", "fields[]": ["Name", "ChebiID"]}
    response = requests.post(url, params=params, timeout=30)
    response.raise_for_status()
    assert response.text.strip(), "Response should not be empty"


def test_reactions_endpoint():
    """Tests the searchReactionDetails endpoint using POST."""
    url = BASE_URL_REST + 'searchReactionDetails'
    params = {"SabioReactionID": "128", "fields[]": ["KeggReactionID", "ReactionEquation"]}
    response = requests.post(url, params=params, timeout=30)
    response.raise_for_status()
    assert response.text.strip(), "Response should not be empty"


def test_reaction_participants_endpoint():
    """Tests the searchReactionParticipants endpoint using POST."""
    url = BASE_URL_REST + 'searchReactionParticipants'
    params = {"SabioReactionID": "128", "fields[]": ["Name", "Role"]}
    response = requests.post(url, params=params, timeout=30)
    response.raise_for_status()
    assert response.text.strip(), "Response should not be empty"


def test_reaction_modifiers_endpoint():
    """Tests the searchReactionModifiers endpoint using POST."""
    url = BASE_URL_REST + 'searchReactionModifiers'
    params = {"SabioReactionID": "128", "fields[]": ["EntryID", "Name", "Role"]}
    response = requests.post(url, params=params, timeout=30)
    response.raise_for_status()
    assert response.text.strip(), "Response should not be empty"


def test_compound_synonyms_endpoint():
    """Tests the searchCompoundSynonyms endpoint."""
    url = BASE_URL_REST + 'searchCompoundSynonyms'
    params = {"SabioCompoundID": "36", "fields[]": ["Name", "NameType"]}
    response = requests.post(url, params=params, timeout=30)
    response.raise_for_status()
    assert response.text.strip(), "Response should not be empty"


def test_enzyme_synonyms_endpoint():
    """Tests the searchEnzymeSynonyms endpoint."""
    url = BASE_URL_REST + 'searchEnzymeSynonyms'
    params = {"ECNumber": "5.1.99.6", "fields[]": ["Name", "NameType"]}
    response = requests.post(url, params=params, timeout=30)
    response.raise_for_status()
    assert response.text.strip(), "Response should not be empty"


def test_pathway_synonyms_endpoint():
    """Tests the searchPathwaySynonyms endpoint."""
    url = BASE_URL_REST + 'searchPathwaySynonyms'
    params = {"KeggPathwayID": "00010", "fields[]": ["Name", "NameType"]}
    response = requests.post(url, params=params, timeout=30)
    response.raise_for_status()
    assert response.text.strip(), "Response should not be empty"