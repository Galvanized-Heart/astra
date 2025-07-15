import json
import re

from lxml import etree

XML_PATH = '../../data/raw/sabiork/sabio_batch_00152.xml'
OUTPUT_JSON_PATH = 'sabio_rk_data.json'

NS = {
    'sbml': 'http://www.sbml.org/sbml/level3/version1/core',
    'sbrk': 'http://sabiork.h-its.org',
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'bqbiol': 'http://biomodels.net/biology-qualifiers/'
}

def parse_source(kinetic_law):
    data = {'pubmedID': None}
    try:
        pubmed_nodes = kinetic_law.xpath(
            './/bqbiol:isDescribedBy/rdf:Bag/rdf:li[contains(@rdf:resource, "pubmed")]',
            namespaces=NS
        )
        if pubmed_nodes:
            resource_url = pubmed_nodes[0].get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource')
            data['pubmedID'] = resource_url.split('/')[-1]
    except Exception as e:
        print(f"  - Could not parse PubMed ID: {e}")
    return data

def parse_reaction_info(reaction_element):
    data = {'ecClass': None, 'reactionID': None}
    if reaction_element is None:
        return data
    try:
        ec_nodes = reaction_element.xpath(
            './/bqbiol:isVersionOf/rdf:Bag/rdf:li[contains(@rdf:resource, "ec-code")]',
            namespaces=NS
        )
        if ec_nodes:
            resource_url = ec_nodes[0].get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource')
            data['ecClass'] = resource_url.split('/')[-1]

        reac_nodes = reaction_element.xpath(
            './/bqbiol:is/rdf:Bag/rdf:li[contains(@rdf:resource, "sabiork.reaction")]',
            namespaces=NS
        )
        if reac_nodes:
            resource_url = reac_nodes[0].get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource')
            data['reactionID'] = resource_url.split('/')[-1]
    except Exception as e:
        print(f"  - Could not parse reaction info: {e}")
    return data

def parse_conditions(kinetic_law):
    data = {'temperature': None, 'tempUnits': None, 'pH': None, 'buffer': None}
    try:
        conditions = kinetic_law.find('.//sbrk:experimentalConditions', namespaces=NS)
        if conditions is None:
            return data
        temp_node = conditions.find('sbrk:temperature/sbrk:startValueTemperature', namespaces=NS)
        if temp_node is not None:
            data['temperature'] = float(temp_node.text)
        temp_unit_node = conditions.find('sbrk:temperature/sbrk:temperatureUnit', namespaces=NS)
        if temp_unit_node is not None:
            data['tempUnits'] = temp_unit_node.text
        ph_node = conditions.find('sbrk:pH/sbrk:startValuepH', namespaces=NS)
        if ph_node is not None:
            data['pH'] = float(ph_node.text)
        buffer_node = conditions.find('sbrk:buffer', namespaces=NS)
        if buffer_node is not None and buffer_node.text:
            data['buffer'] = buffer_node.text.strip()
    except Exception as e:
        print(f"  - Could not parse conditions: {e}")
    return data

def parse_kinetic_parameters(kinetic_law):
    """
    Parses kinetic parameters into robust objects, correctly distinguishing
    between base parameters (kcat, Km) and derived ones (kcat/Km).
    """
    params = {"kcat": [], "km": [], "vmax": [], "ki": [], "kcat_km": []}
    
    local_params = kinetic_law.xpath('.//sbml:localParameter', namespaces=NS)

    for p in local_params:
        param_id_attr = p.get('id', '')
        param_value = p.get('value')
        param_units = p.get('units')
        
        species_match = re.search(r'(SPC|ENZ)_\w+', param_id_attr)
        param_species_id = species_match.group(0) if species_match else None

        param_data = {
            'value': float(param_value) if param_value is not None else None,
            'units': param_units,
            'speciesID': param_species_id
        }

        param_id_lower = param_id_attr.lower()

        if 'kcat_km' in param_id_lower:
            params["kcat_km"].append(param_data)
        elif 'kcat' in param_id_lower:
            params["kcat"].append(param_data)
        elif 'km' in param_id_lower:
            params["km"].append(param_data)
        elif 'vmax' in param_id_lower:
            params["vmax"].append(param_data)
        elif 'ki' in param_id_lower:
            params["ki"].append(param_data)
            
    # Remove keys for empty lists to keep JSON clean
    return {k: v for k, v in params.items() if v}

def create_species_lookup(root):
    """Parses all species in the model once to create a lookup dictionary."""
    species_lookup = {}
    all_species_nodes = root.xpath("//sbml:species", namespaces=NS)
    for species_node in all_species_nodes:
        species_id = species_node.get('id')
        if not species_id:
            continue
        details = {
            'id': species_id,
            'name': species_node.get('name'),
            'chebi': [], 'kegg': [], 'uniprot': []
        }
        resources = species_node.xpath('.//rdf:li', namespaces=NS)
        for res in resources:
            res_url = res.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource', '')
            if 'chebi' in res_url:
                details['chebi'].append(res_url.split(':')[-1])
            elif 'kegg.compound' in res_url:
                details['kegg'].append(res_url.split('/')[-1])
            elif 'uniprot' in res_url:
                details['uniprot'].append(res_url.split('/')[-1])
        species_lookup[species_id] = details
    return species_lookup

def parse_participants(reaction_element, species_lookup):
    """Parses reactants, products, and modifiers using a pre-computed lookup table."""
    data = {'reactants': [], 'products': [], 'modifiers': [], 'uniprotID': []}
    if reaction_element is None: return data

    for reactant_ref in reaction_element.xpath('.//sbml:listOfReactants/sbml:speciesReference', namespaces=NS):
        species_id = reactant_ref.get('species')
        if species_id in species_lookup:
            data['reactants'].append(species_lookup[species_id])
    
    for product_ref in reaction_element.xpath('.//sbml:listOfProducts/sbml:speciesReference', namespaces=NS):
        species_id = product_ref.get('species')
        if species_id in species_lookup:
            data['products'].append(species_lookup[species_id])

    for modifier_ref in reaction_element.xpath('.//sbml:listOfModifiers/sbml:modifierSpeciesReference', namespaces=NS):
        species_id = modifier_ref.get('species')
        if species_id in species_lookup:
            details = species_lookup[species_id]
            data['modifiers'].append(details)
            if details.get('uniprot'):
                data['uniprotID'].extend(details['uniprot'])

    if data['uniprotID']:
        data['uniprotID'] = sorted(list(set(data['uniprotID'])))

    return data

def main():
    print(f"Loading and parsing XML file: {XML_PATH}...")
    try:
        tree = etree.parse(XML_PATH)
        root = tree.getroot()
    except OSError:
        print(f"Error: The file '{XML_PATH}' was not found.")
        return
    except etree.XMLSyntaxError as e:
        print(f"Error: The XML file is malformed. {e}")
        return

    print("Pre-computing species lookup table for efficiency...")
    species_lookup = create_species_lookup(root)
    print(f"Found {len(species_lookup)} unique species definitions.")

    kinetic_laws = root.xpath("//sbml:kineticLaw", namespaces=NS)
    print(f"Found {len(kinetic_laws)} kinetic law entries to process.")
    all_entries_data = {}

    for law in kinetic_laws:
        meta_id = law.get('metaid')
        if not meta_id or not meta_id.startswith('META_KL_'):
            continue
        entry_id = meta_id.replace('META_KL_', '')
        
        entry_data = {'entryID': entry_id}
        reaction = law.getparent()

        entry_data.update(parse_source(law))
        entry_data.update(parse_conditions(law))
        entry_data.update(parse_reaction_info(reaction))
        entry_data.update(parse_kinetic_parameters(law))
        # Pass the pre-computed lookup table to the participants parser
        entry_data.update(parse_participants(reaction, species_lookup))
        
        all_entries_data[entry_id] = entry_data

    print(f"\nSaving {len(all_entries_data)} entries to {OUTPUT_JSON_PATH}...")
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(all_entries_data, f, indent=4, sort_keys=True)
    
    print("Processing complete.")

if __name__ == "__main__":
    main()