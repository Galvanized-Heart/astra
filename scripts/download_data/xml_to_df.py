
import xml.etree.ElementTree as ET

import pandas as pd

# Path to your SABIO-RK XML file
xml_file = '../../data/raw/sabiork/sabio_batch_00152.xml'

# A list to hold the parsed data. Each item will be a dictionary representing one row.
parsed_data = []

# Use iterparse for memory efficiency
# We are interested in the 'reaction' elements
# The 'end' event means the element and its children have been fully parsed
context = ET.iterparse(xml_file, events=('end',))

for event, elem in context:
    # Check if we have finished parsing a 'reaction' element
    # The namespace {http://sabio.h-its.org/schema} might be needed depending on the file
    # For simplicity, we'll check the tag name directly. You may need to adjust this.
    if elem.tag.endswith('reaction'):
        try:
            reaction_id = elem.get('entryID')

            # Find the kinetic law (can be None)
            kinetic_law = elem.find('.//{*}kineticLaw') # The {*} ignores namespaces

            if kinetic_law:
                # Find all parameters within the kinetic law
                parameters = kinetic_law.findall('.//{*}parameter')
                for param in parameters:
                    param_type = param.get('type')
                    # We only care about Km for this example
                    if param_type == 'Km':
                        km_value = param.get('startValue')
                        substrate_ref = param.get('relSubstrate') # ID of the substrate this Km refers to

                        # Find the substrate name using the reference
                        substrate_name = ''
                        substrate_chebi = ''
                        reactant = elem.find(".//{*}reactant[@sabiorkID='" + substrate_ref + "']")
                        if reactant is not None:
                            compound = reactant.find('.//{*}compound')
                            if compound is not None:
                                substrate_name = compound.get('name')
                                # Example of finding a cross-reference
                                chebi_ref = compound.find(".//{*}crossReference[@source='CHEBI']")
                                if chebi_ref is not None:
                                    substrate_chebi = chebi_ref.get('id')
                        
                        # Find the enzyme name
                        enzyme_name = ''
                        modifier = elem.find(".//{*}modifier[@type='enzyme']")
                        if modifier is not None:
                            enzyme_name = modifier.find('.//{*}compound').get('name')


                        # Append a dictionary for this specific data point
                        parsed_data.append({
                            'reaction_id': reaction_id,
                            'enzyme': enzyme_name,
                            'substrate': substrate_name,
                            'substrate_chebi_id': substrate_chebi,
                            'parameter_type': 'Km',
                            'parameter_value': float(km_value) if km_value else None
                        })

        except Exception as e:
            print(f"Error parsing reaction {elem.get('entryID')}: {e}")

        # Clear the element from memory to keep memory usage low
        elem.clear()

# Convert the list of dictionaries to a Pandas DataFrame
df = pd.DataFrame(parsed_data)

print(df.head())
print(f"Successfully parsed {len(df)} Km entries.")

# Save to CSV
df.to_csv('sabio_rk_km_data.csv', index=False)