from rdkit import Chem

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYUX")

def is_valid_protein_sequence(seq):
    """Checks if a protein sequence contains only valid amino acid characters."""
    if not isinstance(seq, str) or not seq:
        return False
    return set(seq.upper()) <= VALID_AMINO_ACIDS

def is_valid_smiles(smiles):
    """Checks if a SMILES string is valid using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def preprocess_and_validate_data(df, prot_col, lig_col):
    """Loads data, validates sequences and SMILES, and returns a clean DataFrame."""    
    initial_count = len(df)
    print(f"Loaded {initial_count} rows.")

    # Validate protein sequences
    df['protein_valid'] = df[prot_col].apply(is_valid_protein_sequence)
    
    # Validate SMILES strings
    df['smiles_valid'] = df[lig_col].apply(is_valid_smiles)

    # Filter out invalid rows
    valid_df = df[df['protein_valid'] & df['smiles_valid']].copy()
    final_count = len(valid_df)
    
    dropped_count = initial_count - final_count
    if dropped_count > 0:
        print(f"Dropped {dropped_count} invalid rows.")

    # Clean up validation columns
    valid_df = valid_df.drop(columns=['protein_valid', 'smiles_valid'])
    
    return valid_df.reset_index(drop=True)
