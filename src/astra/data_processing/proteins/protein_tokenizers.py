import numpy as np
from sklearn.preprocessing import OneHotEncoder
from transformers import AutoTokenizer

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWYX'
# The categories must be in a list of lists format for the encoder
categories = [list(AMINO_ACIDS)]

# Create and fit the encoder
onehot_encoder = OneHotEncoder(categories=categories, sparse_output=False)

sequence = "MKTAY"
# Reshape the sequence data for the encoder
seq_array = np.array(list(sequence)).reshape(-1, 1)

one_hot_encoded = onehot_encoder.fit_transform(seq_array)

print(f"Shape of one-hot matrix: {one_hot_encoded.shape}")
print("One-hot encoding for 'M':")
print(one_hot_encoded[0])

# Load a tokenizer for a popular protein model (ESM-2)
# This will download the tokenizer configuration and vocabulary
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

sequences = ["MKTAYIA", "ASKG"]

# The tokenizer handles everything:
# - Adding special tokens (<cls>, <eos>)
# - Converting to integer IDs
# - Padding sequences to the same length
# - Creating an attention mask (to ignore padding)
# - Returning tensors in the right format (PyTorch, TensorFlow, etc.)
outputs = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

print("Tokenizer Vocabulary Size:", tokenizer.vocab_size)
print("\nEncoded Output:")
print(outputs)

# To see the actual tokens:
print("\nDecoded Tokens for first sequence:")
tokens = tokenizer.convert_ids_to_tokens(outputs['input_ids'][0])
print(tokens)