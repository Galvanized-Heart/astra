import re
from collections import defaultdict
from tqdm import tqdm

import torch


class BaseTokenizer:
    """
    A base class for tokenizers. It handles vocabulary creation,
    special tokens, and the encoding/padding logic.
    """
    def __init__(self, pad_token='[PAD]', unk_token='[UNK]', sos_token='[SOS]', eos_token='[EOS]'):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token

        # Initialize vocab with special tokens
        self.special_tokens = [pad_token, unk_token, sos_token, eos_token]
        self.vocab = defaultdict(self._get_unk_id)
        self.inverse_vocab = {}
        self._update_vocab(self.special_tokens)

    def _get_unk_id(self):
        return self.vocab[self.unk_token]

    def _update_vocab(self, tokens):
        for token in tokens:
            if token not in self.vocab:
                token_id = len(self.vocab)
                self.vocab[token] = token_id
                self.inverse_vocab[token_id] = token

    def __len__(self):
        """Returns the size of the vocabulary."""
        return len(self.vocab)

    def fit_on_sequences(self, sequences):
        """Builds the vocabulary based on a list of sequences."""
        all_tokens = set()
        for seq in sequences:
            all_tokens.update(self._tokenize(seq))
        self._update_vocab(sorted(list(all_tokens))) # sorted for consistency

    def _tokenize(self, sequence):
        """Converts a sequence string into a list of token strings. Must be implemented by subclasses."""
        raise NotImplementedError

    def encode(self, sequence):
        """Converts a sequence of tokens into a list of integer IDs, adding special tokens."""
        tokens = self._tokenize(sequence)
        encoded = [self.vocab[self.sos_token]]
        encoded.extend([self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens])
        encoded.append(self.vocab[self.eos_token])
        return encoded

    def batch_encode_plus(self, sequences, padding=True, return_tensors=True):
        """
        Tokenizes a batch of sequences, pads them to the same length, and
        returns a dictionary of tensors.
        """
        encoded_batch = [self.encode(seq) for seq in sequences]

        if not padding:
            return encoded_batch

        max_len = max(len(seq) for seq in encoded_batch)
        pad_id = self.vocab[self.pad_token]

        input_ids = []
        attention_mask = []

        for encoded_seq in tqdm(encoded_batch):
            # Pad sequence to max_len
            padded_seq = encoded_seq + [pad_id] * (max_len - len(encoded_seq))
            input_ids.append(padded_seq)

            # Create attention mask (1 for real tokens, 0 for padding)
            mask = [1] * len(encoded_seq) + [0] * (max_len - len(encoded_seq))
            attention_mask.append(mask)

        if return_tensors:
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        else:
            return {"input_ids": input_ids, "attention_mask": attention_mask}


class LigandTokenizer(BaseTokenizer):
    """
    Regex-based tokenizer for SMILES strings.
    This pattern splits SMILES into atoms (e.g., 'Cl', 'Br'), rings, bonds, etc.
    """
    def __init__(self):
        super().__init__()
        self.smiles_token_pattern = re.compile(r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])")

    def _tokenize(self, sequence):
        return self.smiles_token_pattern.findall(sequence)
    

class ProteinTokenizer(BaseTokenizer):
    """Simple character-level tokenizer for protein sequences."""
    def _tokenize(self, sequence):
        # Each amino acid is a token
        return list(sequence)