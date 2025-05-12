from typing import List, Set, Optional, Tuple, Dict, Union, NamedTuple


### byte pair encoding implementation
def get_stats(tokens: List[int]) -> Dict[Tuple[int, int], int]:
    """Calculate frequency statistics of adjacent token pairs in the input sequence.
    
    Args:
        tokens: List of token IDs to analyze
        
    Returns:
        Dictionary mapping token pairs to their frequency counts
    """
    stats = {}
    for pair in zip(tokens[:-1], tokens[1:]):
        stats[pair] = stats.get(pair, 0) + 1

    return stats


def merge_pairs(tokens: List[int], pair: Tuple[int, int], new_token: int) -> List[int]:
    """Merge all occurrences of a token pair into a single new token.
    
    Args:
        tokens: List of token IDs to process
        pair: Tuple of two token IDs to merge
        new_token: Token ID to use for the merged pair
        
    Returns:
        New list of tokens with all occurrences of the pair merged
    """
    merged = []
    cur_idx = 0
    while cur_idx < len(tokens):
        if (
            cur_idx < len(tokens) - 1
            and tokens[cur_idx] == pair[0]
            and tokens[cur_idx + 1] == pair[1]
        ):
            merged.append(new_token)
            cur_idx += 1
        else:
            merged.append(tokens[cur_idx])

        cur_idx += 1
    return merged


class BPE:
    """A named tuple representing a Byte Pair Encoding (BPE) tokenizer.
    
    Attributes:
        merges: List of token pairs that have been merged during training
        token_values: Dictionary mapping token IDs to their corresponding values
        input_vocab: Dictionary mapping input token IDs to their corresponding values
    """
    merges: List[Tuple[int, int]]
    token_values: Dict[int, Tuple]
    input_vocab: Dict[int, int]

    def __init__(self, merges: List[Tuple[int, int]] = [], token_values: Dict[int, Tuple] = {}, input_vocab: Dict[int, int] = {}):
        self.merges = merges
        self.token_values = token_values
        self.input_vocab = input_vocab
    

    def learn(
        self,
        tokens: Union[List[int], bytes, str],
        max_output_vocab: int,
        input_vocab: Optional[Set] = None,
    ) -> List[int]:
        """Learn a BPE tokenizer from input tokens.
        
        Args:
            tokens: Input tokens as list of integers, bytes, or string
            max_output_vocab: Maximum size of the output vocabulary
            input_vocab: Optional set of input vocabulary tokens
            
        Returns:
            Tuple containing:
                - List of tokenized input
                - BPE object containing merges and token values
        """
        tokens = ensure_list(tokens)
        if input_vocab is None:
            input_vocab = set(tokens)

        input_vocab = list(input_vocab) # convert to list to fix order.

        # the zero-token corresponds to the empty string.
        self.merges = [(0, x) for x in input_vocab]

        self.input_vocab = {x: i+1 for i, x in enumerate(input_vocab)}

        self.token_values = {
            idx + 1: (merge[1],)
            for idx, merge in enumerate(self.merges)  # reserve zero for empty string
        }

        inverse_token_values = {v: k for k, v in self.token_values.items()}

        tokens = [inverse_token_values[(token,)] for token in tokens]

        if len(tokens) < 2:
            return tokens


        while len(self.merges) < max_output_vocab:

            stats = get_stats(tokens)

            most_frequent_pair = max(stats, key=stats.get)

            if stats[most_frequent_pair] == 1:
                break

            new_tok = len(self.merges) + 1  # reserve zero for empty string
            tokens = merge_pairs(tokens, most_frequent_pair, new_tok)
            self.merges.append(most_frequent_pair)
            self.token_values[new_tok] = (
                self.token_values[most_frequent_pair[0]] + self.token_values[most_frequent_pair[1]]
            )
            inverse_token_values[self.token_values[new_tok]] = new_tok

        return tokens


    def encode(self, tokens: Union[List[int], str, bytes]) -> List[int]:
        """Encode input tokens using a trained BPE tokenizer.
        
        Args:
            tokens: List of input token IDs
            bpe: Trained BPE tokenizer
            
        Returns:
            List of encoded token IDs
        """
        tokens = ensure_list(tokens)
        tokens = [self.input_vocab[token] for token in tokens]
        for tok_id, pair in enumerate(self.merges):
            # if this is really just a singleton, skip the merge because there is nothing to do.
            if pair[0] == 0:
                continue

            tokens = merge_pairs(tokens, pair, tok_id+1)
        return tokens


    def decode(self, tokens: Union[List[int], str, bytes]) -> str:
        """Decode BPE-encoded tokens back to their original form.
        
        Args:
            tokens: List of encoded token IDs, or string/bytes to decode
            bpe: Trained BPE tokenizer
            
        Returns:
            Decoded list of original tokens
        """
        tokens = ensure_list(tokens)
        return [tok for token in tokens for tok in self.token_values[token]]


### secondary processing to find "contextual" tokenization


def get_context_stats(tokens: List[int], vocab: Optional[Set[int]]):
    """Calculate statistics about token sequences in different contexts.

    specifically, we return a dictionary such thatfor each pair of possible tokens X,Y
    and each string S = X...Y appearing in the input token list such that X and Y do not appear in the interior of the string,
    result[X][Y][S[1:]] = # of times S appears in the input token list.

    Args:
        tokens: List of token IDs to analyze
        vocab: Optional set of vocabulary tokens to consider
        
    Returns:
        Dictionary containing statistics about token sequences in different contexts
    """
    # build stats dictionaries
    stats = {context: {token: {} for token in vocab} for context in vocab}

    # build "starting point" index lookup

    start_idx = {v: -1 for v in vocab}

    for idx, token in enumerate(tokens):

        # update the string finders for all the other tokens

        for v in vocab:
            start = start_idx[v]
            if start != -1:
                sub_string = tuple(tokens[start+1:idx+1])
                stats[v][token][sub_string] = stats[v][token].get(sub_string, 0) + 1
        # restart the string finder for the current token
        start_idx[token] = idx

    return stats


def learn_contextual_tokenizer(
    tokens: Union[List[int], bytes, str], vocab: Optional[Set[int]] = None
) -> Dict[int, Dict[int, str]]:
    """Learn a contextual tokenizer from input tokens.

    The contextual tokenizer is a dictionary mapping token ids to another dictionary that maps
    token ids to lists of tokens.

    This dictionary is constructed in such a way that tokenizer[X][Y] is the longest substring
    appearing in the input token list that starts with X, ends with Y and does not contain X or Y
    in the interior of the substring.
    
    Args:
        tokens: Input tokens as list of integers, bytes, or string
        vocab: Optional set of vocabulary tokens to consider
        
    Returns:
        Dictionary mapping context tokens to their contextual token mappings
    """
    tokens = ensure_list(tokens)
    if vocab is None:
        vocab = set(tokens)

    contextual_token_counts = get_context_stats(tokens, vocab)

    contextual_tokens = {v: {0: ()} for v in vocab}

    for context in vocab:
        for end_token in vocab:
            if len(contextual_token_counts[context][end_token]) > 0:
                longest_string = max(
                    contextual_token_counts[context][end_token],
                    key=contextual_token_counts[context][end_token].get,
                )
                contextual_tokens[context][end_token] = longest_string

    contextual_tokens[0] = {v: (v,) for v in vocab}

    return contextual_tokens




def contextual_encode(
    tokens: List[int], contextual_tokens: Dict[int, Dict[int, str]]
) -> List[int]:
    """Encode tokens using a contextual tokenizer.
    
    Args:
        tokens: List of input token IDs
        contextual_tokens: Dictionary of contextual token mappings
        
    Returns:
        List of contextually encoded token IDs
    """
    encoded = [0]

    # start with empty context
    context = 0

    cur_idx = 0

    while cur_idx < len(tokens):
        best_match = 0
        best_value = ()
        for tok_idx, tok_value in contextual_tokens[context].items():
            # if tokens[cur_idx:] starts with the tuple tok_value then we have a match
            if is_prefix(tokens[cur_idx:], tok_value):
                best_match = tok_idx
                best_value = tok_value

        encoded.append(best_match)
        context = best_match
        cur_idx += len(best_value)

    return encoded


def contextual_decode(
    tokens: List[int], contextual_tokens: Dict[int, Dict[int, str]]
) -> str:
    """Decode contextually encoded tokens back to their original form.
    
    Args:
        tokens: List of contextually encoded token IDs
        contextual_tokens: Dictionary of contextual token mappings
        
    Returns:
        Decoded list of original tokens
    """
    return [
        item
        for context, token in zip(tokens[:-1], tokens[1:])
        for item in contextual_tokens[context][token]
    ]


def is_prefix(sequence: List[int], prefix: Tuple[int, ...]) -> bool:
    """Check if a sequence starts with a given prefix.
    
    Args:
        sequence: List of integers to check
        prefix: Tuple of integers representing the prefix
        
    Returns:
        True if sequence starts with prefix, False otherwise
    """
    if len(prefix) > len(sequence):
        return False
    return all(sequence[i] == prefix[i] for i in range(len(prefix)))


def ensure_list(tokens: Union[List[int], str, bytes]) -> List[int]:
    """Convert input tokens to a list of integers.
    
    Args:
        tokens: Input tokens as list of integers, string, or bytes
        
    Returns:
        List of integer token IDs
    """
    if isinstance(tokens, str):
        return tokens.encode("utf-8")
    elif isinstance(tokens, bytes):
        return [int(c) for c in tokens]
    return tokens



class ContextualBPE:
    """A named tuple representing a contextual BPE tokenizer.
    
    Attributes:
        bpe: Base BPE tokenizer
        contextual_tokens: Dictionary mapping context tokens to their contextual token mappings
    """
    bpe: BPE
    contextual_tokens: Dict[int, Dict[int, str]]

    def __init__(self, bpe: BPE = BPE(), contextual_tokens: Dict[int, Dict[int, str]] = {}):
        self.bpe = bpe
        self.contextual_tokens = contextual_tokens


    def learn(
        self,
        text: str, max_output_vocab: int, input_vocab: Optional[Set[str]] = None
    ) -> Tuple[List[int]]:
        """Learn a contextual BPE tokenizer from input text.
        
        Args:
            text: Input text to learn from
            max_output_vocab: Maximum size of the output vocabulary
            input_vocab: Optional set of input vocabulary tokens
            
        Returns:
            Tuple containing:
                - List of encoded tokens
                - ContextualBPE object containing the trained tokenizer
        """
        self.bpe = BPE()
        bpe_tokens = self.bpe.learn(text, max_output_vocab, input_vocab)
        bpe_vocab = set(range(len(self.bpe.merges) + 1))
        self.contextual_tokens = learn_contextual_tokenizer(bpe_tokens, bpe_vocab)

        # encoded = contextual_encode(bpe_tokens, self.contextual_tokens)
        # return encoded






    def encode(
        self,
        tokens: Union[List[int], str, bytes]
    ) -> List[int]:
        """Perform end-to-end encoding using a contextual BPE tokenizer.
        
        Args:
            tokens: Input tokens as list of integers, string, or bytes
            
        Returns:
            List of encoded token IDs
        """
        tokens = ensure_list(tokens)
        bpe_tokens = self.bpe.encode(tokens)
        encoded = contextual_encode(
            bpe_tokens, self.contextual_tokens
        )
        return encoded


    def decode(
        self,
        tokens: Union[List[int], str, bytes]
    ) -> List[int]:
        """Perform end-to-end decoding using a contextual BPE tokenizer.
        
        Args:
            tokens: Input tokens as list of integers, string, or bytes
            
        Returns:
            List of decoded token IDs
        """
        tokens = contextual_decode(tokens, self.contextual_tokens)
        tokens = self.bpe.decode(tokens)
        return tokens


if __name__ == "__main__":
    text = "aaabdaaabac"
    max_output_vocab = 100
    bpe = BPE()
    tokens = bpe.learn(text, 100)
    print("bpe encoding: ", tokens)
    print("bpe: ", bpe)
    print("bpe decoding: ", bpe.decode(tokens))

    re_encoded_tokens = bpe.encode(text.encode("utf-8"))
    print("re-encoded tokens: ", re_encoded_tokens)

    contextual_tokenizer = learn_contextual_tokenizer(
        tokens, set(range(len(bpe.merges) + 1))
    )
    print("contextual tokenizer: ", contextual_tokenizer)
    tokens = contextual_encode(tokens, contextual_tokenizer)
    print("contextual tokens: ", tokens)
    print(
        "contextual decoding: ", contextual_decode(tokens, contextual_tokenizer)
    )

    contextual_bpe = ContextualBPE()
    contextual_bpe.learn(text, 100)
    print("contextual bpe: ", contextual_bpe)
    print("contextual bpe encoding: ", contextual_bpe.encode(text.encode("utf-8")))
    print("contextual bpe decoding: ", contextual_bpe.decode(contextual_bpe.encode(text.encode("utf-8"))))
