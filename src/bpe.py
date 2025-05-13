from typing import List, Set, Optional, Tuple, Dict, Union, NamedTuple


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


def ensure_list(tokens: Union[List[int], Set[int], str, bytes]) -> List[int]:
    """Convert input tokens to a list of integers.

    Args:
        tokens: Input tokens as list of integers, string, or bytes

    Returns:
        List of integer token IDs
    """
    if isinstance(tokens, str):
        tokens = tokens.encode("utf-8")
    if isinstance(tokens, bytes):
        return [int(c) for c in tokens]
    if isinstance(tokens, set):
        return [int(c) for c in tokens]
    return tokens


# base class for all tokenizers
class Tokenizer:
    """Base class for all tokenizer implementations.

    This abstract class defines the interface that all tokenizer implementations must follow.
    Tokenizers are used to convert between different token representations while preserving
    the original information.
    """

    def learn(
        self,
        tokens: Union[List[int], str, bytes],
        input_vocab: Optional[Set[int]] = None,
    ):
        """Learn tokenization patterns from input data.

        Args:
            tokens: Input tokens to learn from, can be list of integers, string, or bytes
            input_vocab: Optional set of input vocabulary tokens to consider
        """
        pass

    def encode(self, tokens: Union[List[int], str, bytes]) -> List[int]:
        """Encode input tokens into a new token representation.

        Args:
            tokens: Input tokens to encode, can be list of integers, string, or bytes

        Returns:
            List of encoded token IDs
        """
        pass

    def decode(self, tokens: List[int]) -> List[int]:
        """Decode encoded tokens back to their original representation.

        Args:
            tokens: Encoded tokens to decode

        Returns:
            List of decoded token IDs
        """
        pass


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


class BPE(Tokenizer):
    """A named tuple representing a Byte Pair Encoding (BPE) tokenizer.

    Attributes:
        merges: List of token pairs that have been merged during training
        token_values: Dictionary mapping token IDs to their corresponding values
        input_vocab: Dictionary mapping input token IDs to their corresponding values
    """

    merges: List[Tuple[int, int]]
    token_values: Dict[int, Tuple]
    input_vocab: Dict[int, int]
    max_output_vocab: int
    max_merges: int

    def __init__(
        self,
        merges: List[Tuple[int, int]] = [],
        token_values: Dict[int, Tuple] = {},
        input_vocab: Dict[int, int] = {},
        max_output_vocab: Optional[int] = None,
        max_merges: Optional[int] = None,
    ):

        if max_merges is None and max_output_vocab is None:
            raise ValueError("max_merges or max_output_vocab must be provided")
        self.merges = merges
        self.token_values = token_values
        self.input_vocab = input_vocab
        self.max_output_vocab = max_output_vocab
        self.max_merges = max_merges

    def learn(
        self,
        tokens: List[int],
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
        if input_vocab is None:
            input_vocab = set(tokens)

        input_vocab = list(input_vocab)  # convert to list to fix order.

        # the zero-token corresponds to the empty string.
        self.merges = [(0, x) for x in input_vocab]

        if self.max_output_vocab is None:
            self.max_output_vocab = self.max_merges + len(self.merges)

        self.input_vocab = set(input_vocab)

        self.token_values = {
            merge[1]: (merge[1],)
            for idx, merge in enumerate(self.merges)  # reserve zero for empty string
        }

        inverse_token_values = {v: k for k, v in self.token_values.items()}

        tokens = [inverse_token_values[(token,)] for token in tokens]

        if len(tokens) < 2:
            self.output_vocab = set(range(1, len(self.merges) + 1))
            return tokens

        next_token = max(self.token_values.keys()) + 1

        while len(self.merges) < self.max_output_vocab:

            stats = get_stats(tokens)

            most_frequent_pair = max(stats, key=stats.get)

            if stats[most_frequent_pair] == 1:
                break

            # reserve zero for empty string
            tokens = merge_pairs(tokens, most_frequent_pair, next_token)

            self.merges.append(most_frequent_pair)
            self.token_values[next_token] = (
                self.token_values[most_frequent_pair[0]]
                + self.token_values[most_frequent_pair[1]]
            )
            inverse_token_values[self.token_values[next_token]] = next_token

            next_token += 1

        self.output_vocab = set(range(1, len(self.merges) + 1))
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
        # tokens = [self.input_vocab[token] for token in tokens]
        for tok_id, pair in enumerate(self.merges):
            # if this is really just a singleton, skip the merge because there is nothing to do.
            if pair[0] == 0:
                continue

            tokens = merge_pairs(tokens, pair, tok_id + 1)
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


class DefragEncoder(Tokenizer):
    """A tokenizer that maps input vocabulary tokens to a continuous range of integers.

    This tokenizer replaces the input vocabulary with a continuous range of integers [1, len(vocab)].
    This is useful for ensuring that token IDs are contiguous and start from 1.
    We start from 1 because eventually 0 will be reserved for the empty string.

    Attributes:
        vocab_to_token: Dictionary mapping vocabulary tokens to their new token IDs
        token_to_vocab: Dictionary mapping new token IDs back to original vocabulary tokens
        input_vocab: Set of input vocabulary tokens
        output_vocab: Set of output token IDs
    """

    vocab_to_token: Dict[int, int]
    token_to_vocab: Dict[int, int]
    input_vocab: Set[int]
    output_vocab: Set[int]

    def __init__(self):
        """Initialize a new DefragEncoder with empty mappings."""
        self.vocab_to_token = {}
        self.token_to_vocab = {}
        self.input_vocab = set()
        self.output_vocab = set()

    def learn(
        self,
        tokens: Union[List[int], Set[int], str, bytes],
        input_vocab: Optional[Set[int]] = None,
    ):
        """Learn the vocabulary mapping from input tokens.

        Args:
            tokens: Input tokens to learn from, can be list of integers, set of integers, string, or bytes
            input_vocab: Optional set of input vocabulary tokens to consider
        """
        tokens = ensure_list(tokens)

        if input_vocab is None:
            self.input_vocab = set(tokens)
        else:
            self.input_vocab = input_vocab

        print("input vocab: ",input_vocab)

        self.output_vocab = set(range(1, len(self.input_vocab) + 1))
        self.vocab_to_token = {v: i + 1 for i, v in enumerate(self.input_vocab)}
        self.token_to_vocab = {i + 1: v for i, v in enumerate(self.input_vocab)}

    def encode(self, tokens: Union[List[int], str, bytes]) -> List[int]:
        """Encode input tokens using the learned vocabulary mapping.

        Args:
            tokens: Input tokens to encode, can be list of integers, string, or bytes

        Returns:
            List of encoded token IDs in the range [1, len(vocab)]
        """
        tokens = ensure_list(tokens)
        return [self.vocab_to_token[token] for token in tokens]

    def decode(self, tokens: List[int]) -> List[int]:
        """Decode encoded tokens back to their original vocabulary tokens.

        Args:
            tokens: Encoded token IDs to decode

        Returns:
            List of original vocabulary tokens
        """
        return [self.token_to_vocab[token] for token in tokens]


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
                sub_string = tuple(tokens[start + 1 : idx + 1])
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

    # zero is the "empty string" token.
    # the empty string context can generate any singleton
    contextual_tokens = {v: {0: ()} for v in vocab}
    # contextual_tokens = {v: {} for v in vocab}
    for context in vocab:
        for end_token in vocab:
            if end_token == 0:
                # the empty token must always mean the empty string.
                continue
            if len(contextual_token_counts[context][end_token]) > 0:
                most_frequent_string = max(
                    contextual_token_counts[context][end_token],
                    key=contextual_token_counts[context][end_token].get,
                )
                contextual_tokens[context][end_token] = most_frequent_string

    # empty string can generate any singleton
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

    # start with empty context
    encoded = []
    context = 0

    cur_idx = 0
    while cur_idx < len(tokens):
        best_match = 0
        best_value = ()
        for tok_idx, tok_value in contextual_tokens[context].items():
            # if tokens[cur_idx:] starts with the tuple tok_value then we have a match
            if is_prefix(tokens[cur_idx:], tok_value):
                # find the longest match
                if len(tok_value) > len(best_value):
                    best_match = tok_idx
                    best_value = tok_value
        if best_match == 0:
            print("no match found for context: ", context, " at index: ", cur_idx, " with next token: ", tokens[cur_idx])
            if context == 0:
                print("context map: ", contextual_tokens[context])
                import sys
                sys.exit(0)
        encoded.append(best_match)
        context = best_match
        cur_idx += len(best_value)

    return encoded


def contextual_decode(
    tokens: List[int],
    contextual_tokens: Dict[int, Dict[int, str]],
    initial_context: int = 0,
) -> str:
    """Decode contextually encoded tokens back to their original form.

    Args:
        tokens: List of contextually encoded token IDs
        contextual_tokens: Dictionary of contextual token mappings

    Returns:
        Decoded list of original tokens
    """
    tokens = [0] + tokens
    return [
        item
        for context, token in zip(tokens[:-1], tokens[1:])
        for item in contextual_tokens[context][token]
    ]


class ContextualEncoder(Tokenizer):
    """A tokenizer that uses context to determine token mappings.

    This tokenizer learns contextual relationships between tokens and uses them to encode
    and decode sequences. The context helps determine which token mappings to use based
    on the surrounding tokens.

    Attributes:
        contextual_tokens: Dictionary mapping context tokens to their contextual token mappings
        input_vocab: Set of input vocabulary tokens
        output_vocab: Set of output token IDs
    """

    contextual_tokens: Dict[int, Dict[int, str]]
    input_vocab: Set[int]
    output_vocab: Set[int]

    def __init__(self, contextual_tokens: Dict[int, Dict[int, str]] = {}):
        """Initialize a new ContextualEncoder.

        Args:
            contextual_tokens: Optional dictionary of pre-learned contextual token mappings
        """
        self.contextual_tokens = contextual_tokens

    def learn(
        self,
        tokens: List[int],
        input_vocab: Optional[Set[str]] = None,
    ) -> Tuple[List[int]]:
        """Learn contextual token mappings from input tokens.

        Args:
            tokens: Input tokens to learn from
            input_vocab: Optional set of input vocabulary tokens to consider

        Returns:
            Tuple containing the encoded tokens
        """
        self.contextual_tokens = learn_contextual_tokenizer(tokens, input_vocab)
        self.input_vocab = set(self.contextual_tokens.keys())
        self.output_vocab = set(self.contextual_tokens.keys())

    def encode(self, tokens: List[int]) -> List[int]:
        """Encode input tokens using contextual token mappings.

        Args:
            tokens: Input tokens to encode

        Returns:
            List of contextually encoded token IDs
        """
        tokens = ensure_list(tokens)
        encoded = contextual_encode(tokens, self.contextual_tokens)
        return encoded

    def decode(self, tokens: Union[List[int], str, bytes]) -> List[int]:
        """Decode contextually encoded tokens back to their original form.

        Args:
            tokens: Contextually encoded tokens to decode

        Returns:
            List of decoded original tokens
        """
        tokens = contextual_decode(tokens, self.contextual_tokens)
        return tokens


class ComposedTokenizer(Tokenizer):
    """A tokenizer that applies a sequence of tokenizers in order.

    This tokenizer chains multiple tokenizers together, applying them sequentially
    to transform the input tokens. The output of each tokenizer becomes the input
    to the next one in the sequence.

    Attributes:
        tokenizers: List of tokenizers to apply in sequence
        input_vocab: Set of input vocabulary tokens from the first tokenizer
        output_vocab: Set of output token IDs from the last tokenizer
    """

    def __init__(self, tokenizers: List[Tokenizer]):
        """Initialize a new ComposedTokenizer.

        Args:
            tokenizers: List of tokenizers to apply in sequence
        """
        self.tokenizers = tokenizers
        self.input_vocab = set()
        self.output_vocab = set()

    def learn(
        self,
        tokens: Union[List[int], str, bytes],
        input_vocab: Optional[Set[int]] = None,
        debug: bool = False,
    ):
        """Learn tokenization patterns from input data using all tokenizers in sequence.

        Args:
            tokens: Input tokens to learn from, can be list of integers, string, or bytes
            input_vocab: Optional set of input vocabulary tokens to consider
        """
        tokens = ensure_list(tokens)

        for idx, tokenizer in enumerate(self.tokenizers):
            tokenizer.learn(tokens, input_vocab)
            tokens = tokenizer.encode(tokens)
            input_vocab = tokenizer.output_vocab

        self.output_vocab = self.tokenizers[-1].output_vocab
        self.input_vocab = self.tokenizers[0].input_vocab

    def encode(self, tokens: Union[List[int], str, bytes]) -> List[int]:
        """Encode input tokens by applying all tokenizers in sequence.

        Args:
            tokens: Input tokens to encode, can be list of integers, string, or bytes

        Returns:
            List of encoded token IDs after applying all tokenizers
        """
        print('sdfsdfsdf')
        tokens = ensure_list(tokens)
        print("encoding tokens: ", tokens)
        for idx, tokenizer in enumerate(self.tokenizers):
            tokens = tokenizer.encode(tokens)
            print("encoded tokens: ", tokens)
        return tokens

    def decode(self, tokens: Union[List[int], str, bytes]) -> List[int]:
        """Decode encoded tokens by applying all tokenizers in reverse sequence.

        Args:
            tokens: Encoded tokens to decode, can be list of integers, string, or bytes

        Returns:
            List of decoded original tokens
        """
        tokens = ensure_list(tokens)
        for idx, tokenizer in enumerate(self.tokenizers[::-1]):
            tokens = tokenizer.decode(tokens)
        return tokens


__all__ = ["BPE", "DefragEncoder", "ContextualEncoder", "ComposedTokenizer"]

if __name__ == "__main__":

    text = "aaabdaaabacaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabcd"
    max_output_vocab = 10000
    bpe = ComposedTokenizer([DefragEncoder(), BPE(max_output_vocab=max_output_vocab)])
    bpe.learn(text, debug=False)
    tokens = bpe.encode(text)
    print("bpe encoding: ", tokens)
    print("bpe: ", bpe)
    print("bpe decoding: ", bpe.decode(tokens))

    re_encoded_tokens = bpe.encode(text)
    print("re-encoded tokens: ", re_encoded_tokens)

    print(
        "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    )

    text = "aaabdaaabacaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabcd"
    max_output_vocab = 4
    bpe = ComposedTokenizer([DefragEncoder(), BPE(max_output_vocab=max_output_vocab)])
    bpe.learn(text)
    tokens = bpe.encode(text)
    print("bpe encoding: ", tokens)
    print("bpe: ", bpe)
    print("bpe decoding: ", bpe.decode(tokens))

    re_encoded_tokens = bpe.encode(text)
    print("re-encoded tokens: ", re_encoded_tokens)

    print(
        "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    )

    contextual_bpe = ComposedTokenizer(
        [DefragEncoder(), BPE(max_output_vocab=max_output_vocab), ContextualEncoder()]
    )
    contextual_bpe.learn(text)
    print("contextual bpe: ", contextual_bpe)
    print("contextual bpe encoding: ", contextual_bpe.encode(text.encode("utf-8")))
    print(
        "contextual bpe decoding: ",
        contextual_bpe.decode(contextual_bpe.encode(text.encode("utf-8"))),
    )

    print(
        "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    )

    contextual_bpe = ComposedTokenizer(
        [DefragEncoder(), BPE(max_output_vocab=10000), ContextualEncoder()]
    )
    contextual_bpe.learn(text)
    print("contextual bpe: ", contextual_bpe)
    print("contextual bpe encoding: ", contextual_bpe.encode(text.encode("utf-8")))
    print(
        "contextual bpe decoding: ",
        contextual_bpe.decode(contextual_bpe.encode(text.encode("utf-8"))),
    )
