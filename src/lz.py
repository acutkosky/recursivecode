from typing import Any, Optional, Dict, Set, Tuple, Union, List
import pygtrie


# we will replace input symbols not seen in the "learning" phase with 0
# in the encoding and decoding phase.
# Hopefully this doesn't really happen too much.
UNKNOWN_SYMBOL = 0
EMPTY_TOKEN = -1


TOKEN_TYPE = int


INPUT_SYMBOL_SEQUENCE_TYPE = Union[str, bytes, List[TOKEN_TYPE]]


def get_set_element(s: Set):
    # see https://stackoverflow.com/questions/59825/how-to-retrieve-an-element-from-a-set-without-removing-it
    for x in s:
        return x
    raise ValueError("set is empty")

def ensure_list(to_encode: INPUT_SYMBOL_SEQUENCE_TYPE) -> List[TOKEN_TYPE]:
    if isinstance(to_encode, str):
        return list(to_encode.encode('utf-8'))
    elif isinstance(to_encode, bytes):
        return list(to_encode)
    elif isinstance(to_encode, list):
        return to_encode
    else:
        raise ValueError("Invalid input type")

def get_input_vocab(to_encode: INPUT_SYMBOL_SEQUENCE_TYPE) -> Set[TOKEN_TYPE]:
    if isinstance(to_encode, str):
        return set(to_encode.encode('utf-8'))
    elif isinstance(to_encode, bytes):
        return set(to_encode)
    else:
        raise ValueError("Invalid input type")

class Coder:
    def update_vocab(self, to_encode: INPUT_SYMBOL_SEQUENCE_TYPE) -> None:
        raise NotImplementedError("update_vocab not implemented")

    def encode(self, to_encode: INPUT_SYMBOL_SEQUENCE_TYPE, learn: bool=False) -> List[TOKEN_TYPE]:
        raise NotImplementedError("encode not implemented")
    
    def encode_one_token(self, to_encode: INPUT_SYMBOL_SEQUENCE_TYPE, learn: bool=False) -> Tuple[TOKEN_TYPE, INPUT_SYMBOL_SEQUENCE_TYPE]:
        raise NotImplementedError("encode_one_token not implemented")

    def decode(self, to_decode: List[TOKEN_TYPE]) -> INPUT_SYMBOL_SEQUENCE_TYPE:
        raise NotImplementedError("decode not implemented")



class LZCoder(Coder):
    encoded_vocab: Dict[TOKEN_TYPE, Tuple[TOKEN_TYPE]]
    token_map: pygtrie.Trie
    input_vocab: Set[int]
    vocab_size: int
    unused_tokens: Set[TOKEN_TYPE]


    def __init__(self, output_vocab_size: Optional[int]=None, input_vocab: Optional[Set[TOKEN_TYPE]]=None):
        self.input_vocab = set(input_vocab) if input_vocab is not None else set([])
        self.unused_tokens = set(range(output_vocab_size))

        assert len(self.input_vocab) <= output_vocab_size, "output vocab size is smaller than input vocab size!"

        self.token_map = pygtrie.Trie()
        self.encoded_vocab = {EMPTY_TOKEN: ()}
        self.token_map[()] = EMPTY_TOKEN
        for c in self.input_vocab:
            self._add_new_token((c,), get_set_element(self.unused_tokens))

        self.vocab_size = output_vocab_size + 1 # plus one because the empty token is -1

    def encode_one_token(self, to_encode: List[TOKEN_TYPE], learn: bool = False) -> Tuple[Tuple[TOKEN_TYPE], TOKEN_TYPE]:

        prefix, token = self._propose_next_token(to_encode, learn)
        if token not in self.encoded_vocab:
            self._add_new_token(prefix, token)

        return prefix, token

    def _propose_next_token(self, to_encode: List[TOKEN_TYPE], learn: bool = False) -> Tuple[TOKEN_TYPE]:
        prefix, token = self.token_map.longest_prefix(to_encode)
        if learn and len(prefix) < len(to_encode):
            if self.vocab_size is None or len(self.token_map) < self.vocab_size:
                # add new token that is prefix + next input symbol
                prefix = tuple(to_encode[:len(prefix)+1])
                token = get_set_element(self.unused_tokens)
        return prefix, token
    
    def _add_new_token(self, prefix: Tuple[TOKEN_TYPE], token: TOKEN_TYPE):
        self.encoded_vocab[token] = prefix
        self.token_map[prefix] = token
        self.unused_tokens.remove(token)
        assert len(self.token_map) == len(self.encoded_vocab)
    
    def update_vocab(self, to_encode: bytes):
        for c in to_encode:
            if c not in self.input_vocab:
                new_token = get_set_element(self.unused_tokens)
                self._add_new_token((c,), new_token)
                self.input_vocab.add(c)
            if len(self.token_map) >= self.vocab_size:
                raise ValueError("output vocab size is smaller than input vocab size!")

    def encode(self, to_encode: str, learn: bool=False):

        to_encode = ensure_list(to_encode)
            
        encoded = []

        while len(to_encode) > 0:
            prefix, token = self.encode_one_token(to_encode, learn)
            if len(prefix) == 0:
                if learn:
                    raise ValueError("could not match any tokens: the output dictionary is full!")
                else:
                    raise ValueError("could not match any tokens: did you mean to enable learning?")
            encoded.append(token)
            to_encode = to_encode[len(prefix):]
        
        return encoded

    def decode_one_token(self, to_decode: TOKEN_TYPE):
        return self.encoded_vocab[to_decode]


    def decode(self, to_decode: bytes):
        decoded = []
        for t in to_decode:
            decoded += list(self.encoded_vocab[t])
        
        return decoded



class HierachicalLZCoder(Coder):
    vocab_size: int
    coders: Dict[TOKEN_TYPE, LZCoder]

    def __init__(self, output_vocab_size: Optional[int]=None, input_vocab: Optional[Set[int]]=None):

        if input_vocab is not None:
            assert len(input_vocab) <= output_vocab_size, "output vocab size is smaller than input vocab size!"

        self.vocab_size = output_vocab_size
        self.coders = {
            EMPTY_TOKEN: LZCoder(output_vocab_size, input_vocab=input_vocab)
        }

    def update_vocab(self, to_encode: bytes):
        self.coders[EMPTY_TOKEN].update_vocab(to_encode)

    def encode_one_token(self, to_encode: INPUT_SYMBOL_SEQUENCE_TYPE, context: TOKEN_TYPE, learn: bool=False):
        if context not in self.coders:
            if learn:
                # make a new coder. We can let it learn its own input vocab.
                # TRICKY: it's possible that the new coder's token map will fill up
                # before it sees the entire input vocab. In this case it will output
                # EMPTY_TOKEN whenever it encounters an unknown input symbol.
                # This is ok, because we the coder for the EMPTY_TOKEN context is
                # initialized with the full input vocab, so it will be able to encode
                # this "unknown" symbol. This allows us to "skip" input symbols
                # that are very rare in this context, and might allow for compression
                # even if the input vocab size is equal to the encoding vocab size.
                # TODO: check if this is better than just initializing the new coder
                # with the full input vocab.
                self.coders[context] = LZCoder(self.vocab_size, input_vocab=set([]))
            else:
                raise ValueError("context not in coders")

        prefix, token = self.coders[context]._propose_next_token(to_encode, learn)

        if token in self.coders[context].encoded_vocab:
            return prefix, token
        
        if not learn:
            raise ValueError("trying to add new token, but learning is disabled!")
        
        # we want to add a new token for this context. But what should the symbol be?
        # the one proposed is just an arbirary unused symbol. We are going to be smarter:
        # we'll ask which symbol all of the *other* contexts would have chosen, and
        # then use the most commonly recommended untaken symbol.

        symbol_counts = {token: 0}

        assert token not in self.coders[context].encoded_vocab, "token is already in the encoded vocab!"
        assert token in self.coders[context].unused_tokens, "token is not in the unused tokens!"

        for other_context in self.coders:
            if other_context == context:
                continue
            _, other_token = self.coders[other_context]._propose_next_token(to_encode, learn)
            if other_token in self.coders[other_context].encoded_vocab:
                symbol_counts[other_token] = symbol_counts.get(other_token, 0) + 1
        
        # now we find the untaken symbol with the highest count
        # sort the symbols by count:
        sorted_symbols = sorted(symbol_counts, key=symbol_counts.get, reverse=True)
        for symbol in sorted_symbols:
            if symbol not in self.coders[context].encoded_vocab:
                token = symbol
                break
        
        self.coders[context]._add_new_token(prefix, token)

        return prefix, token
    
    def encode(self, to_encode: INPUT_SYMBOL_SEQUENCE_TYPE, learn: bool=False):
        context = EMPTY_TOKEN
        encoded = []

        to_encode = ensure_list(to_encode)

        while len(to_encode) > 0:
            prefix, token = self.encode_one_token(to_encode, context, learn)
            encoded.append(token)
            context = token
            to_encode = to_encode[len(prefix):]

        return encoded
    
    def decode(self, to_decode: bytes):
        context = EMPTY_TOKEN
        decoded = []
        for t in to_decode:
            decoded += list(self.coders[context].decode_one_token(t))
            context = t
        return decoded



        




__all__ = ["LZCoder", "HierachicalLZCoder"]