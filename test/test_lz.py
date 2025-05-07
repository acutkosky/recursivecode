import pytest
from src.lz import LZCoder, HierachicalLZCoder, ensure_list, EMPTY_TOKEN
import math

def test_basic_encode_decode():
    # Test with a simple string
    coder = LZCoder(output_vocab_size=256)
    test_str = "hello world"
    
    # Encode
    encoded = coder.encode(test_str, learn=True)
    assert isinstance(encoded, list)
    assert len(encoded) > 0

    print(encoded)
    
    # Decode
    decoded = coder.decode(encoded)
    assert isinstance(decoded, list)
    assert bytes(decoded).decode('utf-8') == test_str

def test_different_input_types():
    
    # Test with string
    str_input = "test"
    coder = LZCoder(output_vocab_size=256)
    str_encoded = coder.encode(str_input, learn=True)
    assert isinstance(str_encoded, list)
    
    # Test with bytes
    bytes_input = b"test"
    coder = LZCoder(output_vocab_size=256)
    bytes_encoded = coder.encode(bytes_input, learn=True)
    assert isinstance(bytes_encoded, list)
    assert bytes_encoded == str_encoded
    
    # Test with list
    list_input = [116, 101, 115, 116]  # ASCII for "test"
    coder = LZCoder(output_vocab_size=256)
    list_encoded = coder.encode(list_input, learn=True)
    assert isinstance(list_encoded, list)
    assert list_encoded == str_encoded

def test_learning_new_tokens():
    coder = LZCoder(output_vocab_size=512, input_vocab=list(range(256)))
    
    # First encode without learning
    test_str = "hello"
    encoded_no_learn = coder.encode(test_str, learn=False)

    decoded_no_learn = coder.decode(encoded_no_learn)
    assert bytes(decoded_no_learn).decode('utf-8') == test_str
    
    # Then encode with learning
    encoded_with_learn = coder.encode(test_str, learn=True)
    
    # The encoded output should be different because new tokens were learned
    assert len(encoded_no_learn) > len(encoded_with_learn)

def test_vocab_size_constraint():
    # Test that we can't create a coder with output vocab size smaller than input vocab
    with pytest.raises(AssertionError):
        LZCoder(output_vocab_size=2, input_vocab={1, 2, 3})

    coder = LZCoder(output_vocab_size=4)
    encoded = coder.encode("abcdaaaaaaa", learn=True)
    assert encoded == [0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0]

    with pytest.raises(ValueError):
        coder = LZCoder(output_vocab_size=3)
        encoded = coder.encode("abcdaaaaaaa", learn=True)
        assert encoded == [0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0]

def test_ensure_list():
    # Test string input
    str_input = "test"
    str_list = ensure_list(str_input)
    assert isinstance(str_list, list)
    assert str_list == [116, 101, 115, 116]  # ASCII values
    
    # Test bytes input
    bytes_input = b"test"
    bytes_list = ensure_list(bytes_input)
    assert isinstance(bytes_list, list)
    assert bytes_list == [116, 101, 115, 116]
    
    # Test list input
    list_input = [1, 2, 3]
    list_output = ensure_list(list_input)
    assert isinstance(list_output, list)
    assert list_output == [1, 2, 3]
    
    # Test invalid input
    with pytest.raises(ValueError):
        ensure_list(123)  # int is not a valid input type 

def test_hierarchical_basic_encode():
    # Test basic encoding with hierarchical coder
    coder = HierachicalLZCoder(output_vocab_size=256)
    test_str = "hello world"
    
    # Encode
    encoded = coder.encode(test_str, learn=True)
    assert isinstance(encoded, list)
    assert len(encoded) > 0

    decoded = coder.decode(encoded)
    assert bytes(decoded).decode('utf-8') == test_str

def test_hierarchical_context_learning():
    # Test that new contexts are learned
    coder = HierachicalLZCoder(output_vocab_size=256, input_vocab=set(range(256)))
    test_str = "hello world"
    
    # First encode without learning
    with pytest.raises(ValueError):
        encoded_no_learn = coder.encode(test_str, learn=False)
    print("moving on...")
    # Then encode with learning
    encoded_with_learn = coder.encode(test_str, learn=True)
    assert len(encoded_with_learn) > 0
    
    # Verify that new contexts were created
    assert len(coder.coders) > 1  # Should have more than just the EMPTY_TOKEN coder

def test_hierarchical_vocab_update():
    # Test vocabulary updating
    coder = HierachicalLZCoder(output_vocab_size=256)
    test_bytes = b"hello"
    
    # Update vocab
    coder.update_vocab(test_bytes)
    
    # Verify that the input vocab was updated
    assert all(ord(c) in coder.coders[EMPTY_TOKEN].input_vocab for c in "hello")

def test_hierarchical_vocab_size_constraint():
    # Test that we can't create a coder with output vocab size smaller than input vocab
    with pytest.raises(AssertionError):
        HierachicalLZCoder(output_vocab_size=2, input_vocab={1, 2, 3})

def test_hierarchical_encode_one_token():
    # Test single token encoding with context
    coder = HierachicalLZCoder(output_vocab_size=256)
    test_bytes = b"hello"
    
    # First token should use EMPTY_TOKEN context
    prefix, token = coder.encode_one_token(test_bytes, EMPTY_TOKEN, learn=True)
    assert isinstance(token, int)
    assert len(prefix) < len(test_bytes)
    
    # Using the same context again should work
    prefix2, token2 = coder.encode_one_token(test_bytes[len(prefix):], token, learn=True)
    assert isinstance(token2, int)
    assert len(prefix2) < len(test_bytes)
    
    # Using an unknown context without learning should fail
    with pytest.raises(ValueError):
        coder.encode_one_token(test_bytes, 999, learn=False) 

def test_hierarchical_encode_repeated_compression():
    # technically the output vocab size is one more than input because of the empty token.

    with open('test/compression_test_text.txt', 'r') as f:
        to_encode = f.read().strip()

    to_encode_list = ensure_list(to_encode)
    input_vocab = set(to_encode_list)
    coder = HierachicalLZCoder(output_vocab_size=len(input_vocab), input_vocab=input_vocab)

    encoded = coder.encode(to_encode_list, learn=True)

    encoded_length = len(encoded) * math.log(len(input_vocab)+1, 2)

    encoded_vocab = set(encoded)

    second_coder = HierachicalLZCoder(output_vocab_size=2*len(encoded_vocab), input_vocab=encoded_vocab)

    second_encoded = second_coder.encode(encoded, learn=True)
    second_encoded_length = len(second_encoded) * math.log(2*len(encoded_vocab)+1, 2)

    second_decoded = second_coder.decode(second_encoded)
    assert second_decoded == encoded


    double_vocab_size = HierachicalLZCoder(output_vocab_size=2*len(encoded_vocab), input_vocab=input_vocab)
    double_vocab_size_encoded = double_vocab_size.encode(to_encode_list, learn=True)

    double_vocab_size_encoded_length = len(double_vocab_size_encoded) * math.log(2*len(input_vocab)+1, 2)

    decoded = coder.decode(encoded)
    assert bytes(decoded).decode('utf-8') == to_encode

    lz_coder = LZCoder(output_vocab_size=10*len(input_vocab)+1, input_vocab=input_vocab)
    lz_encoded = lz_coder.encode(to_encode, learn=True)

    lz_encoded_length = len(lz_encoded) * math.log(10*len(input_vocab)+1, 2)

    # print("len(lz_encoded): ", lz_encoded_length)
    # print("len(encoded): ", encoded_length)
    # print("len(second_encoded): ", second_encoded_length)
    # print("len(double_vocab_size_encoded): ", double_vocab_size_encoded_length)
    # print("len(to_encode_list): ", len(to_encode_list) * math.log(len(input_vocab), 2))

    # print("len(encoded_vocab): ", len(encoded_vocab))
    # print("len(input_vocab): ", len(input_vocab))

    # print(encoded)


    assert len(encoded) * math.log(len(input_vocab), 2) < len(to_encode_list) * math.log(len(input_vocab), 2)

    # we beat the basic lz coder, using a smaller vocab size
    assert lz_encoded_length > encoded_length
    assert second_encoded_length < encoded_length
    assert second_encoded_length < double_vocab_size_encoded_length


