import pytest
import os
import sys

# Add the source directory to the path to find our module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from contok.bpe import BPE, ensure_list
except ImportError:
    pytest.skip("contok.bpe module not found. Skipping BPE implementation tests.", allow_module_level=True)


def test_bpe_basic():
    """Test basic BPE functionality with a simple string."""
    text = "aaabdaaabac"
    max_output_vocab = 10
    
    # Create BPE tokenizer
    bpe = BPE(max_output_vocab=max_output_vocab)
    
    # Test learning, encoding, and decoding
    tokens = ensure_list(text)
    bpe.learn(tokens)
    encoded = bpe.encode(tokens)
    decoded = bpe.decode(encoded)
    
    # Verify the decoded result matches the original input
    print("Original:", tokens)
    print("Encoded:", encoded)
    print("Decoded:", decoded)
    
    # Since we're working with ASCII, this simple comparison should work
    assert tokens == decoded
    
    # Verify that encoding reduces the sequence length
    assert len(encoded) < len(tokens)


def test_bpe_empty_string():
    """Test BPE with empty string input."""
    text = ""
    max_output_vocab = 10
    
    # Create BPE tokenizer
    bpe = BPE(max_output_vocab=max_output_vocab)
    
    # Test learning, encoding, and decoding
    tokens = ensure_list(text)
    bpe.learn(tokens)
    encoded = bpe.encode(tokens)
    decoded = bpe.decode(encoded)
    
    # Verify the decoded result matches the original input
    assert tokens == decoded


def test_bpe_single_char():
    """Test BPE with single character input."""
    text = "a"
    max_output_vocab = 10
    
    # Create BPE tokenizer
    bpe = BPE(max_output_vocab=max_output_vocab)
    
    # Test learning, encoding, and decoding
    tokens = ensure_list(text)
    bpe.learn(tokens)
    encoded = bpe.encode(tokens)
    decoded = bpe.decode(encoded)
    
    # Verify the decoded result matches the original input
    assert tokens == decoded


if __name__ == "__main__":
    test_bpe_basic()
    test_bpe_empty_string()
    test_bpe_single_char()
    print("All tests passed!") 