import pytest
import os
import sys

# Add the source directory to the path to find our module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Import the C++ implementations
    from contok.bpe import BPE, DefragEncoder, ContextualEncoder, ensure_list
    
    # Import the Python implementation of ComposedTokenizer for our tests
    from src.bpe import ComposedTokenizer
except ImportError:
    pytest.skip("Required modules not found. Skipping ComposedTokenizer tests.", allow_module_level=True)


def test_composed_defrag_bpe():
    """Test using C++ tokenizers with the Python ComposedTokenizer."""
    text = "aaabdaaabac"
    tokens = ensure_list(text)
    
    # Create C++ tokenizers
    defrag = DefragEncoder()
    bpe = BPE(max_output_vocab=10)
    
    # Use the Python ComposedTokenizer to compose them
    tokenizer = ComposedTokenizer([defrag, bpe])
    
    # Learn, encode, and decode
    tokenizer.learn(tokens)
    encoded = tokenizer.encode(tokens)
    decoded = tokenizer.decode(encoded)
    
    # Convert to string for comparison
    decoded_str = bytes(decoded).decode('utf-8')
    
    # Print debug info
    print("Original:", text)
    print("Encoded:", encoded)
    print("Decoded tokens:", decoded)
    print("Decoded string:", decoded_str)
    
    # Verify the decoded result matches the original input
    assert decoded_str == text
    
    # Verify that encoding reduces the sequence length (compression)
    assert len(encoded) < len(text)


def test_composed_three_encoders():
    """Test using all three C++ encoders in a ComposedTokenizer."""
    text = "the quick brown fox jumps over the lazy dog"
    tokens = ensure_list(text)
    
    # Create C++ tokenizers
    defrag = DefragEncoder()
    bpe = BPE(max_output_vocab=50)
    contextual = ContextualEncoder()
    
    # Use the Python ComposedTokenizer
    tokenizer = ComposedTokenizer([defrag, bpe, contextual])
    
    # Learn, encode, and decode
    tokenizer.learn(tokens)
    encoded = tokenizer.encode(tokens)
    decoded = tokenizer.decode(encoded)
    
    # Convert to string for comparison
    decoded_str = bytes(decoded).decode('utf-8')
    
    # Print results
    print(f"Original: {text}")
    print(f"Encoded: {encoded}")
    print(f"Encoded length: {len(encoded)}")
    print(f"Original length: {len(text)}")
    print(f"Compression ratio: {len(text)/len(encoded):.2f}x")
    
    # Verify
    assert decoded_str == text
    assert len(encoded) < len(text)


if __name__ == "__main__":
    test_composed_defrag_bpe()
    test_composed_three_encoders()
    print("All tests passed!") 