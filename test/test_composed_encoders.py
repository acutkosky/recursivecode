import pytest
import os
import sys

# Add the source directory to the path to find our module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Import the C++ implementations
    from contok.bpe import BPE, DefragEncoder, ContextualEncoder, ensure_list
    
    # Import the Python implementation of ComposedTokenizer
    from src.bpe import ComposedTokenizer
except ImportError:
    pytest.skip("Required modules not found. Skipping encoder tests.", allow_module_level=True)


def test_composed_defrag_bpe():
    """Test using C++ DefragEncoder with C++ BPE in a ComposedTokenizer."""
    text = "aaabdaaabac"
    
    # Create a ComposedTokenizer with C++ DefragEncoder and C++ BPE
    tokenizer = ComposedTokenizer([
        DefragEncoder(),
        BPE(max_output_vocab=10)
    ])
    
    # Learn, encode, and decode
    tokenizer.learn(text)
    encoded = tokenizer.encode(text)
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


def test_composed_defrag_bpe_contextual():
    """Test using all three C++ encoders in a ComposedTokenizer."""
    text = "aaabdaaabac"
    
    # Create a ComposedTokenizer with all three C++ encoders
    tokenizer = ComposedTokenizer([
        DefragEncoder(),
        BPE(max_output_vocab=10),
        ContextualEncoder()
    ])
    
    # Learn, encode, and decode
    tokenizer.learn(text)
    encoded = tokenizer.encode(text)
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


def test_mixed_encoders():
    """Test using a mix of C++ and Python encoders in a ComposedTokenizer."""
    text = "aaabdaaabac" * 3  # Repeat to create more patterns
    
    # Import Python implementation of BPE
    from src.bpe import BPE as PyBPE
    
    # Create a ComposedTokenizer with C++ DefragEncoder and Python BPE
    tokenizer1 = ComposedTokenizer([
        DefragEncoder(),
        PyBPE(max_output_vocab=10)
    ])
    
    # Create a ComposedTokenizer with Python BPE and C++ ContextualEncoder
    tokenizer2 = ComposedTokenizer([
        DefragEncoder(),
        BPE(max_output_vocab=10),
        ContextualEncoder()
    ])
    
    # Learn and encode with both tokenizers
    tokenizer1.learn(text)
    tokenizer2.learn(text)
    
    encoded1 = tokenizer1.encode(text)
    encoded2 = tokenizer2.encode(text)
    
    # Decode and verify
    decoded1 = tokenizer1.decode(encoded1)
    decoded2 = tokenizer2.decode(encoded2)
    
    decoded_str1 = bytes(decoded1).decode('utf-8')
    decoded_str2 = bytes(decoded2).decode('utf-8')
    
    # Print debug info
    print("Original:", text)
    print("Encoded 1 (Cpp Defrag + Py BPE):", encoded1)
    print("Encoded 2 (Cpp Defrag + Cpp BPE + Cpp Contextual):", encoded2)
    
    # Verify decoded results match the original input
    assert decoded_str1 == text
    assert decoded_str2 == text


def test_long_text_compression():
    """Test compression of longer text with various combinations."""
    # Create a longer string with repeated patterns
    text = "the quick brown fox jumps over the lazy dog " * 5
    
    # Create tokenizers with different configurations
    tokenizer1 = ComposedTokenizer([DefragEncoder(), BPE(max_output_vocab=100)])
    tokenizer2 = ComposedTokenizer([
        DefragEncoder(), 
        BPE(max_output_vocab=100),
        ContextualEncoder()
    ])
    
    # Learn and encode
    tokenizer1.learn(text)
    tokenizer2.learn(text)
    
    encoded1 = tokenizer1.encode(text)
    encoded2 = tokenizer2.encode(text)
    
    # Decode and verify
    decoded1 = tokenizer1.decode(encoded1)
    decoded2 = tokenizer2.decode(encoded2)
    
    decoded_str1 = bytes(decoded1).decode('utf-8')
    decoded_str2 = bytes(decoded2).decode('utf-8')
    
    # Print compression stats
    print(f"Original length: {len(text)}")
    print(f"Encoded length (Defrag+BPE): {len(encoded1)}")
    print(f"Encoded length (Defrag+BPE+Contextual): {len(encoded2)}")
    print(f"Compression ratio 1: {len(text)/len(encoded1):.2f}x")
    print(f"Compression ratio 2: {len(text)/len(encoded2):.2f}x")
    
    # Verify the decoded results match the original input
    assert decoded_str1 == text
    assert decoded_str2 == text
    
    # Verify that compression occurred
    assert len(encoded1) < len(text)
    assert len(encoded2) < len(text)


if __name__ == "__main__":
    test_composed_defrag_bpe()
    test_composed_defrag_bpe_contextual()
    test_mixed_encoders()
    test_long_text_compression()
    print("All tests passed!") 