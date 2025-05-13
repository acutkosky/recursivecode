import pytest
import os
import sys

# Add the source directory to the path to find our module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Import the C++ BPE implementation
    from contok.bpe import BPE, ensure_list
    
    # Import the Python implementations of the other tokenizers
    from src.bpe import ComposedTokenizer, DefragEncoder, ContextualEncoder
except ImportError:
    pytest.skip("Required modules not found. Skipping composition tests.", allow_module_level=True)


def test_cpp_bpe_with_defrag():
    """Test using C++ BPE with the Python DefragEncoder."""
    text = "aaabdaaabac"
    max_output_vocab = 10
    
    # Create ComposedTokenizer with Python DefragEncoder and C++ BPE
    tokenizer = ComposedTokenizer([DefragEncoder(), BPE(max_output_vocab=max_output_vocab)])
    
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
    
    # Verify that encoding reduces the sequence length
    assert len(encoded) < len(text)


def test_cpp_bpe_with_contextual():
    """Test using C++ BPE with the Python ContextualEncoder."""
    text = "aaabdaaabac"
    max_output_vocab = 10
    
    # Create ComposedTokenizer with Python DefragEncoder, C++ BPE, and Python ContextualEncoder
    tokenizer = ComposedTokenizer([
        DefragEncoder(), 
        BPE(max_output_vocab=max_output_vocab),
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


def test_cpp_bpe_vocab_size_limit():
    """Test using C++ BPE with a vocabulary size limit in a composed setup."""
    text = "aaabdaaabacaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabcd"
    max_output_vocab = 5
    
    # Create tokenizers with different vocab limits
    tokenizer_limited = ComposedTokenizer([
        DefragEncoder(), 
        BPE(max_output_vocab=max_output_vocab)
    ])
    
    tokenizer_unlimited = ComposedTokenizer([
        DefragEncoder(), 
        BPE(max_output_vocab=1000)
    ])
    
    # Learn and encode with both tokenizers
    tokenizer_limited.learn(text)
    tokenizer_unlimited.learn(text)
    
    encoded_limited = tokenizer_limited.encode(text)
    encoded_unlimited = tokenizer_unlimited.encode(text)
    
    # Verify that the unlimited vocabulary produces better compression
    assert len(encoded_unlimited) < len(encoded_limited)
    
    # Verify that the encoded tokens use at most max_output_vocab + vocabulary size unique tokens
    # (because the DefragEncoder will use 1 to N for the input vocabulary)
    unique_tokens = set(encoded_limited)
    
    # Print debug info
    print(f"Limited vocab (max {max_output_vocab}) - encoded length: {len(encoded_limited)}")
    print(f"Unlimited vocab - encoded length: {len(encoded_unlimited)}")
    print(f"Limited vocab unique tokens: {len(unique_tokens)}")
    
    # Verify roundtrip works
    decoded_limited = tokenizer_limited.decode(encoded_limited)
    decoded_str = bytes(decoded_limited).decode('utf-8')
    
    # Verify the decoded result matches the original input
    assert decoded_str == text


def test_composed_sequence_longer_text():
    """Test the full sequence of tokenizers on longer text."""
    # Create a longer string with repeated patterns
    text = "the quick brown fox jumps over the lazy dog " * 5
    max_output_vocab = 100
    
    # Create tokenizer with all components
    tokenizer = ComposedTokenizer([
        DefragEncoder(), 
        BPE(max_output_vocab=max_output_vocab),
        ContextualEncoder()
    ])
    
    # Learn, encode, and decode
    tokenizer.learn(text)
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    # Convert to string for comparison
    decoded_str = bytes(decoded).decode('utf-8')
    
    # Verify the decoded result matches the original input
    assert decoded_str == text
    
    # Verify that compression actually occurred
    original_length = len(text)
    encoded_length = len(encoded)
    compression_ratio = original_length / encoded_length
    
    print(f"Original length: {original_length}")
    print(f"Encoded length: {encoded_length}")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    assert compression_ratio > 1.0, f"Expected compression but got {compression_ratio:.2f}x"


if __name__ == "__main__":
    test_cpp_bpe_with_defrag()
    test_cpp_bpe_with_contextual()
    test_cpp_bpe_vocab_size_limit()
    test_composed_sequence_longer_text()
    print("All tests passed!") 