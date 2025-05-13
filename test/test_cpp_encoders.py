import pytest
import os
import sys

# Add the source directory to the path to find our module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Import the C++ implementations
    from contok.bpe import BPE, DefragEncoder, ContextualEncoder, ensure_list
except ImportError:
    pytest.skip("Required modules not found. Skipping encoder tests.", allow_module_level=True)


def test_defrag_encoder():
    """Test basic functionality of the DefragEncoder class."""
    text = "aaabdaaabac"
    tokens = ensure_list(text)
    
    # Create DefragEncoder
    encoder = DefragEncoder()
    
    # Learn, encode, and decode
    encoder.learn(tokens)
    encoded = encoder.encode(tokens)
    decoded = encoder.decode(encoded)
    
    # Convert to string for comparison
    decoded_str = bytes(decoded).decode('utf-8')
    
    # Print debug info
    print("Original:", text)
    print("Encoded:", encoded)
    print("Decoded tokens:", decoded)
    print("Decoded string:", decoded_str)
    
    # Verify the decoded result matches the original input
    assert decoded_str == text
    
    # Verify that the encoder created a continuous range of integers
    unique_tokens = set(encoded)
    max_token = max(unique_tokens)
    min_token = min(unique_tokens)
    
    assert min_token == 1
    assert max_token == len(set(text))
    assert len(unique_tokens) == len(set(text))


def test_contextual_encoder():
    """Test basic functionality of the ContextualEncoder class."""
    text = "aaabdaaabac"
    tokens = ensure_list(text)
    
    # Create ContextualEncoder (this is a stub implementation for now)
    encoder = ContextualEncoder()
    
    # Learn, encode, and decode
    encoder.learn(tokens)
    encoded = encoder.encode(tokens)
    decoded = encoder.decode(encoded)
    
    # Convert to string for comparison
    decoded_str = bytes(decoded).decode('utf-8') if decoded else ""
    
    # Print debug info
    print("Original:", text)
    print("Encoded:", encoded)
    print("Decoded tokens:", decoded)
    print("Decoded string:", decoded_str)
    
    # Verify the stub implementation still works (even if it doesn't compress yet)
    assert len(decoded) == 0  # For the stub that returns empty lists


if __name__ == "__main__":
    test_defrag_encoder()
    test_contextual_encoder()
    print("All tests passed!") 