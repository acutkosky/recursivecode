import pytest
import os
import sys
import time

# Add the source directory to the path to find our module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Import the C++ implementations
    from contok.bpe import BPE, DefragEncoder, ContextualEncoder, ensure_list
    
    # Import the Python implementation of ComposedTokenizer
    from src.bpe import ComposedTokenizer
except ImportError:
    pytest.skip("Required modules not found. Skipping workflow tests.", allow_module_level=True)


def test_single_sentence():
    """Test basic encoding and decoding of a single sentence."""
    text = "the quick brown fox jumps over the lazy dog"
    
    # Create tokenizer with all three encoders
    tokenizer = ComposedTokenizer([
        DefragEncoder(),
        BPE(max_output_vocab=50),
        ContextualEncoder()
    ])
    
    # Learn, encode, and decode
    tokenizer.learn(text)
    encoded = tokenizer.encode(text)
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


def test_repeated_text():
    """Test encoding and decoding of repeated text to check if patterns are recognized."""
    text = "hello world " * 10
    
    # Create tokenizer with all three encoders
    tokenizer = ComposedTokenizer([
        DefragEncoder(),
        BPE(max_output_vocab=30),
        ContextualEncoder()
    ])
    
    # Learn, encode, and decode
    tokenizer.learn(text)
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    # Convert to string for comparison
    decoded_str = bytes(decoded).decode('utf-8')
    
    # Print results
    print(f"Original: {text}")
    print(f"Encoded: {encoded}")
    print(f"Encoded length: {len(encoded)}")
    print(f"Original length: {len(text)}")
    print(f"Compression ratio: {len(text)/len(encoded):.2f}x")
    
    # Verify - for repeated text, we expect good compression
    assert decoded_str == text
    assert len(encoded) < len(text)


def test_complex_text():
    """Test encoding and decoding of more complex text with various characters."""
    text = """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
    Vestibulum nec odio eget nunc tincidunt tincidunt. 
    Cras sagittis, tortor ut lacinia commodo, turpis nisl feugiat urna, 
    in ultrices mi urna vel urna. Aenean euismod, diam sit amet volutpat dictum, 
    magna velit tincidunt magna, at gravida tellus eros ut risus.
    """
    
    # Create tokenizer with all three encoders
    tokenizer = ComposedTokenizer([
        DefragEncoder(),
        BPE(max_output_vocab=200),
        ContextualEncoder()
    ])
    
    # Learn, encode, and decode
    tokenizer.learn(text)
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    # Convert to string for comparison
    decoded_str = bytes(decoded).decode('utf-8')
    
    # Print results
    print(f"Original length: {len(text)}")
    print(f"Encoded length: {len(encoded)}")
    print(f"Compression ratio: {len(text)/len(encoded):.2f}x")
    
    # Verify
    assert decoded_str == text
    assert len(encoded) < len(text)


def test_benchmark():
    """Benchmark performance of the encoding and decoding processes."""
    # Generate a longer text for benchmarking
    text = "the quick brown fox jumps over the lazy dog. " * 100
    
    # Create tokenizers for comparison
    tokenizer1 = ComposedTokenizer([DefragEncoder(), BPE(max_output_vocab=100)])
    tokenizer2 = ComposedTokenizer([
        DefragEncoder(), 
        BPE(max_output_vocab=100),
        ContextualEncoder()
    ])
    
    # Benchmark learning
    start_time = time.time()
    tokenizer1.learn(text)
    bpe_learn_time = time.time() - start_time
    
    start_time = time.time()
    tokenizer2.learn(text)
    full_learn_time = time.time() - start_time
    
    # Benchmark encoding
    start_time = time.time()
    encoded1 = tokenizer1.encode(text)
    bpe_encode_time = time.time() - start_time
    
    start_time = time.time()
    encoded2 = tokenizer2.encode(text)
    full_encode_time = time.time() - start_time
    
    # Benchmark decoding
    start_time = time.time()
    decoded1 = tokenizer1.decode(encoded1)
    bpe_decode_time = time.time() - start_time
    
    start_time = time.time()
    decoded2 = tokenizer2.decode(encoded2)
    full_decode_time = time.time() - start_time
    
    # Convert to strings for comparison
    decoded_str1 = bytes(decoded1).decode('utf-8')
    decoded_str2 = bytes(decoded2).decode('utf-8')
    
    # Print results
    print(f"Original length: {len(text)}")
    print(f"BPE encoded length: {len(encoded1)}")
    print(f"Full encoded length: {len(encoded2)}")
    print(f"BPE compression ratio: {len(text)/len(encoded1):.2f}x")
    print(f"Full compression ratio: {len(text)/len(encoded2):.2f}x")
    
    print(f"\nBenchmark results:")
    print(f"BPE learn time: {bpe_learn_time:.4f}s")
    print(f"Full learn time: {full_learn_time:.4f}s")
    print(f"BPE encode time: {bpe_encode_time:.4f}s")
    print(f"Full encode time: {full_encode_time:.4f}s")
    print(f"BPE decode time: {bpe_decode_time:.4f}s")
    print(f"Full decode time: {full_decode_time:.4f}s")
    
    # Verify
    assert decoded_str1 == text
    assert decoded_str2 == text


if __name__ == "__main__":
    print("\n=== Testing Single Sentence ===")
    test_single_sentence()
    
    print("\n=== Testing Repeated Text ===")
    test_repeated_text()
    
    print("\n=== Testing Complex Text ===")
    test_complex_text()
    
    print("\n=== Running Benchmark ===")
    test_benchmark()
    
    print("\nAll tests passed!") 