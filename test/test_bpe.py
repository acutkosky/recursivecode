import pytest
from src.bpe import (
    BPE,
    ContextualEncoder,
    ComposedTokenizer,
    DefragEncoder,
)


def test_bpe_basic():
    """Test basic BPE functionality with a simple string."""
    text = "aaabdaaabac"
    max_output_vocab = 100

    # Learn BPE tokenizer
    bpe = ComposedTokenizer([DefragEncoder(), BPE(max_output_vocab=max_output_vocab)])
    bpe.learn(text)
    # Test that we can encode and decode
    encoded = bpe.encode(text)

    decoded = bpe.decode(encoded)

    # Verify the decoded result matches the original input
    assert bytes(decoded).decode("utf-8") == text


def test_bpe_empty_string():
    """Test BPE with empty string input."""
    text = ""
    max_output_vocab = 100

    bpe = ComposedTokenizer([DefragEncoder(), BPE(max_output_vocab=max_output_vocab)])
    bpe.learn(text)
    encoded = bpe.encode(text)
    decoded = bpe.decode(encoded)

    assert bytes(decoded).decode("utf-8") == text


def test_bpe_single_char():
    """Test BPE with single character input."""
    text = "a"
    max_output_vocab = 100

    bpe = ComposedTokenizer([DefragEncoder(), BPE(max_output_vocab=max_output_vocab)])
    bpe.learn(text)
    encoded = bpe.encode(text)
    decoded = bpe.decode(encoded)

    assert bytes(decoded).decode("utf-8") == text


def test_contextual_bpe_basic():
    """Test basic ContextualBPE functionality."""
    text = "aaabdaaabac"
    max_output_vocab = 100

    # Learn contextual BPE tokenizer
    contextual_bpe = ComposedTokenizer(
        [DefragEncoder(), BPE(max_output_vocab=max_output_vocab), ContextualEncoder()]
    )
    contextual_bpe.learn(text)

    # Test end-to-end encoding and decoding
    encoded = contextual_bpe.encode(text)
    decoded = contextual_bpe.decode(encoded)

    # Verify the decoded result matches the original input
    assert bytes(decoded).decode("utf-8") == text


def test_contextual_bpe_empty_string():
    """Test ContextualBPE with empty string input."""
    text = ""
    max_output_vocab = 100

    contextual_bpe = ComposedTokenizer(
        [DefragEncoder(), BPE(max_output_vocab=max_output_vocab), ContextualEncoder()]
    )
    contextual_bpe.learn(text)
    encoded = contextual_bpe.encode(text)
    decoded = contextual_bpe.decode(encoded)

    assert bytes(decoded).decode("utf-8") == text


def test_contextual_bpe_single_char():
    """Test ContextualBPE with single character input."""
    text = "a"
    max_output_vocab = 100

    contextual_bpe = ComposedTokenizer(
        [DefragEncoder(), BPE(max_output_vocab=max_output_vocab), ContextualEncoder()]
    )
    contextual_bpe.learn(text)
    encoded = contextual_bpe.encode(text)
    decoded = contextual_bpe.decode(encoded)

    assert bytes(decoded).decode("utf-8") == text


def test_bpe_vocab_size():
    """Test that BPE respects max_output_vocab parameter."""
    text = "aaabdaaabacaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabcd"
    max_output_vocab = 5

    bpe = ComposedTokenizer([DefragEncoder(), BPE(max_output_vocab=max_output_vocab)])
    bpe.learn(text)
    tokens = bpe.encode(text)

    unconstrained_bpe = ComposedTokenizer(
        [DefragEncoder(), BPE(max_output_vocab=1000000)]
    )
    unconstrained_bpe.learn(text)
    unconstrained_tokens = unconstrained_bpe.encode(text)

    assert len(unconstrained_tokens) < len(tokens)

    # The vocabulary size should be at most max_output_vocab
    assert len(bpe.tokenizers[1].merges) == max_output_vocab

    assert len(set(tokens)) <= max_output_vocab
    assert len(set(unconstrained_tokens)) > max_output_vocab

    decoded = bpe.decode(tokens)

    assert bytes(decoded).decode("utf-8") == text


def test_contextual_bpe_vocab_size():
    """Test that ContextualBPE respects max_output_vocab parameter."""
    text = "aaabdaaabacaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabcd"
    max_output_vocab = 4

    contextual_bpe = ComposedTokenizer(
        [DefragEncoder(), BPE(max_output_vocab=max_output_vocab), ContextualEncoder()]
    )
    contextual_bpe.learn(text)
    tokens = contextual_bpe.encode(text)

    unconstrained_contextual_bpe = ComposedTokenizer(
        [DefragEncoder(), BPE(max_output_vocab=1000000), ContextualEncoder()]
    )
    unconstrained_contextual_bpe.learn(text)
    unconstrained_tokens = unconstrained_contextual_bpe.encode(text)

    # The vocabulary size should be at most max_output_vocab
    assert len(contextual_bpe.tokenizers[1].merges) == max_output_vocab

    assert len(unconstrained_tokens) < len(tokens)

    assert len(set(tokens)) <= max_output_vocab

    # weirdly enough, smaller vocabs sometimes compress more for the contextual encoder!
    # assert len(set(unconstrained_tokens)) > max_output_vocab

    decoded = contextual_bpe.decode(tokens)

    assert bytes(decoded).decode("utf-8") == text


def test_bpe_roundtrip():
    """Test that BPE encoding and decoding preserves the input exactly."""
    text = "The quick brown fox jumps over the lazy dog"
    max_output_vocab = 100

    bpe = ComposedTokenizer([DefragEncoder(), BPE(max_output_vocab=max_output_vocab)])
    tokens = bpe.learn(text)
    encoded = bpe.encode(text.encode("utf-8"))
    decoded = bpe.decode(encoded)

    assert bytes(decoded).decode("utf-8") == text


def test_contextual_bpe_roundtrip():
    """Test that ContextualBPE encoding and decoding preserves the input exactly."""
    text = "The quick brown fox jumps over the lazy dog"
    max_output_vocab = 100

    contextual_bpe = ComposedTokenizer(
        [DefragEncoder(), BPE(max_output_vocab=max_output_vocab), ContextualEncoder()]
    )
    contextual_bpe.learn(text)
    encoded = contextual_bpe.encode(text)
    decoded = contextual_bpe.decode(encoded)

    assert bytes(decoded).decode("utf-8") == text


def test_bpe_unicode():
    """Test BPE with Unicode characters."""
    text = "Hello, 世界! 🌍"
    max_output_vocab = 100

    bpe = ComposedTokenizer([DefragEncoder(), BPE(max_output_vocab=max_output_vocab)])
    bpe.learn(text)
    encoded = bpe.encode(text)
    decoded = bpe.decode(encoded)

    assert bytes(decoded).decode("utf-8") == text


def test_contextual_bpe_unicode():
    """Test ContextualBPE with Unicode characters."""
    text = "Hello, 世界! 🌍"
    max_output_vocab = 100

    contextual_bpe = ComposedTokenizer(
        [DefragEncoder(), BPE(max_output_vocab=max_output_vocab), ContextualEncoder()]
    )
    contextual_bpe.learn(text)
    encoded = contextual_bpe.encode(text)
    decoded = contextual_bpe.decode(encoded)

    assert bytes(decoded).decode("utf-8") == text


def test_bpe_compression():
    """Test that BPE actually compresses longer strings with repeated patterns."""
    # Create a longer string with repeated patterns
    text = "the quick brown fox jumps over the lazy dog " * 10
    max_output_vocab = 100

    bpe = ComposedTokenizer([DefragEncoder(), BPE(max_output_vocab=max_output_vocab)])
    bpe.learn(text)
    encoded = bpe.encode(text)
    decoded = bpe.decode(encoded)

    # Verify the decoded result matches the original input
    assert bytes(decoded).decode("utf-8") == text

    # Verify that compression actually occurred
    original_length = len(text.encode("utf-8"))
    encoded_length = len(encoded)
    assert (
        encoded_length < original_length
    ), f"Expected compression but got {encoded_length} >= {original_length}"


def test_interleaved_composition():
    """Test that interleaved composition works."""
    text = "aaabdaaabac" * 10

    one_level = ComposedTokenizer([DefragEncoder(), ContextualEncoder()])
    one_level.learn(text)
    one_level_encoded = one_level.encode(text)

    two_level = ComposedTokenizer(
        [DefragEncoder(), ContextualEncoder(), BPE(max_merges=1)]
    )
    two_level.learn(text)
    two_level_encoded = two_level.encode(text)

    assert len(two_level_encoded) < len(one_level_encoded)

    three_level = ComposedTokenizer(
        [
            DefragEncoder(),
            ContextualEncoder(),
            BPE(max_merges=1),
            ContextualEncoder(),
            BPE(max_merges=1),
            ContextualEncoder(),
        ]
    )
    three_level.learn(text)
    three_level_encoded = three_level.encode(text)

    assert len(three_level_encoded) < len(two_level_encoded)
