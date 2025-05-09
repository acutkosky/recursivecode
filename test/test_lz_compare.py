import pytest
import math
import importlib.util

# Import Python implementation
from src.lz import LZCoder as PyLZCoder, HierarchicalLZCoder as PyHierarchicalLZCoder, ensure_list as py_ensure_list, EMPTY_TOKEN as PY_EMPTY_TOKEN

# Check if the C++ module is available
if importlib.util.find_spec("hlz") is None:
    pytest.skip("hlz module not found. Skipping comparison tests.", allow_module_level=True)

# Import C++ implementation
from hlz import LZCoder as CppLZCoder, HierarchicalLZCoder as CppHierarchicalLZCoder, ensure_list as cpp_ensure_list, EMPTY_TOKEN as CPP_EMPTY_TOKEN

def test_ensure_list_equivalence():
    """Test that ensure_list behaves the same in both implementations"""
    test_cases = [
        "test",
        b"test",
        [116, 101, 115, 116],  # ASCII for "test"
        [1, 2, 3]
    ]
    
    for test_input in test_cases:
        py_result = py_ensure_list(test_input)
        cpp_result = cpp_ensure_list(test_input)
        assert py_result == cpp_result, f"ensure_list results differ for input {test_input}"
    
    # Test invalid input
    with pytest.raises(ValueError):
        py_ensure_list(123)
    with pytest.raises(RuntimeError):  # C++ throws runtime_error
        cpp_ensure_list(123)

def test_lz_coder_equivalence():
    """Test that LZCoder behaves the same in both implementations"""
    test_cases = [
        "hello world",
        "test string",
        "a" * 100,  # Repeated character
        "".join(chr(i) for i in range(128))  # All ASCII characters
    ]
    
    for test_str in test_cases:
        # Create coders with same parameters
        py_coder = PyLZCoder(output_vocab_size=256)
        cpp_coder = CppLZCoder(output_vocab_size=256)
        
        # Test encoding
        py_encoded = py_coder.encode(test_str, learn=True)
        cpp_encoded = cpp_coder.encode(test_str, learn=True)
        assert py_encoded == cpp_encoded, f"Encoding results differ for input {test_str}"
        
        # Test decoding
        py_decoded = py_coder.decode(py_encoded)
        cpp_decoded = cpp_coder.decode(cpp_encoded)
        assert py_decoded == cpp_decoded, f"Decoding results differ for input {test_str}"
        assert bytes(py_decoded).decode('utf-8') == test_str

def test_hierarchical_lz_coder_equivalence():
    """Test that HierarchicalLZCoder behaves the same in both implementations"""
    test_cases = [
        "hello world",
        "test string",
        "a" * 100,  # Repeated character
        "".join(chr(i) for i in range(128))  # All ASCII characters
    ]
    
    for test_str in test_cases:
        # Create coders with same parameters
        py_coder = PyHierarchicalLZCoder(output_vocab_size=256)
        cpp_coder = CppHierarchicalLZCoder(output_vocab_size=256)
        
        # Test encoding
        py_encoded = py_coder.encode(test_str, learn=True)
        cpp_encoded = cpp_coder.encode(test_str, learn=True)
        assert py_encoded == cpp_encoded, f"Encoding results differ for input {test_str}"
        
        # Test decoding
        py_decoded = py_coder.decode(py_encoded)
        cpp_decoded = cpp_coder.decode(cpp_encoded)
        assert py_decoded == cpp_decoded, f"Decoding results differ for input {test_str}"
        assert bytes(py_decoded).decode('utf-8') == test_str

def test_vocab_size_constraints():
    """Test that both implementations handle vocab size constraints the same way"""
    # Test with invalid vocab size
    try:
        PyLZCoder(output_vocab_size=2, input_vocab={1, 2, 3})
        assert False, "PyLZCoder should have raised an error"
    except Exception:
        pass

    try:
        CppLZCoder(output_vocab_size=2, input_vocab={1, 2, 3})
        assert False, "CppLZCoder should have raised an error"
    except Exception:
        pass

    # Test with valid but small vocab size
    py_coder = PyLZCoder(output_vocab_size=4)
    cpp_coder = CppLZCoder(output_vocab_size=4)

    test_str = "abcdaaaaaaa"
    py_encoded = py_coder.encode(test_str, learn=True)
    cpp_encoded = cpp_coder.encode(test_str, learn=True)
    assert py_encoded == cpp_encoded

    # Test with too small vocab size
    try:
        PyLZCoder(output_vocab_size=3).encode(test_str, learn=True)
        assert False, "PyLZCoder should have raised an error"
    except Exception:
        pass

    try:
        CppLZCoder(output_vocab_size=3).encode(test_str, learn=True)
        assert False, "CppLZCoder should have raised an error"
    except Exception:
        pass

def test_learning_behavior():
    """Test that both implementations learn new tokens in the same way"""
    test_str = "hello"
    
    # Create coders with same parameters
    py_coder = PyLZCoder(output_vocab_size=512, input_vocab=list(range(256)))
    cpp_coder = CppLZCoder(output_vocab_size=512, input_vocab=list(range(256)))
    
    # First encode without learning
    py_encoded_no_learn = py_coder.encode(test_str, learn=False)
    cpp_encoded_no_learn = cpp_coder.encode(test_str, learn=False)
    assert py_encoded_no_learn == cpp_encoded_no_learn
    
    # Then encode with learning
    py_encoded_with_learn = py_coder.encode(test_str, learn=True)
    cpp_encoded_with_learn = cpp_coder.encode(test_str, learn=True)
    assert py_encoded_with_learn == cpp_encoded_with_learn
    
    # Verify that learning made the output shorter
    assert len(py_encoded_no_learn) > len(py_encoded_with_learn)
    assert len(cpp_encoded_no_learn) > len(cpp_encoded_with_learn)

if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 