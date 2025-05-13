#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/unordered_set.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/optional.h>

#include "lz.hpp"
#include "bpe.hpp"

namespace nb = nanobind;
using namespace nb::literals;

// LZ implementation bindings
namespace lz_bindings {
using namespace lz;

// Helper functions for type conversion
TokenSequence convert_to_token_sequence(const nb::object& obj) {
    TokenSequence result;
    
    // Handle different Python types
    if (nb::isinstance<nb::str>(obj)) {
        // Convert Python string to UTF-8 bytes and then to token sequence
        std::string str = nb::cast<std::string>(obj);
        return ensure_list(str);
    } 
    else if (nb::isinstance<nb::bytes>(obj)) {
        // Convert Python bytes to token sequence
        std::vector<uint8_t> bytes;
        for (auto item : obj) {
            bytes.push_back(nb::cast<uint8_t>(item));
        }
        return ensure_list(bytes);
    } 
    else if (nb::isinstance<nb::list>(obj) || nb::isinstance<nb::tuple>(obj)) {
        // Convert Python list/tuple to token sequence
        for (auto item : obj) {
            result.push_back(nb::cast<TokenType>(item));
        }
        return result;
    }
    
    throw std::runtime_error("Input must be str, bytes, or list of integers");
}

VocabSet convert_to_vocab_set(const nb::object& obj) {
    VocabSet result;
    
    // Handle different Python types
    if (nb::isinstance<nb::str>(obj)) {
        // Convert Python string to UTF-8 bytes and then to vocab set
        std::string str = nb::cast<std::string>(obj);
        return get_input_vocab(str);
    } 
    else if (nb::isinstance<nb::bytes>(obj)) {
        // Convert Python bytes to vocab set
        std::vector<uint8_t> bytes;
        for (auto item : obj) {
            bytes.push_back(nb::cast<uint8_t>(item));
        }
        return get_input_vocab(bytes);
    } 
    else if (nb::isinstance<nb::set>(obj) || nb::isinstance<nb::list>(obj) || nb::isinstance<nb::tuple>(obj)) {
        // Convert Python set/list/tuple to vocab set
        for (auto item : obj) {
            result.insert(nb::cast<TokenType>(item));
        }
        return result;
    }
    
    throw std::runtime_error("Input must be str, bytes, or set/list/tuple of integers");
}

// Wrapper functions for type conversion
TokenSequence encode_wrapper(LZCoder& self, const nb::object& to_encode, bool learn = true) {
    return self.encode(convert_to_token_sequence(to_encode), learn);
}

void update_vocab_wrapper(LZCoder& self, const nb::object& to_encode) {
    self.update_vocab(convert_to_token_sequence(to_encode));
}

std::tuple<TokenTuple, TokenType> encode_one_token_wrapper(LZCoder& self, const nb::object& to_encode, bool learn = true) {
    return self.encode_one_token(convert_to_token_sequence(to_encode), learn);
}

TokenSequence encode_wrapper_hierarchical(HierarchicalLZCoder& self, const nb::object& to_encode, bool learn = true) {
    return self.encode(convert_to_token_sequence(to_encode), learn);
}

void update_vocab_wrapper_hierarchical(HierarchicalLZCoder& self, const nb::object& to_encode) {
    self.update_vocab(convert_to_token_sequence(to_encode));
}

std::tuple<TokenTuple, TokenType> encode_one_token_wrapper_hierarchical(HierarchicalLZCoder& self, const nb::object& to_encode, TokenType context, bool learn = true) {
    return self.encode_one_token(convert_to_token_sequence(to_encode), context, learn);
}

} // namespace lz_bindings

// BPE implementation bindings
namespace bpe_bindings {
using namespace bpe;

// Helper functions for type conversion
TokenSequence convert_to_token_sequence(const nb::object& obj) {
    TokenSequence result;
    
    // Handle different Python types
    if (nb::isinstance<nb::str>(obj)) {
        // Convert Python string to UTF-8 bytes and then to token sequence
        std::string str = nb::cast<std::string>(obj);
        return ensure_list(str);
    } 
    else if (nb::isinstance<nb::bytes>(obj)) {
        // Convert Python bytes to token sequence
        std::vector<uint8_t> bytes;
        for (auto item : obj) {
            bytes.push_back(nb::cast<uint8_t>(item));
        }
        return ensure_list(bytes);
    } 
    else if (nb::isinstance<nb::list>(obj) || nb::isinstance<nb::tuple>(obj)) {
        // Convert Python list/tuple to token sequence
        for (auto item : obj) {
            result.push_back(nb::cast<TokenType>(item));
        }
        return result;
    }
    
    throw std::runtime_error("Input must be str, bytes, or list of integers");
}

// Convert Python set/list/tuple to vocab set
std::optional<VocabSet> convert_to_vocab_set_optional(const nb::object& obj) {
    if (obj.is_none()) {
        return std::nullopt;
    }
    
    VocabSet result;
    
    // Handle different Python types
    if (nb::isinstance<nb::str>(obj)) {
        // Convert Python string to UTF-8 bytes and then to vocab set
        std::string str = nb::cast<std::string>(obj);
        auto tokens = ensure_list(str);
        return VocabSet(tokens.begin(), tokens.end());
    } 
    else if (nb::isinstance<nb::bytes>(obj)) {
        // Convert Python bytes to vocab set
        std::vector<uint8_t> bytes;
        for (auto item : obj) {
            bytes.push_back(nb::cast<uint8_t>(item));
        }
        auto tokens = ensure_list(bytes);
        return VocabSet(tokens.begin(), tokens.end());
    } 
    else if (nb::isinstance<nb::set>(obj) || nb::isinstance<nb::list>(obj) || nb::isinstance<nb::tuple>(obj)) {
        // Convert Python set/list/tuple to vocab set
        for (auto item : obj) {
            result.insert(nb::cast<TokenType>(item));
        }
        return result;
    }
    
    throw std::runtime_error("Input vocab must be str, bytes, or set/list/tuple of integers");
}

// Wrapper functions for BPE
TokenSequence learn_wrapper(BPE& self, const nb::object& tokens, const nb::object& input_vocab = nb::none()) {
    TokenSequence tokens_seq = convert_to_token_sequence(tokens);
    std::optional<VocabSet> vocab_opt = convert_to_vocab_set_optional(input_vocab);
    
    self.learn(tokens_seq, vocab_opt);
    
    // Return the encoded tokens as a result
    return self.encode(tokens_seq);
}

TokenSequence encode_wrapper(BPE& self, const nb::object& tokens) {
    return self.encode(convert_to_token_sequence(tokens));
}

TokenSequence decode_wrapper(BPE& self, const nb::object& tokens) {
    return self.decode(convert_to_token_sequence(tokens));
}

} // namespace bpe_bindings

/**
 * Define a Python module named 'contok' that provides the same API as the original 'lz.py'
 * This module should be a drop-in replacement for the Python module.
 */
NB_MODULE(contok, m) {
    // Create the lz submodule
    auto lz_mod = m.def_submodule("lz", "LZ compression implementation");
    
    // Module docstring
    lz_mod.doc() = "C++ implementation of LZ compression with Python bindings";

    // Export constants
    lz_mod.attr("UNKNOWN_SYMBOL") = lz::UNKNOWN_SYMBOL;
    lz_mod.attr("EMPTY_TOKEN") = lz::EMPTY_TOKEN;

    // Export utility functions
    lz_mod.def("get_set_element", &lz::get_set_element, "s"_a);
    lz_mod.def("ensure_list", &lz_bindings::convert_to_token_sequence, "to_encode"_a);
    lz_mod.def("get_input_vocab", &lz_bindings::convert_to_vocab_set, "to_encode"_a);

    // Define LZCoder class
    nb::class_<lz::LZCoder>(lz_mod, "LZCoder")
        .def(nb::init<int, lz::VocabSet>(), "output_vocab_size"_a = -1, "input_vocab"_a = lz::VocabSet())
        .def("update_vocab", &lz_bindings::update_vocab_wrapper, "to_encode"_a)
        .def("encode", &lz_bindings::encode_wrapper, "to_encode"_a, "learn"_a = true)
        .def("encode_one_token", &lz_bindings::encode_one_token_wrapper, "to_encode"_a, "learn"_a = true)
        .def("decode", static_cast<lz::TokenSequence (lz::LZCoder::*)(const lz::TokenSequence&)>(&lz::LZCoder::decode), "to_decode"_a)
        .def("get_input_vocab", &lz::LZCoder::get_input_vocab)
        .def("get_encoded_vocab", &lz::LZCoder::get_encoded_vocab);

    // Define HierarchicalLZCoder class
    nb::class_<lz::HierarchicalLZCoder>(lz_mod, "HierarchicalLZCoder")
        .def(nb::init<int, lz::VocabSet>(), "output_vocab_size"_a = -1, "input_vocab"_a = lz::VocabSet())
        .def("update_vocab", &lz_bindings::update_vocab_wrapper_hierarchical, "to_encode"_a)
        .def("encode", &lz_bindings::encode_wrapper_hierarchical, "to_encode"_a, "learn"_a = true)
        .def("encode_one_token", &lz_bindings::encode_one_token_wrapper_hierarchical, "to_encode"_a, "context"_a, "learn"_a = true)
        .def("decode", static_cast<lz::TokenSequence (lz::HierarchicalLZCoder::*)(const lz::TokenSequence&)>(&lz::HierarchicalLZCoder::decode), "to_decode"_a)
        .def("get_coders", &lz::HierarchicalLZCoder::get_coders);
        
    // Create the bpe submodule
    auto bpe_submodule = m.def_submodule("bpe", "BPE implementation");
    
    // Module docstring
    bpe_submodule.doc() = "C++ implementation of BPE with Python bindings";
    
    // Export utility functions
    bpe_submodule.def("ensure_list", &bpe_bindings::convert_to_token_sequence, "to_encode"_a);
    
    // Define BPE class
    nb::class_<bpe::BPE>(bpe_submodule, "BPE")
        .def(nb::init<const std::vector<bpe::TokenPair>&, 
                      const std::unordered_map<bpe::TokenType, bpe::TokenTuple>&,
                      const std::unordered_map<bpe::TokenType, bpe::TokenType>&,
                      std::optional<int>,
                      std::optional<int>>(),
             "merges"_a = std::vector<bpe::TokenPair>(),
             "token_values"_a = std::unordered_map<bpe::TokenType, bpe::TokenTuple>(),
             "input_vocab"_a = std::unordered_map<bpe::TokenType, bpe::TokenType>(),
             "max_output_vocab"_a = std::nullopt,
             "max_merges"_a = std::nullopt)
        .def(nb::init<std::optional<int>, std::optional<int>>(),
             "max_output_vocab"_a = std::nullopt, "max_merges"_a = std::nullopt)
        .def("learn", &bpe_bindings::learn_wrapper, "tokens"_a, "input_vocab"_a = nb::none())
        .def("encode", &bpe_bindings::encode_wrapper, "tokens"_a)
        .def("decode", &bpe_bindings::decode_wrapper, "tokens"_a)
        .def_prop_ro("output_vocab", &bpe::BPE::get_output_vocab)
        .def_prop_ro("input_vocab", &bpe::BPE::get_input_vocab);
} 