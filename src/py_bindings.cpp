#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/unordered_set.h>
#include <nanobind/stl/unordered_map.h>

#include "lz.hpp"

namespace nb = nanobind;
using namespace nb::literals;
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

/**
 * Define a Python module named 'contok' that provides the same API as the original 'lz.py'
 * This module should be a drop-in replacement for the Python module.
 */
NB_MODULE(contok, m) {
    // Create the lz submodule
    auto lz = m.def_submodule("lz", "LZ compression implementation");
    
    // Module docstring
    lz.doc() = "C++ implementation of LZ compression with Python bindings";

    // Export constants
    lz.attr("UNKNOWN_SYMBOL") = lz::UNKNOWN_SYMBOL;
    lz.attr("EMPTY_TOKEN") = lz::EMPTY_TOKEN;

    // Export utility functions
    lz.def("get_set_element", &lz::get_set_element, "s"_a);
    lz.def("ensure_list", &convert_to_token_sequence, "to_encode"_a);
    lz.def("get_input_vocab", &convert_to_vocab_set, "to_encode"_a);

    // Define LZCoder class
    nb::class_<lz::LZCoder>(lz, "LZCoder")
        .def(nb::init<int, lz::VocabSet>(), "output_vocab_size"_a = -1, "input_vocab"_a = lz::VocabSet())
        .def("update_vocab", &update_vocab_wrapper, "to_encode"_a)
        .def("encode", &encode_wrapper, "to_encode"_a, "learn"_a = true)
        .def("encode_one_token", &encode_one_token_wrapper, "to_encode"_a, "learn"_a = true)
        .def("decode", static_cast<lz::TokenSequence (lz::LZCoder::*)(const lz::TokenSequence&)>(&lz::LZCoder::decode), "to_decode"_a)
        .def("get_input_vocab", &lz::LZCoder::get_input_vocab)
        .def("get_encoded_vocab", &lz::LZCoder::get_encoded_vocab);

    // Define HierarchicalLZCoder class
    nb::class_<lz::HierarchicalLZCoder>(lz, "HierarchicalLZCoder")
        .def(nb::init<int, lz::VocabSet>(), "output_vocab_size"_a = -1, "input_vocab"_a = lz::VocabSet())
        .def("update_vocab", &update_vocab_wrapper_hierarchical, "to_encode"_a)
        .def("encode", &encode_wrapper_hierarchical, "to_encode"_a, "learn"_a = true)
        .def("encode_one_token", &encode_one_token_wrapper_hierarchical, "to_encode"_a, "context"_a, "learn"_a = true)
        .def("decode", static_cast<lz::TokenSequence (lz::HierarchicalLZCoder::*)(const lz::TokenSequence&)>(&lz::HierarchicalLZCoder::decode), "to_decode"_a)
        .def("get_coders", &lz::HierarchicalLZCoder::get_coders);
} 