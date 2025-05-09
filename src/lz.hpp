#pragma once

#include <vector>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <cstdint>

namespace lz {

// Constants matching the Python implementation
constexpr int UNKNOWN_SYMBOL = 0;
constexpr int EMPTY_TOKEN = -1;

// Type definitions to match Python types
using TokenType = int;
using TokenSequence = std::vector<TokenType>;
using TokenTuple = std::vector<TokenType>; // Equivalent to Python tuple
using VocabSet = std::set<TokenType>;

// Forward declaration of Trie class (equivalent to pygtrie.Trie)
class Trie;

/**
 * Gets an element from a set without removing it
 * Equivalent to the get_set_element function in Python
 */
TokenType get_set_element(const VocabSet& s);

/**
 * Ensures input is converted to a list of tokens
 * Handles str, bytes, and list types from Python
 */
TokenSequence ensure_list(const std::string& to_encode);
TokenSequence ensure_list(const std::vector<uint8_t>& to_encode);
TokenSequence ensure_list(const TokenSequence& to_encode);

/**
 * Gets input vocabulary from input data
 * Equivalent to get_input_vocab in Python
 */
VocabSet get_input_vocab(const std::string& to_encode);
VocabSet get_input_vocab(const std::vector<uint8_t>& to_encode);

/**
 * Base Coder class - abstract interface
 * Equivalent to the Python Coder class
 */
class Coder {
public:
    virtual ~Coder() = default;

    // Update the vocabulary with new input
    virtual void update_vocab(const TokenSequence& to_encode) = 0;

    // Encode input to tokens
    virtual TokenSequence encode(const TokenSequence& to_encode, bool learn = false) = 0;

    // Encode a single token
    virtual std::tuple<TokenTuple, TokenType> encode_one_token(const TokenSequence& to_encode, bool learn = false) = 0;

    // Decode tokens back to original form
    virtual TokenSequence decode(const TokenSequence& to_decode) = 0;

    // Update the vocabulary with new input
    virtual void update_vocab(const std::string& to_encode) { update_vocab(ensure_list(to_encode)); }
    virtual void update_vocab(const std::vector<uint8_t>& to_encode) { update_vocab(ensure_list(to_encode)); }

    // Encode input to tokens
    virtual TokenSequence encode(const std::string& to_encode, bool learn = false) { return encode(ensure_list(to_encode), learn); }
    virtual TokenSequence encode(const std::vector<uint8_t>& to_encode, bool learn = false) { return encode(ensure_list(to_encode), learn); }
};

/**
 * Trie implementation (equivalent to pygtrie.Trie)
 */
class Trie {
public:
    Trie();
    ~Trie();

    // Copy constructor and assignment operator
    Trie(const Trie& other);
    Trie& operator=(const Trie& other);

    // Move constructor and assignment operator
    Trie(Trie&&) noexcept = default;
    Trie& operator=(Trie&&) noexcept = default;

    void insert(const TokenTuple& key, TokenType value);
    TokenType get(const TokenTuple& key) const;
    bool contains(const TokenTuple& key) const;
    std::tuple<TokenTuple, TokenType> longest_prefix(const TokenSequence& sequence) const;
    size_t size() const;

private:
    struct Node {
        TokenType value = EMPTY_TOKEN;
        bool is_end = false;
        std::map<TokenType, std::unique_ptr<Node>> children;

        // Copy constructor for deep copying
        Node(const Node& other)
            : value(other.value)
            , is_end(other.is_end) {
            for (const auto& [key, child] : other.children) {
                children[key] = std::make_unique<Node>(*child);
            }
        }

        // Default constructor
        Node() = default;
    };

    std::unique_ptr<Node> root;
};

/**
 * LZCoder implementation
 * Equivalent to the Python LZCoder class
 */
class LZCoder : public Coder {
public:
    // Constructor with optional output vocab size and input vocab
    LZCoder(int output_vocab_size = -1, VocabSet input_vocab = VocabSet());

    // Copy constructor and assignment operator
    LZCoder(const LZCoder&) = default;
    LZCoder& operator=(const LZCoder&) = default;

    // Move constructor and assignment operator
    LZCoder(LZCoder&&) noexcept = default;
    LZCoder& operator=(LZCoder&&) noexcept = default;

    // Update vocabulary with new input
    void update_vocab(const TokenSequence& to_encode) override;

    // Encode input to tokens
    TokenSequence encode(const TokenSequence& to_encode, bool learn = false) override;

    // Encode a single token
    std::tuple<TokenTuple, TokenType> encode_one_token(const TokenSequence& to_encode, bool learn = false) override;

    // Decode tokens back to original form
    TokenSequence decode(const TokenSequence& to_decode) override;

    // Decode a single token
    TokenTuple decode_one_token(TokenType to_decode);

    // Propose next token for encoding
    std::tuple<TokenTuple, TokenType> _propose_next_token(const TokenSequence& to_encode, bool learn = false) const;

    // Add a new token to the vocabulary
    void _add_new_token(const TokenTuple& prefix, TokenType token);

    // Accessors for Python bindings
    const VocabSet& get_unused_tokens() const { return unused_tokens; }
    const VocabSet& get_input_vocab() const { return input_vocab; }
    const std::map<TokenType, TokenTuple>& get_encoded_vocab() const { return encoded_vocab; }

private:
    int vocab_size;  // Changed from output_vocab_size to match Python
    VocabSet input_vocab;
    VocabSet unused_tokens;
    std::map<TokenType, TokenTuple> encoded_vocab;
    Trie token_map;
};

/**
 * HierarchicalLZCoder implementation
 * Equivalent to the Python HierarchicalLZCoder class
 */
class HierarchicalLZCoder : public Coder {
public:
    HierarchicalLZCoder(int output_vocab_size, VocabSet input_vocab);
    void update_vocab(const TokenSequence& to_encode) override;
    TokenSequence encode(const TokenSequence& to_encode, bool learn = false) override;
    std::tuple<TokenTuple, TokenType> encode_one_token(const TokenSequence& to_encode, bool learn = false) override;
    std::tuple<TokenTuple, TokenType> encode_one_token(const TokenSequence& to_encode, TokenType context, bool learn = false);
    TokenSequence decode(const TokenSequence& to_decode) override;

    // Accessor for Python bindings
    const std::map<TokenType, LZCoder>& get_coders() const { return coders; }

private:
    int vocab_size;
    std::map<TokenType, LZCoder> coders;
};

} // namespace lz 