#include "lz.hpp"
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <cassert>

namespace lz {

//
// Utility functions implementation
//

/**
 * Gets an element from a set without removing it
 * Equivalent to get_set_element from Python
 */
TokenType get_set_element(const VocabSet& s) {
    if (s.empty()) {
        throw std::runtime_error("Cannot get element from empty set");
    }
    return *s.begin();
}

/**
 * Ensures input is converted to a list of tokens (from string)
 * Equivalent to ensure_list for string input in Python 
 */
TokenSequence ensure_list(const std::string& to_encode) {
    TokenSequence result;
    for (char c : to_encode) {
        result.push_back(static_cast<uint8_t>(c));
    }
    return result;
}

/**
 * Ensures input is converted to a list of tokens (from bytes)
 * Equivalent to ensure_list for bytes input in Python
 */
TokenSequence ensure_list(const std::vector<uint8_t>& to_encode) {
    TokenSequence result;
    for (uint8_t c : to_encode) {
        result.push_back(c);
    }
    return result;
}

/**
 * Ensures input is converted to a list of tokens (from token sequence)
 * Equivalent to ensure_list for list input in Python
 */
TokenSequence ensure_list(const TokenSequence& to_encode) {
    return to_encode;
}

/**
 * Gets input vocabulary from string input data
 * Equivalent to get_input_vocab for string in Python
 */
VocabSet get_input_vocab(const std::string& to_encode) {
    VocabSet result;
    for (char c : to_encode) {
        result.insert(static_cast<uint8_t>(c));
    }
    return result;
}

/**
 * Gets input vocabulary from bytes input data
 * Equivalent to get_input_vocab for bytes in Python
 */
VocabSet get_input_vocab(const std::vector<uint8_t>& to_encode) {
    VocabSet result;
    for (uint8_t c : to_encode) {
        result.insert(c);
    }
    return result;
}

//
// Trie implementation 
//

/**
 * Trie constructor
 */
Trie::Trie() : root(std::make_unique<Node>()) {}

/**
 * Trie destructor
 */
Trie::~Trie() = default;

/**
 * Trie copy constructor
 */
Trie::Trie(const Trie& other) {
    if (other.root) {
        root = std::make_unique<Node>(*other.root);
    }
}

/**
 * Trie assignment operator
 */
Trie& Trie::operator=(const Trie& other) {
    if (this != &other) {
        if (other.root) {
            root = std::make_unique<Node>(*other.root);
        } else {
            root.reset();
        }
    }
    return *this;
}

/**
 * Insert a key-value pair into the trie
 */
void Trie::insert(const TokenTuple& key, TokenType value) {
    Node* current = root.get();
    for (TokenType token : key) {
        if (current->children.find(token) == current->children.end()) {
            current->children[token] = std::make_unique<Node>();
        }
        current = current->children[token].get();
    }
    current->value = value;
    current->is_end = true;
}

/**
 * Get value associated with key
 */
TokenType Trie::get(const TokenTuple& key) const {
    const Node* current = root.get();
    for (TokenType token : key) {
        auto it = current->children.find(token);
        if (it == current->children.end()) {
            throw std::runtime_error("key not found");
        }
        current = it->second.get();
    }
    if (!current->is_end) {
        throw std::runtime_error("key not found");
    }
    return current->value;
}

/**
 * Check if key exists in the trie
 */
bool Trie::contains(const TokenTuple& key) const {
    try {
        get(key);
        return true;
    } catch (const std::runtime_error&) {
        return false;
    }
}

/**
 * Find longest prefix of sequence that exists in the trie
 * Returns tuple of (prefix, token)
 */
std::tuple<TokenTuple, TokenType> Trie::longest_prefix(const TokenSequence& sequence) const {
    const Node* current = root.get();
    TokenTuple prefix;
    TokenType value = EMPTY_TOKEN;

    for (TokenType token : sequence) {
        auto it = current->children.find(token);
        if (it == current->children.end()) {
            break;
        }
        prefix.push_back(token);
        current = it->second.get();
        if (current->is_end) {
            value = current->value;
        }
    }

    return {prefix, value};
}

/**
 * Get size of the trie (number of entries)
 */
size_t Trie::size() const {
    size_t count = 0;
    std::function<void(const Node*)> dfs = [&](const Node* node) {
        if (node->is_end) {
            ++count;
        }
        for (const auto& [_, child] : node->children) {
            dfs(child.get());
        }
    };
    dfs(root.get());
    return count;
}

//
// LZCoder implementation
//

/**
 * LZCoder constructor
 */
LZCoder::LZCoder(int output_vocab_size, VocabSet input_vocab)
    : vocab_size(-1)  // Initialize to -1 (None in Python)
    , input_vocab(std::move(input_vocab)) {
    
    // Initialize empty token
    encoded_vocab[EMPTY_TOKEN] = TokenTuple();
    token_map.insert(TokenTuple(), EMPTY_TOKEN);

    if (output_vocab_size > 0) {
        // Check if output vocab size is sufficient for input vocab
        if (static_cast<int>(this->input_vocab.size()) > output_vocab_size) {
            throw std::runtime_error("AssertionError: len(self.input_vocab) <= output_vocab_size");
        }

        // Initialize unused tokens from 0 to output_vocab_size-1
        for (int i = 0; i < output_vocab_size; ++i) {
            unused_tokens.insert(i);
        }

        // Initialize input vocab tokens
        for (TokenType c : this->input_vocab) {
            TokenType token = *unused_tokens.begin();  // Always use the smallest available token
            _add_new_token({c}, token);
        }

        // Set vocab_size to output_vocab_size + 1 to account for empty token
        vocab_size = output_vocab_size + 1;
    }
}

/**
 * Update vocabulary with new input (string)
 * Equivalent to LZCoder.update_vocab in Python
 */
void LZCoder::update_vocab(const TokenSequence& to_encode) {
    for (TokenType c : to_encode) {
        if (input_vocab.find(c) == input_vocab.end()) {
            if (unused_tokens.empty()) {
                throw std::runtime_error("no unused tokens available");
            }
            TokenType new_token = *unused_tokens.begin();  // Always use the smallest available token
            _add_new_token({c}, new_token);
            input_vocab.insert(c);
            if (vocab_size > 0 && static_cast<int>(token_map.size()) >= vocab_size) {
                throw std::runtime_error("output vocab size is smaller than input vocab size!");
            }
        }
    }
}

/**
 * Encode string input to tokens
 * Equivalent to LZCoder.encode for string in Python
 */
TokenSequence LZCoder::encode(const TokenSequence& to_encode, bool learn) {
    TokenSequence encoded;
    TokenSequence remaining = to_encode;

    while (!remaining.empty()) {
        auto [prefix, token] = encode_one_token(remaining, learn);
        if (prefix.empty()) {
            if (learn) {
                throw std::runtime_error("could not match any tokens: the output dictionary is full!");
            } else {
                throw std::runtime_error("could not match any tokens: did you mean to enable learning?");
            }
        }
        encoded.push_back(token);
        remaining.erase(remaining.begin(), remaining.begin() + prefix.size());
    }

    return encoded;
}

/**
 * Encode a single token
 * Equivalent to LZCoder.encode_one_token in Python
 */
std::tuple<TokenTuple, TokenType> LZCoder::encode_one_token(const TokenSequence& to_encode, bool learn) {
    auto [prefix, token] = _propose_next_token(to_encode, learn);
    if (encoded_vocab.find(token) == encoded_vocab.end()) {
        if (!learn) {
            throw std::runtime_error("could not match any tokens: did you mean to enable learning?");
        }
        if (vocab_size > 0 && static_cast<int>(token_map.size()) >= vocab_size) {
            throw std::runtime_error("could not match any tokens: the output dictionary is full!");
        }
        if (unused_tokens.empty()) {
            throw std::runtime_error("no unused tokens available");
        }
        TokenType new_token = *unused_tokens.begin();  // Always use the smallest available token
        _add_new_token(prefix, new_token);
        return {prefix, new_token};
    }
    return {prefix, token};
}

/**
 * Decode tokens back to original form
 * Equivalent to LZCoder.decode in Python
 */
TokenSequence LZCoder::decode(const TokenSequence& to_decode) {
    TokenSequence decoded;
    for (TokenType token : to_decode) {
        auto token_decoded = decode_one_token(token);
        decoded.insert(decoded.end(), token_decoded.begin(), token_decoded.end());
    }
    return decoded;
}

/**
 * Decode a single token
 * Equivalent to LZCoder.decode_one_token in Python
 */
TokenTuple LZCoder::decode_one_token(TokenType to_decode) {
    auto it = encoded_vocab.find(to_decode);
    if (it == encoded_vocab.end()) {
        throw std::runtime_error("token not found in encoded vocab");
    }
    return it->second;
}

/**
 * Propose next token for encoding
 * Equivalent to LZCoder._propose_next_token in Python
 */
std::tuple<TokenTuple, TokenType> LZCoder::_propose_next_token(const TokenSequence& to_encode, bool learn) const {
    auto [prefix, token] = token_map.longest_prefix(to_encode);
    
    if (learn && prefix.size() < to_encode.size()) {
        if (vocab_size < 0 || static_cast<int>(token_map.size()) < vocab_size) {
            // Add new token that is prefix + next input symbol
            TokenTuple new_prefix = prefix;
            new_prefix.push_back(to_encode[prefix.size()]);
            if (!unused_tokens.empty()) {
                token = *unused_tokens.begin();  // Always use the smallest available token
            }
            return {new_prefix, token};
        }
    }
    
    return {prefix, token};
}

/**
 * Add a new token to the vocabulary
 * Equivalent to LZCoder._add_new_token in Python
 */
void LZCoder::_add_new_token(const TokenTuple& prefix, TokenType token) {
    encoded_vocab[token] = prefix;
    token_map.insert(prefix, token);
    unused_tokens.erase(token);
}

//
// HierarchicalLZCoder implementation
//

/**
 * HierarchicalLZCoder constructor
 * Equivalent to HierarchicalLZCoder.__init__ in Python
 */
HierarchicalLZCoder::HierarchicalLZCoder(int output_vocab_size, VocabSet input_vocab)
    : vocab_size(output_vocab_size) {
    
    if (!input_vocab.empty() && input_vocab.size() > static_cast<size_t>(output_vocab_size)) {
        throw std::runtime_error("AssertionError: len(input_vocab) <= output_vocab_size");
    }

    // Initialize the empty token coder with the input vocab
    coders[EMPTY_TOKEN] = LZCoder(output_vocab_size, input_vocab);  // Copy the input vocab
}

/**
 * Update vocabulary with new input (string)
 * Equivalent to HierarchicalLZCoder.update_vocab in Python
 */
void HierarchicalLZCoder::update_vocab(const TokenSequence& to_encode) {
    coders[EMPTY_TOKEN].update_vocab(to_encode);
}

/**
 * Encode string input to tokens
 * Equivalent to HierarchicalLZCoder.encode for string in Python
 */
TokenSequence HierarchicalLZCoder::encode(const TokenSequence& to_encode, bool learn) {
    TokenSequence encoded;
    TokenType context = EMPTY_TOKEN;
    TokenSequence remaining = to_encode;

    while (!remaining.empty()) {
        auto [prefix, token] = encode_one_token(remaining, context, learn);
        encoded.push_back(token);
        context = token;  // Update context to be the token we just encoded
        remaining.erase(remaining.begin(), remaining.begin() + prefix.size());
    }

    return encoded;
}

/**
 * Encode one token (overridden from Coder)
 * Equivalent to HierarchicalLZCoder.encode_one_token without context in Python
 */
std::tuple<TokenTuple, TokenType> HierarchicalLZCoder::encode_one_token(const TokenSequence& to_encode, bool learn) {
    return encode_one_token(to_encode, EMPTY_TOKEN, learn);
}

/**
 * Encode one token with context
 * Equivalent to HierarchicalLZCoder.encode_one_token with context in Python
 */
std::tuple<TokenTuple, TokenType> HierarchicalLZCoder::encode_one_token(const TokenSequence& to_encode, TokenType context, bool learn) {
    if (to_encode.empty()) {
        return {TokenTuple(), EMPTY_TOKEN};
    }

    if (coders.find(context) == coders.end()) {
        if (!learn) {
            throw std::runtime_error("context not in coders");
        }
        // Make a new coder with empty input vocab
        coders.emplace(context, LZCoder(vocab_size, VocabSet()));
    }

    auto& coder = coders[context];
    auto [prefix, token] = coder._propose_next_token(to_encode, learn);

    if (coder.get_encoded_vocab().find(token) != coder.get_encoded_vocab().end()) {
        return {prefix, token};
    }

    if (!learn) {
        throw std::runtime_error("trying to add new token, but learning is disabled!");
    }

    // We want to add a new token for this context
    // First get the proposed token from the current context
    if (coder.get_unused_tokens().empty()) {
        throw std::runtime_error("no unused tokens available");
    }
    TokenType proposed_token = *coder.get_unused_tokens().begin();  // Always use the smallest available token

    // Count how often each token is proposed by other contexts
    std::map<TokenType, int> symbol_counts;
    symbol_counts[proposed_token] = 0;

    for (const auto& [other_context, other_coder] : coders) {
        if (other_context == context) continue;

        auto [_, other_token] = other_coder._propose_next_token(to_encode, learn);
        if (other_coder.get_encoded_vocab().find(other_token) != other_coder.get_encoded_vocab().end()) {
            symbol_counts[other_token] = symbol_counts.count(other_token) ? symbol_counts[other_token] + 1 : 1;
        }
    }

    // Find the untaken token with highest count
    TokenType best_token = proposed_token;
    int best_count = 0;
    for (const auto& [token, count] : symbol_counts) {
        if (count > best_count && coder.get_unused_tokens().count(token)) {
            best_token = token;
            best_count = count;
        }
    }

    coder._add_new_token(prefix, best_token);
    return {prefix, best_token};
}

/**
 * Decode tokens back to original form
 * Equivalent to HierarchicalLZCoder.decode in Python
 */
TokenSequence HierarchicalLZCoder::decode(const TokenSequence& to_decode) {
    TokenSequence decoded;
    TokenType context = EMPTY_TOKEN;
    for (TokenType token : to_decode) {
        auto it = coders.find(context);
        if (it == coders.end()) {
            throw std::runtime_error("context not in coders");
        }
        auto token_decoded = it->second.decode_one_token(token);
        decoded.insert(decoded.end(), token_decoded.begin(), token_decoded.end());
        context = token;
    }
    return decoded;
}

} // namespace lz 