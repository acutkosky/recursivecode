#ifndef BPE_HPP
#define BPE_HPP

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <tuple>
#include <memory>
#include <optional>
#include <utility>

namespace bpe {

// Type definitions to match Python implementation
using TokenType = int;
using TokenSequence = std::vector<TokenType>;
using TokenPair = std::pair<TokenType, TokenType>;
using TokenTuple = std::vector<TokenType>;
using VocabSet = std::unordered_set<TokenType>;

// Hash function for TokenPair to use in unordered_map
struct TokenPairHash {
    std::size_t operator()(const TokenPair& pair) const {
        return std::hash<TokenType>()(pair.first) ^ (std::hash<TokenType>()(pair.second) << 1);
    }
};

// Hash function for TokenTuple (vector<int>) to use in unordered_map
struct TokenTupleHash {
    std::size_t operator()(const TokenTuple& tuple) const {
        std::size_t seed = tuple.size();
        for (const auto& i : tuple) {
            seed ^= std::hash<TokenType>()(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

// Equals function for TokenTuple
struct TokenTupleEquals {
    bool operator()(const TokenTuple& lhs, const TokenTuple& rhs) const {
        return lhs == rhs;
    }
};

// Utility functions
/**
 * @brief Check if a sequence starts with a given prefix
 * @param sequence List of integers to check
 * @param prefix Vector of integers representing the prefix
 * @return True if sequence starts with prefix, False otherwise
 */
bool is_prefix(const TokenSequence& sequence, const TokenTuple& prefix);

/**
 * @brief Convert input tokens to a vector of integers
 * @param tokens Input tokens as vector of integers, string, or bytes
 * @return Vector of integer token IDs
 */
TokenSequence ensure_list(const std::string& tokens);
TokenSequence ensure_list(const std::vector<uint8_t>& tokens);
TokenSequence ensure_list(const TokenSequence& tokens);
TokenSequence ensure_list(const VocabSet& tokens);

/**
 * @brief Base class for all tokenizer implementations
 */
class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    /**
     * @brief Learn tokenization patterns from input data
     * @param tokens Input tokens to learn from
     * @param input_vocab Optional set of input vocabulary tokens to consider
     * @param debug Whether to print debug information during learning
     */
    virtual void learn(const TokenSequence& tokens, 
                       const std::optional<VocabSet>& input_vocab = std::nullopt,
                       bool debug = false) = 0;

    /**
     * @brief Encode input tokens into a new token representation
     * @param tokens Input tokens to encode
     * @return Vector of encoded token IDs
     */
    virtual TokenSequence encode(const TokenSequence& tokens) = 0;

    /**
     * @brief Decode encoded tokens back to their original representation
     * @param tokens Encoded tokens to decode
     * @return Vector of decoded token IDs
     */
    virtual TokenSequence decode(const TokenSequence& tokens) = 0;
    
    /**
     * @brief Get the input vocabulary set
     * @return Set of input vocabulary tokens
     */
    virtual const VocabSet& get_input_vocab() const = 0;
    
    /**
     * @brief Get the output vocabulary set
     * @return Set of output token IDs
     */
    virtual const VocabSet& get_output_vocab() const = 0;
};

// BPE utility functions
/**
 * @brief Calculate frequency statistics of adjacent token pairs in the input sequence
 * @param tokens Vector of token IDs to analyze
 * @return Unordered map mapping token pairs to their frequency counts
 */
std::unordered_map<TokenPair, int, TokenPairHash> get_stats(const TokenSequence& tokens);

/**
 * @brief Merge all occurrences of a token pair into a single new token
 * @param tokens Vector of token IDs to process
 * @param pair Pair of two token IDs to merge
 * @param new_token Token ID to use for the merged pair
 * @return New vector of tokens with all occurrences of the pair merged
 */
TokenSequence merge_pairs(const TokenSequence& tokens, const TokenPair& pair, TokenType new_token);

// ContextualEncoder utility functions
/**
 * @brief Calculate statistics about token sequences in different contexts
 * @param tokens Vector of token IDs to analyze
 * @param vocab Set of vocabulary tokens to consider
 * @return Nested map of context -> token -> substring -> count
 */
std::unordered_map<TokenType, std::unordered_map<TokenType, std::unordered_map<TokenTuple, int, TokenTupleHash, TokenTupleEquals>>> 
get_context_stats(const TokenSequence& tokens, const VocabSet& vocab);

/**
 * @brief Learn a contextual tokenizer from input tokens
 * @param tokens Vector of token IDs to learn from
 * @param vocab Optional set of vocabulary tokens to consider
 * @return Map of context tokens to their contextual token mappings
 */
std::unordered_map<TokenType, std::unordered_map<TokenType, TokenTuple>> 
learn_contextual_tokenizer(const TokenSequence& tokens, const std::optional<VocabSet>& vocab = std::nullopt);

/**
 * @brief Encode tokens using a contextual tokenizer
 * @param tokens Vector of token IDs to encode
 * @param contextual_tokens Map of context tokens to their contextual token mappings
 * @return Vector of contextually encoded token IDs
 */
TokenSequence contextual_encode(
    const TokenSequence& tokens, 
    const std::unordered_map<TokenType, std::unordered_map<TokenType, TokenTuple>>& contextual_tokens);

/**
 * @brief Decode contextually encoded tokens back to their original form
 * @param tokens Vector of contextually encoded token IDs
 * @param contextual_tokens Map of context tokens to their contextual token mappings
 * @param initial_context Initial context token ID
 * @return Vector of decoded original token IDs
 */
TokenSequence contextual_decode(
    const TokenSequence& tokens, 
    const std::unordered_map<TokenType, std::unordered_map<TokenType, TokenTuple>>& contextual_tokens,
    TokenType initial_context = 0);

/**
 * @brief BPE tokenizer implementation
 */
class BPE : public Tokenizer {
public:
    /**
     * @brief Construct a new BPE tokenizer with all parameters
     * @param merges Optional list of token pairs that have been merged during training
     * @param token_values Optional map of token IDs to their corresponding values
     * @param input_vocab Optional map of input token IDs to their corresponding values
     * @param max_output_vocab Optional maximum size of the output vocabulary
     * @param max_merges Optional maximum number of merges to perform
     */
    BPE(const std::vector<TokenPair>& merges = {},
        const std::unordered_map<TokenType, TokenTuple>& token_values = {},
        const std::unordered_map<TokenType, TokenType>& input_vocab = {},
        std::optional<int> max_output_vocab = std::nullopt,
        std::optional<int> max_merges = std::nullopt);

    /**
     * @brief Construct a new BPE tokenizer with only vocabulary constraints
     * @param max_output_vocab Optional maximum size of the output vocabulary
     * @param max_merges Optional maximum number of merges to perform
     */
    BPE(std::optional<int> max_output_vocab, std::optional<int> max_merges);

    /**
     * @brief Learn a BPE tokenizer from input tokens
     * @param tokens Input tokens as vector of integers
     * @param input_vocab Optional set of input vocabulary tokens
     * @param debug Whether to print debug information during learning
     * @return Vector of tokenized input
     */
    void learn(const TokenSequence& tokens, 
               const std::optional<VocabSet>& input_vocab = std::nullopt,
               bool debug = false) override;

    /**
     * @brief Encode input tokens using a trained BPE tokenizer
     * @param tokens Vector of input token IDs
     * @return Vector of encoded token IDs
     */
    TokenSequence encode(const TokenSequence& tokens) override;

    /**
     * @brief Decode BPE-encoded tokens back to their original form
     * @param tokens Vector of encoded token IDs
     * @return Vector of decoded token IDs
     */
    TokenSequence decode(const TokenSequence& tokens) override;

    /**
     * @brief Get the token values map (for debugging)
     * @return Map of token IDs to their corresponding values
     */
    const std::unordered_map<TokenType, TokenTuple>& get_token_values() const {
        return token_values_;
    }

    /**
     * @brief Get the input vocabulary set
     * @return Set of input vocabulary tokens
     */
    const VocabSet& get_input_vocab() const override {
        return input_vocab_;
    }

    /**
     * @brief Get the output vocabulary set
     * @return Set of output token IDs
     */
    const VocabSet& get_output_vocab() const override {
        return output_vocab_;
    }

private:
    std::vector<TokenPair> merges_;
    std::unordered_map<TokenType, TokenTuple> token_values_;
    VocabSet input_vocab_;  // Changed from std::unordered_map<TokenType, TokenType> to VocabSet
    VocabSet output_vocab_;
    std::optional<int> max_output_vocab_;
    std::optional<int> max_merges_;
};

/**
 * @brief ContextualEncoder implementation
 * Encodes tokens based on their context in the sequence
 */
class ContextualEncoder : public Tokenizer {
public:
    /**
     * @brief Construct a new ContextualEncoder
     * @param max_token_value Optional maximum token value to use
     */
    ContextualEncoder(std::optional<int> max_token_value = std::nullopt);

    /**
     * @brief Learn contextual encoding patterns from input data
     * @param tokens Input tokens to learn from
     * @param input_vocab Optional set of input vocabulary tokens to consider
     * @param debug Whether to print debug information during learning
     */
    void learn(const TokenSequence& tokens, 
               const std::optional<VocabSet>& input_vocab = std::nullopt,
               bool debug = false) override;

    /**
     * @brief Encode input tokens using contextual information
     * @param tokens Input tokens to encode
     * @return Vector of encoded token IDs
     */
    TokenSequence encode(const TokenSequence& tokens) override;

    /**
     * @brief Decode contextually encoded tokens back to their original form
     * @param tokens Encoded tokens to decode
     * @return Vector of decoded token IDs
     */
    TokenSequence decode(const TokenSequence& tokens) override;

    /**
     * @brief Get the input vocabulary set
     * @return Set of input vocabulary tokens
     */
    const VocabSet& get_input_vocab() const override {
        return input_vocab_;
    }

    /**
     * @brief Get the output vocabulary set
     * @return Set of output token IDs
     */
    const VocabSet& get_output_vocab() const override {
        return output_vocab_;
    }

private:
    VocabSet input_vocab_;
    VocabSet output_vocab_;
    std::unordered_map<TokenType, std::unordered_map<TokenType, TokenTuple>> context_map_;
    std::optional<int> max_token_value_;
};

/**
 * @brief DefragEncoder implementation
 * Maps input vocabulary tokens to a continuous range of integers
 */
class DefragEncoder : public Tokenizer {
public:
    /**
     * @brief Construct a new DefragEncoder
     */
    DefragEncoder();

    /**
     * @brief Learn vocabulary mapping from input data
     * @param tokens Input tokens to learn from
     * @param input_vocab Optional set of input vocabulary tokens to consider
     * @param debug Whether to print debug information during learning
     */
    void learn(const TokenSequence& tokens, 
               const std::optional<VocabSet>& input_vocab = std::nullopt,
               bool debug = false) override;

    /**
     * @brief Encode input tokens using the learned vocabulary mapping
     * @param tokens Input tokens to encode
     * @return Vector of encoded token IDs in the range [1, len(vocab)]
     */
    TokenSequence encode(const TokenSequence& tokens) override;

    /**
     * @brief Decode encoded tokens back to their original vocabulary tokens
     * @param tokens Encoded tokens to decode
     * @return Vector of decoded original tokens
     */
    TokenSequence decode(const TokenSequence& tokens) override;

    /**
     * @brief Get the input vocabulary set
     * @return Set of input vocabulary tokens
     */
    const VocabSet& get_input_vocab() const override {
        return input_vocab_;
    }

    /**
     * @brief Get the output vocabulary set
     * @return Set of output token IDs
     */
    const VocabSet& get_output_vocab() const override {
        return output_vocab_;
    }

private:
    std::unordered_map<TokenType, TokenType> vocab_to_token_;
    std::unordered_map<TokenType, TokenType> token_to_vocab_;
    VocabSet input_vocab_;
    VocabSet output_vocab_;
};

/**
 * @brief ComposedTokenizer implementation
 * Composes multiple tokenizers into a single pipeline
 */
class ComposedTokenizer : public Tokenizer {
public:
    /**
     * @brief Construct a new ComposedTokenizer
     * @param tokenizers Vector of tokenizers to compose
     */
    ComposedTokenizer(const std::vector<std::shared_ptr<Tokenizer>>& tokenizers);

    /**
     * @brief Learn from input data by passing it through each tokenizer in sequence
     * @param tokens Input tokens to learn from
     * @param input_vocab Optional set of input vocabulary tokens to consider
     * @param debug Whether to print debug information during learning
     */
    void learn(const TokenSequence& tokens, 
               const std::optional<VocabSet>& input_vocab = std::nullopt,
               bool debug = false) override;

    /**
     * @brief Encode input tokens by passing them through each tokenizer in sequence
     * @param tokens Input tokens to encode
     * @return Vector of encoded token IDs
     */
    TokenSequence encode(const TokenSequence& tokens) override;

    /**
     * @brief Decode tokens by passing them through each tokenizer in reverse sequence
     * @param tokens Encoded tokens to decode
     * @return Vector of decoded token IDs
     */
    TokenSequence decode(const TokenSequence& tokens) override;

    /**
     * @brief Get the input vocabulary set
     * @return Set of input vocabulary tokens
     */
    const VocabSet& get_input_vocab() const override {
        if (tokenizers_.empty()) {
            static VocabSet empty;
            return empty;
        }
        return tokenizers_.front()->get_input_vocab();
    }

    /**
     * @brief Get the output vocabulary set
     * @return Set of output token IDs
     */
    const VocabSet& get_output_vocab() const override {
        if (tokenizers_.empty()) {
            static VocabSet empty;
            return empty;
        }
        return tokenizers_.back()->get_output_vocab();
    }

private:
    std::vector<std::shared_ptr<Tokenizer>> tokenizers_;
};

// namespace bpe
}

#endif // BPE_HPP 