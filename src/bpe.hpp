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
     */
    virtual void learn(const TokenSequence& tokens, 
                       const std::optional<VocabSet>& input_vocab = std::nullopt) = 0;

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
};

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

/**
 * @brief BPE tokenizer implementation
 */
class BPE : public Tokenizer {
public:
    /**
     * @brief Construct a new BPE tokenizer
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
     * @brief Learn a BPE tokenizer from input tokens
     * @param tokens Input tokens as vector of integers
     * @param input_vocab Optional set of input vocabulary tokens
     * @return Vector of tokenized input
     */
    void learn(const TokenSequence& tokens, 
               const std::optional<VocabSet>& input_vocab = std::nullopt) override;

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

private:
    std::vector<TokenPair> merges_;
    std::unordered_map<TokenType, TokenTuple> token_values_;
    std::unordered_map<TokenType, TokenType> input_vocab_;
    VocabSet output_vocab_;
    std::optional<int> max_output_vocab_;
    std::optional<int> max_merges_;
};

// Other classes that could be implemented later:
// class DefragEncoder : public Tokenizer {...};
// class ContextualEncoder : public Tokenizer {...};
// class ComposedTokenizer : public Tokenizer {...};

} // namespace bpe

#endif // BPE_HPP 