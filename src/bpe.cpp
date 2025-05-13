#include "bpe.hpp"
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <iostream>

namespace bpe {

// Utility functions implementation
bool is_prefix(const TokenSequence& sequence, const TokenTuple& prefix) {
    if (prefix.size() > sequence.size()) {
        return false;
    }
    return std::equal(prefix.begin(), prefix.end(), sequence.begin());
}

TokenSequence ensure_list(const std::string& tokens) {
    // Convert a UTF-8 string to a vector of integers
    TokenSequence result;
    for (unsigned char c : tokens) {
        result.push_back(static_cast<TokenType>(c));
    }
    return result;
}

TokenSequence ensure_list(const std::vector<uint8_t>& tokens) {
    // Convert a vector of bytes to a vector of integers
    TokenSequence result;
    for (uint8_t c : tokens) {
        result.push_back(static_cast<TokenType>(c));
    }
    return result;
}

TokenSequence ensure_list(const TokenSequence& tokens) {
    // This is trivial, just return the input as it's already a TokenSequence
    return tokens;
}

TokenSequence ensure_list(const VocabSet& tokens) {
    // Convert a set of integers to a vector of integers
    TokenSequence result(tokens.begin(), tokens.end());
    return result;
}

// BPE implementation
std::unordered_map<TokenPair, int, TokenPairHash> get_stats(const TokenSequence& tokens) {
    // Calculate frequency statistics of adjacent token pairs
    std::unordered_map<TokenPair, int, TokenPairHash> stats;
    for (size_t i = 0; i < tokens.size() - 1; ++i) {
        TokenPair pair = {tokens[i], tokens[i + 1]};
        stats[pair]++;
    }
    return stats;
}

TokenSequence merge_pairs(const TokenSequence& tokens, const TokenPair& pair, TokenType new_token) {
    // Merge all occurrences of a token pair into a single new token
    TokenSequence merged;
    size_t i = 0;
    
    while (i < tokens.size()) {
        if (i < tokens.size() - 1 && tokens[i] == pair.first && tokens[i + 1] == pair.second) {
            merged.push_back(new_token);
            i += 2;  // Skip the pair
        } else {
            merged.push_back(tokens[i]);
            i += 1;
        }
    }
    
    return merged;
}

// BPE class implementation
BPE::BPE(const std::vector<TokenPair>& merges,
         const std::unordered_map<TokenType, TokenTuple>& token_values,
         const std::unordered_map<TokenType, TokenType>& input_vocab,
         std::optional<int> max_output_vocab,
         std::optional<int> max_merges)
    : merges_(merges), 
      token_values_(token_values),
      max_output_vocab_(max_output_vocab),
      max_merges_(max_merges) {
    
    // Initialize input_vocab_ from the input_vocab map
    for (const auto& [key, _] : input_vocab) {
        input_vocab_.insert(key);
    }
    
    // Check if either max_output_vocab or max_merges is provided
    if (!max_output_vocab.has_value() && !max_merges.has_value()) {
        throw std::invalid_argument("max_merges or max_output_vocab must be provided");
    }
}

BPE::BPE(std::optional<int> max_output_vocab, std::optional<int> max_merges)
    : merges_(),
      token_values_(),
      input_vocab_(),
      max_output_vocab_(max_output_vocab),
      max_merges_(max_merges) {
    
    // Check if either max_output_vocab or max_merges is provided
    if (!max_output_vocab.has_value() && !max_merges.has_value()) {
        throw std::invalid_argument("max_merges or max_output_vocab must be provided");
    }
}

void BPE::learn(const TokenSequence& tokens, 
                const std::optional<VocabSet>& input_vocab,
                bool debug) {
    // Initialize input vocabulary if not provided
    VocabSet vocab;
    if (input_vocab.has_value()) {
        vocab = input_vocab.value();
    } else {
        vocab = VocabSet(tokens.begin(), tokens.end());
    }
    
    if (debug) {
        std::cout << "BPE::learn - Starting learning with vocabulary size: " << vocab.size() << std::endl;
    }
    
    // Convert vocab to a list to fix order
    std::vector<TokenType> vocab_list(vocab.begin(), vocab.end());
    
    // Clear previous state and initialize with the input vocabulary
    merges_.clear();
    token_values_.clear();
    input_vocab_.clear();
    output_vocab_.clear();
    
    // Initialize merges with (0, x) pairs for each token in input_vocab
    // 0 corresponds to empty string
    for (TokenType token : vocab_list) {
        merges_.push_back({0, token});
        token_values_[token] = {token};
    }
    
    if (debug) {
        std::cout << "BPE::learn - Initialized " << merges_.size() << " initial merges" << std::endl;
    }
    
    // Set max_output_vocab if not specified
    if (!max_output_vocab_.has_value() && max_merges_.has_value()) {
        max_output_vocab_ = max_merges_.value() + merges_.size();
    }
    
    input_vocab_ = vocab;
    
    // Create inverse token values map for encoding
    std::unordered_map<TokenTuple, TokenType, TokenTupleHash, TokenTupleEquals> inverse_token_values;
    for (const auto& [token, value] : token_values_) {
        inverse_token_values[value] = token;
    }
    
    // Convert input tokens using the initial token mapping
    TokenSequence working_tokens;
    for (TokenType token : tokens) {
        TokenTuple single_token = {token};
        working_tokens.push_back(inverse_token_values[single_token]);
    }
    
    if (debug) {
        std::cout << "BPE::learn - Converted input tokens, ready for merging" << std::endl;
    }
    
    // If there are fewer than 2 tokens, we can't do any merges
    if (working_tokens.size() < 2) {
        if (debug) {
            std::cout << "BPE::learn - Too few tokens for merging, stopping" << std::endl;
        }
        
        // Update output vocabulary
        output_vocab_ = VocabSet();
        for (size_t i = 1; i <= merges_.size(); ++i) {
            output_vocab_.insert(static_cast<TokenType>(i));
        }
        return;
    }
    
    // Start with next_token as max input token value + 1
    TokenType next_token = *std::max_element(vocab_list.begin(), vocab_list.end()) + 1;
    
    if (debug) {
        std::cout << "BPE::learn - Starting merge process with target vocab size: " 
                  << max_output_vocab_.value() << std::endl;
    }
    
    int merge_count = 0;
    
    // Continue merging until we reach the maximum output vocabulary size
    while (merges_.size() < max_output_vocab_.value()) {
        // Calculate pair frequencies
        auto stats = get_stats(working_tokens);
        
        // If there are no pairs or all pairs occur only once, break
        if (stats.empty()) {
            if (debug) {
                std::cout << "BPE::learn - No more pairs to merge, stopping" << std::endl;
            }
            break;
        }
        
        // Find most frequent pair
        auto most_frequent_pair = std::max_element(
            stats.begin(), stats.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; }
        );
        
        // If the most frequent pair only occurs once, break
        if (most_frequent_pair->second == 1) {
            if (debug) {
                std::cout << "BPE::learn - Most frequent pair occurs only once, stopping" << std::endl;
            }
            break;
        }
        
        // Apply the merge operation
        TokenPair pair = most_frequent_pair->first;
        working_tokens = merge_pairs(working_tokens, pair, next_token);
        
        // Update merges and token values
        merges_.push_back(pair);
        
        // Concatenate the tokens represented by the pair elements
        TokenTuple new_token_value;
        const auto& first_token_value = token_values_[pair.first];
        const auto& second_token_value = token_values_[pair.second];
        
        new_token_value.insert(new_token_value.end(), first_token_value.begin(), first_token_value.end());
        new_token_value.insert(new_token_value.end(), second_token_value.begin(), second_token_value.end());
        
        // Associate the new token ID with the merge and its token value
        // The token ID for merges after the initial vocab is the merge index + 1
        TokenType merge_token_id = static_cast<TokenType>(merges_.size());
        token_values_[merge_token_id] = new_token_value;
        inverse_token_values[new_token_value] = merge_token_id;
        
        // Increment next token
        next_token++;
        merge_count++;
        
        if (debug && merge_count % 100 == 0) {
            std::cout << "BPE::learn - Completed " << merge_count << " merges" << std::endl;
        }
    }
    
    if (debug) {
        std::cout << "BPE::learn - Finished learning with " << merge_count
                  << " total merges and final vocabulary size: " << merges_.size() << std::endl;
    }
    
    // Update output vocabulary
    output_vocab_ = VocabSet();
    for (size_t i = 1; i <= merges_.size(); ++i) {
        output_vocab_.insert(static_cast<TokenType>(i));
    }
    
    if (debug) {
        std::cout << "BPE::learn - Final output vocabulary size: " << output_vocab_.size() << std::endl;
    }
    
    // // Debug output of token_values_
    // std::cout << "Final token_values_ after learning:" << std::endl;
    // for (const auto& [token, value] : token_values_) {
    //     std::cout << "Token " << token << " -> [";
    //     for (size_t i = 0; i < value.size(); ++i) {
    //         if (i > 0) std::cout << ", ";
    //         std::cout << value[i];
    //     }
    //     std::cout << "]" << std::endl;
    // }
}

TokenSequence BPE::encode(const TokenSequence& tokens) {
    // Apply merges in the same order they were learned
    TokenSequence encoded = tokens;  // Start with the input tokens
    
    // Apply each merge operation in sequence
    for (size_t i = 0; i < merges_.size(); ++i) {
        // Skip merges that are just adding tokens from input vocab
        if (merges_[i].first == 0) {
            continue;
        }
        
        // The new token ID is the position in merges_ + 1
        // (adding 1 because 0 is reserved for the empty string)
        TokenType new_token = static_cast<TokenType>(i + 1);
        
        // Apply the merge
        encoded = merge_pairs(encoded, merges_[i], new_token);
    }
    
    return encoded;
}

TokenSequence BPE::decode(const TokenSequence& tokens) {
    // Decode BPE-encoded tokens back to their original form
    TokenSequence decoded;
    
    for (TokenType token : tokens) {
        // If the token is in token_values_, expand it
        auto it = token_values_.find(token);
        if (it != token_values_.end()) {
            const auto& expanded = it->second;
            decoded.insert(decoded.end(), expanded.begin(), expanded.end());
        } else {
            // If the token is not found (should not happen), keep it as is
            decoded.push_back(token);
        }
    }
    
    return decoded;
}

// Helper functions for ContextualEncoder

/**
 * Calculate statistics about token sequences in different contexts
 */
std::unordered_map<TokenType, std::unordered_map<TokenType, std::unordered_map<TokenTuple, int, TokenTupleHash, TokenTupleEquals>>> 
get_context_stats(const TokenSequence& tokens, const VocabSet& vocab, bool debug) {
    if (debug) {
        std::cout << "get_context_stats - Starting analysis of " << tokens.size() 
                  << " tokens with vocabulary size " << vocab.size() << std::endl;
    }
    
    // Initialize stats dictionary - for each context and each token, store a map of substrings to counts
    std::unordered_map<TokenType, std::unordered_map<TokenType, std::unordered_map<TokenTuple, int, TokenTupleHash, TokenTupleEquals>>> stats;
    
    // Initialize stats for each context and token pair
    if (debug) {
        std::cout << "get_context_stats - Initializing data structures..." << std::endl;
    }
    
    for (const auto& context : vocab) {
        stats[context] = std::unordered_map<TokenType, std::unordered_map<TokenTuple, int, TokenTupleHash, TokenTupleEquals>>();
        for (const auto& token : vocab) {
            stats[context][token] = std::unordered_map<TokenTuple, int, TokenTupleHash, TokenTupleEquals>();
        }
    }
    
    // Initialize starting point indices for each vocab token
    std::unordered_map<TokenType, int> start_idx;
    for (const auto& v : vocab) {
        start_idx[v] = -1;
    }
    
    if (debug) {
        std::cout << "get_context_stats - Processing token sequence..." << std::endl;
    }
    
    // For tracking progress in debug mode
    size_t last_progress = 0;
    
    // Process each token in the sequence
    for (size_t idx = 0; idx < tokens.size(); ++idx) {
        TokenType token = tokens[idx];
        
        // Update string finders for all other tokens
        for (const auto& v : vocab) {
            int start = start_idx[v];
            if (start != -1) {
                // Extract substring from start+1 to current position
                TokenTuple sub_string(tokens.begin() + start + 1, tokens.begin() + idx + 1);
                stats[v][token][sub_string]++;
            }
        }
        
        // Restart string finder for current token
        start_idx[token] = idx;
        
        // Print progress if in debug mode
        if (debug && (idx * 100 / tokens.size() > last_progress || idx == tokens.size() - 1)) {
            last_progress = idx * 100 / tokens.size();
            std::cout << "get_context_stats - Progress: " << last_progress << "% (" 
                      << idx + 1 << "/" << tokens.size() << " tokens)" << std::endl;
        }
    }
    
    if (debug) {
        // Count total number of context-token pairs with statistics
        size_t context_pairs = 0;
        size_t total_substrings = 0;
        
        for (const auto& [context, token_map] : stats) {
            for (const auto& [token, substring_map] : token_map) {
                if (!substring_map.empty()) {
                    context_pairs++;
                    total_substrings += substring_map.size();
                }
            }
        }
        
        std::cout << "get_context_stats - Finished with statistics for " << context_pairs 
                  << " context-token pairs and " << total_substrings << " unique substrings" << std::endl;
    }
    
    return stats;
}

/**
 * Learn a contextual tokenizer from input tokens
 */
std::unordered_map<TokenType, std::unordered_map<TokenType, TokenTuple>> 
learn_contextual_tokenizer(const TokenSequence& tokens, const std::optional<VocabSet>& vocab_opt, bool debug) {
    if (debug) {
        std::cout << "learn_contextual_tokenizer - Starting with " << tokens.size() << " input tokens" << std::endl;
    }
    
    // Initialize vocabulary if not provided
    VocabSet vocab;
    if (vocab_opt.has_value()) {
        vocab = vocab_opt.value();
    } else {
        vocab = VocabSet(tokens.begin(), tokens.end());
    }
    
    if (debug) {
        std::cout << "learn_contextual_tokenizer - Initialized vocabulary with " << vocab.size() << " tokens" << std::endl;
    }
    
    // Get context statistics
    if (debug) {
        std::cout << "learn_contextual_tokenizer - Computing context statistics..." << std::endl;
    }
    auto contextual_token_counts = get_context_stats(tokens, vocab, debug);
    
    if (debug) {
        std::cout << "learn_contextual_tokenizer - Gathered statistics for " << contextual_token_counts.size() << " contexts" << std::endl;
    }
    
    // Initialize contextual_tokens with empty string context
    std::unordered_map<TokenType, std::unordered_map<TokenType, TokenTuple>> contextual_tokens;
    
    // Zero is the "empty string" token
    // The empty string context can generate empty string
    for (const auto& v : vocab) {
        contextual_tokens[v] = std::unordered_map<TokenType, TokenTuple>();
        contextual_tokens[v][0] = TokenTuple();  // Empty string
    }
    
    if (debug) {
        std::cout << "learn_contextual_tokenizer - Processing contexts and end tokens..." << std::endl;
    }
    
    // For debug progress tracking
    int processed = 0;
    size_t total_pairs = vocab.size() * vocab.size();
    
    // Process each context and end token
    for (const auto& context : vocab) {
        for (const auto& end_token : vocab) {
            // if (debug) {
            //     processed++;
            //     if (processed % 100 == 0 || processed == static_cast<int>(total_pairs)) {
            //         std::cout << "learn_contextual_tokenizer - Processed " << processed << " of " 
            //                   << total_pairs << " context-token pairs" << std::endl;
            //     }
            // }
            
            if (end_token == 0) {
                // The empty token must always mean the empty string
                continue;
            }
            
            // Find the longest substring for this context and end token
            if (!contextual_token_counts[context][end_token].empty()) {
                TokenTuple longest_string;
                int max_count = 0;
                
                // Find most frequent substring
                for (const auto& [substring, count] : contextual_token_counts[context][end_token]) {
                    if (count > max_count) {
                        max_count = count;
                        longest_string = substring;
                    }
                }
                
                contextual_tokens[context][end_token] = longest_string;
            }
        }
    }
    
    // Empty string can generate any singleton
    contextual_tokens[0] = std::unordered_map<TokenType, TokenTuple>();
    for (const auto& v : vocab) {
        TokenTuple singleton = {v};
        contextual_tokens[0][v] = singleton;
    }
    
    if (debug) {
        std::cout << "learn_contextual_tokenizer - Finished processing with " 
                  << contextual_tokens.size() << " unique contexts" << std::endl;
    }
    
    return contextual_tokens;
}

/* 
 * function that prints out the contents of a contextual_tokens map
*/
void print_contextual_tokens(const std::unordered_map<TokenType, std::unordered_map<TokenType, TokenTuple>>& contextual_tokens) {
    for (const auto& [context, token_map] : contextual_tokens) {
        std::cout << "Context: " << context << std::endl;
        for (const auto& [token, value] : token_map) {
            std::cout << "Token: " << token << " -> [";
            for (size_t i = 0; i < value.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << value[i];
            }
            std::cout << "]" << std::endl;
        }
    }
}

/*
 * function that prints out the contextual tokens for a specific context
*/
void print_contextual_tokens_for_context(const std::unordered_map<TokenType, std::unordered_map<TokenType, TokenTuple>>& contextual_tokens, TokenType context) {
    auto it = contextual_tokens.find(context);
    if (it == contextual_tokens.end()) {
        std::cout << "Context " << context << " not found in contextual tokens map" << std::endl;
        return;
    }
    
    std::cout << "Context: " << context << std::endl;
    for (const auto& [token, value] : it->second) {
        std::cout << "Token: " << token << " -> [";
        for (size_t i = 0; i < value.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << value[i];
        }
        std::cout << "]" << std::endl;
    }
}

/**
 * Encode tokens using a contextual tokenizer
 */
TokenSequence contextual_encode(
    const TokenSequence& tokens, 
    const std::unordered_map<TokenType, std::unordered_map<TokenType, TokenTuple>>& contextual_tokens,
    bool debug) {
    
    if (debug) {
        std::cout << "contextual_encode - Starting encoding of " << tokens.size() << " tokens" << std::endl;
    }
    
    // Start with empty context
    TokenSequence encoded;
    TokenType context = 0;
    
    size_t cur_idx = 0;
    
    while (cur_idx < tokens.size()) {
        TokenType best_match = 0;
        TokenTuple best_value;
        
        // Find the best token match in current context
        for (const auto& [tok_idx, tok_value] : contextual_tokens.at(context)) {
            // If tokens[cur_idx:] starts with tok_value, we have a match
            if (cur_idx + tok_value.size() <= tokens.size() && 
                std::equal(tok_value.begin(), tok_value.end(), tokens.begin() + cur_idx)) {
                // Find the longest match
                if (tok_value.size() > best_value.size()) {
                    best_match = tok_idx;
                    best_value = tok_value;
                }
            }
        }

        // if (best_match == 0) {
        //     std::cout << "No match found for context " << context << " and token " << tokens[cur_idx] << " at index " << cur_idx << std::endl;
        //     print_contextual_tokens_for_context(contextual_tokens, context);
        // }
        
        if (debug && best_match == 0) {
            std::cout << "contextual_encode - Warning: No match found for context " << context 
                      << " at index " << cur_idx << std::endl;
        }
        
        encoded.push_back(best_match);
        context = best_match;
        cur_idx += best_value.size();
        
        if (debug && (cur_idx % 1000 == 0 || cur_idx == tokens.size())) {
            std::cout << "contextual_encode - Processed " << cur_idx << " of " << tokens.size() 
                      << " tokens (" << (cur_idx * 100 / tokens.size()) << "%)" << std::endl;
        }
    }
    
    if (debug) {
        std::cout << "contextual_encode - Finished encoding, produced " << encoded.size() 
                  << " tokens from " << tokens.size() << " input tokens" << std::endl;
    }
    
    return encoded;
}

/**
 * Decode contextually encoded tokens back to their original form
 */
TokenSequence contextual_decode(
    const TokenSequence& tokens, 
    const std::unordered_map<TokenType, std::unordered_map<TokenType, TokenTuple>>& contextual_tokens,
    TokenType initial_context,
    bool debug) {
    
    if (debug) {
        std::cout << "contextual_decode - Starting decoding of " << tokens.size() << " tokens" << std::endl;
    }
    
    TokenSequence decoded;
    TokenType context = initial_context;
    
    for (size_t i = 0; i < tokens.size(); ++i) {
        const auto& token = tokens[i];
        
        // Get the token value for this context and token
        const auto& token_value = contextual_tokens.at(context).at(token);
        
        // Add token value to decoded sequence
        decoded.insert(decoded.end(), token_value.begin(), token_value.end());
        
        // Update context
        context = token;
        
        if (debug && (i % 1000 == 0 || i == tokens.size() - 1)) {
            std::cout << "contextual_decode - Processed " << (i + 1) << " of " << tokens.size() 
                      << " tokens (" << ((i + 1) * 100 / tokens.size()) << "%)" << std::endl;
        }
    }
    
    if (debug) {
        std::cout << "contextual_decode - Finished decoding, produced " << decoded.size() 
                  << " tokens from " << tokens.size() << " input tokens" << std::endl;
    }
    
    return decoded;
}

// ContextualEncoder implementation
ContextualEncoder::ContextualEncoder(std::optional<int> max_token_value)
    : max_token_value_(max_token_value) {
}

void ContextualEncoder::learn(const TokenSequence& tokens, 
                         const std::optional<VocabSet>& input_vocab,
                         bool debug) {
    if (debug) {
        std::cout << "ContextualEncoder::learn - Starting learning" << std::endl;
    }
    
    // Learn contextual tokenizer
    context_map_ = learn_contextual_tokenizer(tokens, input_vocab, debug);
    input_vocab_ = VocabSet();
    output_vocab_ = VocabSet();
    
    if (debug) {
        std::cout << "ContextualEncoder::learn - Created context map with " 
                  << context_map_.size() << " contexts" << std::endl;
    }
    
    // Update input and output vocabularies
    for (const auto& [context, _] : context_map_) {
        input_vocab_.insert(context);
        output_vocab_.insert(context);
    }
    
    if (debug) {
        std::cout << "ContextualEncoder::learn - Finished with vocabulary size: " 
                  << output_vocab_.size() << std::endl;
    }
}

TokenSequence ContextualEncoder::encode(const TokenSequence& tokens) {
    // Add debug=false parameter to match the updated contextual_encode function
    return contextual_encode(tokens, context_map_, false);
}

TokenSequence ContextualEncoder::encode_with_debug(const TokenSequence& tokens, bool debug) {
    // Allow passing the debug flag
    return contextual_encode(tokens, context_map_, debug);
}

TokenSequence ContextualEncoder::decode(const TokenSequence& tokens) {
    // Add debug=false parameter to match the updated contextual_decode function
    return contextual_decode(tokens, context_map_, 0, false);
}

TokenSequence ContextualEncoder::decode_with_debug(const TokenSequence& tokens, bool debug) {
    // Allow passing the debug flag
    return contextual_decode(tokens, context_map_, 0, debug);
}

// DefragEncoder implementation
DefragEncoder::DefragEncoder() {
    // Initialize with empty mappings
}

void DefragEncoder::learn(const TokenSequence& tokens, 
                          const std::optional<VocabSet>& input_vocab,
                          bool debug) {
    if (debug) {
        std::cout << "DefragEncoder::learn - Starting learning" << std::endl;
    }
    
    // Determine input vocabulary
    if (input_vocab.has_value()) {
        input_vocab_ = input_vocab.value();
    } else {
        input_vocab_ = VocabSet(tokens.begin(), tokens.end());
    }
    
    if (debug) {
        std::cout << "DefragEncoder::learn - Input vocabulary size: " << input_vocab_.size() << std::endl;
    }
    
    // Create continuous range of integers for output vocabulary
    output_vocab_.clear();
    for (TokenType i = 1; i <= input_vocab_.size(); ++i) {
        output_vocab_.insert(i);
    }
    
    // Create mappings
    vocab_to_token_.clear();
    token_to_vocab_.clear();
    
    TokenType idx = 1;
    for (const auto& token : input_vocab_) {
        vocab_to_token_[token] = idx;
        token_to_vocab_[idx] = token;
        ++idx;
    }
    
    if (debug) {
        std::cout << "DefragEncoder::learn - Created mappings for " 
                  << vocab_to_token_.size() << " tokens" << std::endl;
        std::cout << "DefragEncoder::learn - Output vocabulary size: " << output_vocab_.size() << std::endl;
    }
}

TokenSequence DefragEncoder::encode(const TokenSequence& tokens) {
    // Map input tokens to continuous range
    TokenSequence encoded;
    encoded.reserve(tokens.size());
    
    for (const auto& token : tokens) {
        encoded.push_back(vocab_to_token_[token]);
    }
    
    return encoded;
}

TokenSequence DefragEncoder::decode(const TokenSequence& tokens) {
    // Map continuous range back to original tokens
    TokenSequence decoded;
    decoded.reserve(tokens.size());
    
    for (const auto& token : tokens) {
        decoded.push_back(token_to_vocab_[token]);
    }
    
    return decoded;
}

// ComposedTokenizer implementation
ComposedTokenizer::ComposedTokenizer(const std::vector<std::shared_ptr<Tokenizer>>& tokenizers)
    : tokenizers_(tokenizers) {
    // Nothing else to initialize
}

void ComposedTokenizer::learn(const TokenSequence& tokens, 
                             const std::optional<VocabSet>& input_vocab,
                             bool debug) {
    if (tokenizers_.empty()) {
        if (debug) {
            std::cout << "ComposedTokenizer::learn - No tokenizers to learn with" << std::endl;
        }
        return;
    }
    
    if (debug) {
        std::cout << "ComposedTokenizer::learn - Starting learning with " 
                  << tokenizers_.size() << " tokenizers" << std::endl;
    }
    
    // Start with the input tokens
    TokenSequence current_tokens = tokens;
    
    // For the first tokenizer, use the provided input_vocab
    if (debug) {
        std::cout << "ComposedTokenizer::learn - Learning first tokenizer" << std::endl;
    }
    tokenizers_[0]->learn(current_tokens, input_vocab, debug);
    current_tokens = tokenizers_[0]->encode(current_tokens);
    
    // For the rest of the tokenizers, use no input_vocab constraints
    for (size_t i = 1; i < tokenizers_.size(); ++i) {
        if (debug) {
            std::cout << "ComposedTokenizer::learn - Learning tokenizer " << i + 1 
                      << " of " << tokenizers_.size() << std::endl;
        }
        // pass in the current tokens, and use the previous tokenizer's output vocabulary as the input vocabulary
        tokenizers_[i]->learn(current_tokens, tokenizers_[i-1]->get_output_vocab(), debug);
        current_tokens = tokenizers_[i]->encode(current_tokens);
    }
    
    if (debug) {
        std::cout << "ComposedTokenizer::learn - Finished learning all tokenizers" << std::endl;
        
        if (!tokenizers_.empty()) {
            std::cout << "ComposedTokenizer::learn - Final input vocabulary size: " 
                      << tokenizers_.front()->get_input_vocab().size() << std::endl;
            std::cout << "ComposedTokenizer::learn - Final output vocabulary size: " 
                      << tokenizers_.back()->get_output_vocab().size() << std::endl;
        }
    }
}

TokenSequence ComposedTokenizer::encode(const TokenSequence& tokens) {
    if (tokenizers_.empty()) {
        return tokens;
    }
    
    // Start with the input tokens
    TokenSequence current_tokens = tokens;
    
    // Pass through each tokenizer in sequence
    for (auto& tokenizer : tokenizers_) {
        current_tokens = tokenizer->encode(current_tokens);
    }
    
    return current_tokens;
}

TokenSequence ComposedTokenizer::decode(const TokenSequence& tokens) {
    if (tokenizers_.empty()) {
        return tokens;
    }
    
    // Start with the input tokens
    TokenSequence current_tokens = tokens;
    
    // Pass through each tokenizer in reverse sequence
    for (auto it = tokenizers_.rbegin(); it != tokenizers_.rend(); ++it) {
        current_tokens = (*it)->decode(current_tokens);
    }
    
    return current_tokens;
}

} // namespace bpe 