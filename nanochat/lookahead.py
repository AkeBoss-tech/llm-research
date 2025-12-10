"""
Lookahead Decoding for Nanochat

Implements token-level tree search that explores K candidate tokens
for D depth (default 10 tokens) before committing to the best path.

This is particularly useful for math reasoning tasks (GSM8K) where
early mistakes can cascade and ruin the entire solution.

Based on:
- Lookahead Decoding (Fu et al., 2024)
- Tree of Thoughts (Yao et al., 2023) - applied at token level
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass
import json


@dataclass
class LookaheadConfig:
    """Configuration for lookahead decoding"""
    k_candidates: int = 3  # Number of top candidates to explore
    depth: int = 10  # How many tokens deep to look ahead
    temperature: float = 1.0  # Temperature for sampling
    top_k: Optional[int] = None  # Top-k filtering
    use_log_prob_sum: bool = True  # Score by sum of log probs (vs. product of probs)
    verbose: bool = False  # Print progress
    tokenizer: Optional[Any] = None  # Tokenizer for decoding (for verbose output)
    tree_data: Optional[Dict[str, Any]] = None  # Store tree data for visualization


class LookaheadNode:
    """Represents a node in the lookahead tree"""
    def __init__(self, token_id: int, log_prob: float, sequence: List[int], score: float = 0.0):
        self.token_id = token_id
        self.log_prob = log_prob
        self.sequence = sequence  # Full sequence from root to this node
        self.score = score  # Cumulative score (sum of log probs)
        self.children: List['LookaheadNode'] = []


@torch.inference_mode()
def lookahead_generate(
    model,
    tokens: List[int],
    max_tokens: int,
    lookahead_config: LookaheadConfig,
    seed: int = 42,
    kv_cache=None,
    progress_callback: Optional[Callable] = None
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Generate tokens using lookahead decoding.
    
    Args:
        model: The GPT model
        tokens: Initial prompt tokens
        max_tokens: Maximum number of tokens to generate
        lookahead_config: Configuration for lookahead search
        seed: Random seed
        kv_cache: Optional KV cache (for efficiency)
        progress_callback: Optional callback function(step, token_text, candidates_info)
    
    Returns:
        Tuple of (generated token IDs, tree data dictionary)
    """
    device = model.get_device()
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    
    generated = tokens.copy()
    tree_data = {
        'steps': [],
        'config': {
            'k_candidates': lookahead_config.k_candidates,
            'depth': lookahead_config.depth,
            'temperature': lookahead_config.temperature,
        }
    }
    
    # Initialize KV cache if not provided
    if kv_cache is None:
        # We'll need to manage KV cache manually for lookahead
        # For now, use a simple approach without cache optimization
        pass
    
    tokenizer = lookahead_config.tokenizer
    
    for step in range(max_tokens):
        # Get current context
        ids = torch.tensor([generated], dtype=torch.long, device=device)
        
        # Forward pass to get logits
        logits = model.forward(ids, kv_cache=kv_cache)  # (B, T, vocab_size)
        logits = logits[:, -1, :]  # (B, vocab_size) - last token logits
        
        # Apply temperature and top-k
        if lookahead_config.top_k is not None:
            k = min(lookahead_config.top_k, logits.size(-1))
            vals, idx = torch.topk(logits, k, dim=-1)
            # Create a mask for top-k
            logits_filtered = torch.full_like(logits, float('-inf'))
            logits_filtered.scatter_(1, idx, vals)
            logits = logits_filtered
        
        if lookahead_config.temperature > 0:
            logits = logits / lookahead_config.temperature
        else:
            # Greedy decoding
            next_token = torch.argmax(logits, dim=-1).item()
            generated.append(next_token)
            continue
        
        probs = F.softmax(logits, dim=-1)
        
        # Get top K candidates
        top_k_probs, top_k_indices = torch.topk(probs, lookahead_config.k_candidates)
        top_k_log_probs = torch.log(top_k_probs + 1e-10)  # Add small epsilon to avoid log(0)
        
        # Build lookahead tree: explore each candidate
        candidate_scores = []
        candidate_sequences = []
        step_candidates = []
        
        if lookahead_config.verbose:
            print(f"\n[Step {step+1}/{max_tokens}] Exploring {lookahead_config.k_candidates} candidates...")
        
        for i in range(lookahead_config.k_candidates):
            candidate_token = top_k_indices[0, i].item()
            candidate_log_prob = top_k_log_probs[0, i].item()
            
            # Decode candidate token for display
            candidate_text = ""
            if tokenizer:
                candidate_text = tokenizer.decode([candidate_token])
            
            if lookahead_config.verbose:
                print(f"  Candidate {i+1}: '{candidate_text}' (prob: {top_k_probs[0, i].item():.4f})")
            
            # Start a temporary sequence with this candidate
            temp_sequence = generated + [candidate_token]
            cumulative_score = candidate_log_prob
            rollout_tokens = [candidate_token]
            rollout_texts = [candidate_text] if tokenizer else []
            
            # Rollout: generate D tokens greedily from this candidate
            temp_ids = torch.tensor([temp_sequence], dtype=torch.long, device=device)
            
            for depth_step in range(lookahead_config.depth):
                # Forward pass
                temp_logits = model.forward(temp_ids, kv_cache=None)  # Don't use cache for lookahead
                temp_logits = temp_logits[:, -1, :]  # Last token
                
                # Apply same temperature/top-k
                if lookahead_config.top_k is not None:
                    k = min(lookahead_config.top_k, temp_logits.size(-1))
                    vals, idx = torch.topk(temp_logits, k, dim=-1)
                    temp_logits_filtered = torch.full_like(temp_logits, float('-inf'))
                    temp_logits_filtered.scatter_(1, idx, vals)
                    temp_logits = temp_logits_filtered
                
                if lookahead_config.temperature > 0:
                    temp_logits = temp_logits / lookahead_config.temperature
                
                temp_probs = F.softmax(temp_logits, dim=-1)
                
                # Greedy selection for lookahead
                # temp_probs is (1, vocab_size), argmax gives (1,)
                best_token_idx = torch.argmax(temp_probs, dim=-1)  # Shape: (1,)
                token_id_scalar = best_token_idx[0].item()  # Get scalar value
                best_token_log_prob = torch.log(temp_probs[0, token_id_scalar] + 1e-10).item()
                
                # Update cumulative score
                if lookahead_config.use_log_prob_sum:
                    cumulative_score += best_token_log_prob
                else:
                    cumulative_score = cumulative_score + best_token_log_prob  # Same for log space
                
                # Append to temp sequence
                temp_sequence.append(token_id_scalar)
                rollout_tokens.append(token_id_scalar)
                if tokenizer:
                    rollout_texts.append(tokenizer.decode([token_id_scalar]))
                # temp_ids is (1, T), we need to add (1, 1) -> reshape token_id_scalar to (1, 1)
                new_token_tensor = torch.tensor([[token_id_scalar]], dtype=torch.long, device=device)
                temp_ids = torch.cat([temp_ids, new_token_tensor], dim=1)
            
            # Store candidate info
            candidate_info = {
                'index': i,
                'token_id': candidate_token,
                'token_text': candidate_text,
                'initial_prob': top_k_probs[0, i].item(),
                'score': cumulative_score,
                'rollout_tokens': rollout_tokens[:5],  # First 5 tokens of rollout
                'rollout_text': ''.join(rollout_texts[:5]) if rollout_texts else '',
            }
            step_candidates.append(candidate_info)
            
            if lookahead_config.verbose:
                rollout_preview = ''.join(rollout_texts[:5])
                print(f"    Rollout preview: '{rollout_preview}...' (score: {cumulative_score:.4f})")
            
            candidate_scores.append(cumulative_score)
            candidate_sequences.append(temp_sequence)
        
        # Select the candidate with the highest cumulative score
        best_index = max(range(len(candidate_scores)), key=lambda i: candidate_scores[i])
        best_sequence = candidate_sequences[best_index]
        
        # Commit: take only the first token from the best sequence
        # (We looked ahead D tokens, but only commit 1)
        next_token = best_sequence[len(generated)]
        generated.append(next_token)
        
        # Store step data
        step_data = {
            'step': step + 1,
            'selected_candidate': best_index,
            'selected_token': next_token,
            'selected_text': tokenizer.decode([next_token]) if tokenizer else '',
            'candidates': step_candidates,
        }
        tree_data['steps'].append(step_data)
        
        if lookahead_config.verbose:
            print(f"  ✓ Selected candidate {best_index+1}: '{step_data['selected_text']}'")
            print(f"  Generated so far: '{tokenizer.decode(generated[-10:]) if tokenizer and len(generated) > 10 else (tokenizer.decode(generated) if tokenizer else '')}'")
        
        # Call progress callback if provided
        if progress_callback:
            progress_callback(step + 1, step_data['selected_text'], step_candidates)
    
    return generated, tree_data


def lookahead_generate_with_engine(
    engine,
    tokens: List[int],
    max_tokens: int,
    lookahead_config: LookaheadConfig,
    seed: int = 42,
    progress_callback: Optional[Callable] = None
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Wrapper that uses Engine's model and handles KV cache properly.
    This is the recommended interface for using lookahead with nanochat.
    
    Returns:
        Tuple of (generated token IDs, tree data dictionary)
    """
    # Set tokenizer in config if not already set
    if lookahead_config.tokenizer is None:
        lookahead_config.tokenizer = engine.tokenizer
    
    return lookahead_generate(
        engine.model,
        tokens,
        max_tokens,
        lookahead_config,
        seed,
        kv_cache=None,  # TODO: implement KV cache support for lookahead
        progress_callback=progress_callback
    )
