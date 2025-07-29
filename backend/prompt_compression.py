#!/usr/bin/env python3
"""
Advanced Text-to-Prompt Compression Module
Compresses and optimizes all prompt engineering techniques while maintaining effectiveness
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)

class CompressionLevel(Enum):
    """Compression levels for different use cases"""
    MEDIUM = "medium"    # 30-40% reduction, slight quality loss

@dataclass
class CompressionResult:
    """Results of prompt compression"""
    original_prompt: str
    compressed_prompt: str
    original_length: int
    compressed_length: int
    compression_ratio: float
    technique_used: str
    compression_level: CompressionLevel
    metadata: Dict[str, Any]

class PromptCompressor:
    """
    Advanced prompt compression engine for all prompting techniques
    """
    
    def __init__(self):
        self.compression_stats = {}
        self.technique_patterns = self._load_technique_patterns()
        self.compression_rules = self._load_compression_rules()
        
    def _load_technique_patterns(self) -> Dict[str, Dict]:
        """Load patterns specific to each prompting technique"""
        return {
            "contrastive": {
                "markers": ["✅", "❌", "GOOD APPROACH", "POOR APPROACH", "Good:", "Bad:"],
                "structure_keywords": ["contrast", "example", "approach", "method"],
                "preserve_sections": ["✅", "❌", "Good Approach", "Poor Approach"]
            },
            "few_shot": {
                "markers": ["Example", "Question:", "Answer:", "Input:", "Output:"],
                "structure_keywords": ["example", "demonstration", "pattern", "style"],
                "preserve_sections": ["Example", "Question", "Answer"]
            },
            "react": {
                "markers": ["Thought:", "Action:", "Observation:", "Plan:", "Reflection:"],
                "structure_keywords": ["reasoning", "action", "observation", "plan"],
                "preserve_sections": ["Thought", "Action", "Observation"]
            },
            "auto_cot": {
                "markers": ["Step", "Reasoning", "Chain", "Process", "Analysis"],
                "structure_keywords": ["step", "reasoning", "chain", "process", "analysis"],
                "preserve_sections": ["Step", "Reasoning", "Analysis"]
            },
            "program_of_thought": {
                "markers": ["```python", "```", "def ", "import ", "# "],
                "structure_keywords": ["code", "function", "algorithm", "implementation"],
                "preserve_sections": ["```python", "def ", "import"]
            }
        }
    
    def _load_compression_rules(self) -> Dict[CompressionLevel, Dict]:
        """Load compression rules for different levels"""
        return {
            CompressionLevel.MEDIUM: {
                "remove_redundant_phrases": True,
                "compress_examples": True,
                "shorten_explanations": True,
                "preserve_structure": True,
                "target_reduction": 0.35
            }
        }
    
    def detect_technique(self, prompt: str) -> str:
        """Detect which prompting technique is being used"""
        scores = {}
        
        for technique, patterns in self.technique_patterns.items():
            score = 0
            for marker in patterns["markers"]:
                score += prompt.count(marker)
            for keyword in patterns["structure_keywords"]:
                score += prompt.lower().count(keyword.lower())
            scores[technique] = score
        
        return max(scores, key=scores.get) if scores else "standard"
    
    def compress_prompt(
        self, 
        prompt: str, 
        technique: str = None,
        level: CompressionLevel = CompressionLevel.MEDIUM,
        preserve_quality: bool = True
    ) -> CompressionResult:
        """
        Main compression function that applies technique-specific optimization
        """
        original_length = len(prompt)
        
        if technique is None:
            technique = self.detect_technique(prompt)
        
        logger.info(f"Compressing {technique} prompt with {level.value} compression")
        
        # Apply compression based on technique
        if technique == "contrastive":
            compressed = self._compress_contrastive(prompt, level)
        elif technique == "few_shot":
            compressed = self._compress_few_shot(prompt, level)
        elif technique == "react":
            compressed = self._compress_react(prompt, level)
        elif technique == "auto_cot":
            compressed = self._compress_auto_cot(prompt, level)
        elif technique == "program_of_thought":
            compressed = self._compress_pot(prompt, level)
        else:
            compressed = self._compress_standard(prompt, level)
        
        compressed_length = len(compressed)
        compression_ratio = (original_length - compressed_length) / original_length
        
        result = CompressionResult(
            original_prompt=prompt,
            compressed_prompt=compressed,
            original_length=original_length,
            compressed_length=compressed_length,
            compression_ratio=compression_ratio,
            technique_used=technique,
            compression_level=level,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "compression_methods_applied": self._get_applied_methods(level)
            }
        )
        
        self._update_stats(result)
        return result
    
    def _compress_contrastive(self, prompt: str, level: CompressionLevel) -> str:
        """Compress contrastive prompting while preserving good/bad examples"""
        compressed = prompt
        
        # Preserve the core structure markers
        preserve_patterns = ["✅", "❌", "GOOD APPROACH", "POOR APPROACH"]
        
        if level.value in ["medium", "heavy", "extreme"]:
            # Compress explanatory text between examples
            compressed = re.sub(
                r'(\*\*INSTRUCTION\*\*:.*?)\n\n(.*?)\n\n(\*\*QUESTION\*\*:)',
                r'\1\n\3',
                compressed,
                flags=re.DOTALL
            )
            
            # Shorten verbose explanations
            compressed = re.sub(
                r'(✅.*?:)\s*([^❌]{100,}?)(\n)',
                lambda m: f"{m.group(1)} {self._compress_text(m.group(2), 0.3)}{m.group(3)}",
                compressed,
                flags=re.DOTALL
            )
        
        if level.value in ["heavy", "extreme"]:
            # More aggressive compression
            compressed = self._remove_redundant_phrases(compressed)
            compressed = self._compress_instruction_sections(compressed)
        
        return compressed
    
    def _compress_few_shot(self, prompt: str, level: CompressionLevel) -> str:
        """Compress few-shot prompting while preserving example structure"""
        compressed = prompt
        
        if level.value in ["medium", "heavy", "extreme"]:
            # Compress example descriptions but keep core examples
            compressed = re.sub(
                r'(\*\*Example \d+:\*\*\n)(.*?)(Question:|Answer:)',
                lambda m: f"{m.group(1)}{self._compress_text(m.group(2), 0.4)}{m.group(3)}",
                compressed,
                flags=re.DOTALL
            )
            
            # Shorten explanatory sections
            compressed = re.sub(
                r'(examples of high-quality responses.*?)(Following the style)',
                r'Reference examples for style and depth. \2',
                compressed,
                flags=re.DOTALL | re.IGNORECASE
            )
        
        if level.value in ["heavy", "extreme"]:
            # Reduce number of examples (keep only the best one)
            if level == CompressionLevel.EXTREME:
                compressed = re.sub(
                    r'(\*\*Example 2:\*\*.*?)(\*\*NEW QUESTION\*\*)',
                    r'\2',
                    compressed,
                    flags=re.DOTALL
                )
        
        return compressed
    
    def _compress_react(self, prompt: str, level: CompressionLevel) -> str:
        """Compress ReAct prompting while preserving reasoning structure"""
        compressed = prompt
        
        if level.value in ["medium", "heavy", "extreme"]:
            # Compress action descriptions
            compressed = re.sub(
                r'(\*\*Available Actions\*\*:)(.*?)(\*\*Question\*\*)',
                r'\1 analyze_context, search_knowledge, code_example, verify_solution, synthesize_answer\n\n\3',
                compressed,
                flags=re.DOTALL
            )
            
            # Shorten format explanations
            compressed = re.sub(
                r'(\*\*Format\*\*:\n)(.*?)(```)',
                r'\1\3',
                compressed,
                flags=re.DOTALL
            )
        
        if level.value in ["heavy", "extreme"]:
            # Compress the example reasoning chain
            compressed = re.sub(
                r'(Observation: Looking at the question.*?)(Continue the ReAct process)',
                r'Observation: [Analyze question] \2',
                compressed,
                flags=re.DOTALL
            )
        
        return compressed
    
    def _compress_auto_cot(self, prompt: str, level: CompressionLevel) -> str:
        """Compress Auto-CoT prompting while preserving reasoning steps"""
        compressed = prompt
        
        if level.value in ["medium", "heavy", "extreme"]:
            # Compress step descriptions
            compressed = re.sub(
                r'(\*\*Step \d+:.*?\*\*\n)(- .*?\n)*',
                lambda m: f"{m.group(1)}{self._compress_bullet_points(m.group(0))}",
                compressed,
                flags=re.DOTALL
            )
        
        if level.value in ["heavy", "extreme"]:
            # Merge similar steps
            compressed = re.sub(
                r'(\*\*Step 4: Solution Development\*\*.*?)(\*\*Step 5: Verification)',
                r'\2',
                compressed,
                flags=re.DOTALL
            )
        
        return compressed
    
    def _compress_pot(self, prompt: str, level: CompressionLevel) -> str:
        """Compress Program-of-Thought prompting while preserving code structure"""
        compressed = prompt
        
        if level.value in ["medium", "heavy", "extreme"]:
            # Compress code comments but preserve structure
            compressed = re.sub(
                r'(# .*?)\n',
                lambda m: f"# {self._compress_text(m.group(1)[2:], 0.5)}\n" if len(m.group(1)) > 20 else m.group(0),
                compressed
            )
            
            # Compress explanatory sections
            compressed = re.sub(
                r'(## \d+\. .*?\n)(.*?)(```)',
                lambda m: f"{m.group(1)}{self._compress_text(m.group(2), 0.4)}{m.group(3)}",
                compressed,
                flags=re.DOTALL
            )
        
        if level.value in ["heavy", "extreme"]:
            # Remove some optional sections
            compressed = re.sub(
                r'(## 5\. Alternative Approaches.*?)(## 6\. Practical Usage)',
                r'\2',
                compressed,
                flags=re.DOTALL
            )
        
        return compressed
    
    def _compress_standard(self, prompt: str, level: CompressionLevel) -> str:
        """Compress standard prompts"""
        compressed = prompt
        
        if level.value in ["medium", "heavy", "extreme"]:
            compressed = self._remove_redundant_phrases(compressed)
            compressed = self._compress_instruction_sections(compressed)
        
        if level.value in ["heavy", "extreme"]:
            compressed = self._aggressive_compression(compressed)
        
        return compressed
    
    def _compress_text(self, text: str, ratio: float) -> str:
        """Compress arbitrary text by the given ratio"""
        sentences = re.split(r'[.!?]\s+', text)
        target_sentences = max(1, int(len(sentences) * (1 - ratio)))
        
        # Keep the most important sentences (first and last, plus some middle ones)
        if len(sentences) <= target_sentences:
            return text
        
        compressed_sentences = []
        if target_sentences >= 1:
            compressed_sentences.append(sentences[0])  # First sentence
        if target_sentences >= 2:
            compressed_sentences.append(sentences[-1])  # Last sentence
        if target_sentences > 2:
            # Add some middle sentences
            step = len(sentences) // (target_sentences - 2)
            for i in range(step, len(sentences) - 1, step):
                if len(compressed_sentences) < target_sentences:
                    compressed_sentences.append(sentences[i])
        
        return '. '.join(compressed_sentences) + ('.' if not compressed_sentences[-1].endswith('.') else '')
    
    def _remove_redundant_phrases(self, text: str) -> str:
        """Remove commonly redundant phrases"""
        redundant_patterns = [
            (r'\b(please note that|it is important to note|it should be noted)\b', ''),
            (r'\b(in order to|so as to)\b', 'to'),
            (r'\b(due to the fact that)\b', 'because'),
            (r'\b(at this point in time|at the present time)\b', 'now'),
            (r'\b(in the event that)\b', 'if'),
            (r'\b(make use of)\b', 'use'),
            (r'\b(in the vicinity of)\b', 'near'),
            (r'\b(a large number of)\b', 'many'),
            (r'\s+', ' '),  # Multiple spaces to single space
        ]
        
        compressed = text
        for pattern, replacement in redundant_patterns:
            compressed = re.sub(pattern, replacement, compressed, flags=re.IGNORECASE)
        
        return compressed.strip()
    
    def _compress_instruction_sections(self, text: str) -> str:
        """Compress lengthy instruction sections"""
        # Compress bullet points
        text = re.sub(
            r'(\d+\.\s+)(.*?)\n',
            lambda m: f"{m.group(1)}{self._compress_text(m.group(2), 0.3)}\n" if len(m.group(2)) > 50 else m.group(0),
            text
        )
        
        return text
    
    def _compress_bullet_points(self, text: str) -> str:
        """Compress bullet point lists"""
        lines = text.split('\n')
        compressed_lines = []
        
        for line in lines:
            if line.strip().startswith('-') and len(line) > 60:
                # Compress long bullet points
                bullet_text = line.strip()[2:]  # Remove "- "
                compressed_bullet = self._compress_text(bullet_text, 0.4)
                compressed_lines.append(f"- {compressed_bullet}")
            else:
                compressed_lines.append(line)
        
        return '\n'.join(compressed_lines)
    
    def _aggressive_compression(self, text: str) -> str:
        """Apply aggressive compression techniques"""
        # Remove filler words and phrases
        filler_patterns = [
            r'\b(very|really|quite|rather|somewhat|basically|essentially|actually|literally)\b',
            r'\b(as you can see|as mentioned|as stated|clearly|obviously|of course)\b',
            r'\b(furthermore|moreover|additionally|in addition|also)\b(?=.*\b(furthermore|moreover|additionally|in addition|also)\b)',
        ]
        
        compressed = text
        for pattern in filler_patterns:
            compressed = re.sub(pattern, '', compressed, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        compressed = re.sub(r'\s+', ' ', compressed)
        compressed = re.sub(r'\n\s*\n', '\n\n', compressed)
        
        return compressed.strip()
    
    
    def _get_applied_methods(self, level: CompressionLevel) -> List[str]:
        """Get list of compression methods applied for this level"""
        rules = self.compression_rules[level]
        methods = []
        
        if rules.get("remove_redundant_phrases"):
            methods.append("redundant_phrase_removal")
        if rules.get("compress_examples"):
            methods.append("example_compression")
        if rules.get("shorten_explanations"):
            methods.append("explanation_shortening")
        if not rules.get("preserve_structure"):
            methods.append("structure_modification")
        
        return methods
    
    
    

# Global instance for easy access
prompt_compressor = PromptCompressor()

# Helper functions for integration
def compress_prompt(
    prompt: str,
    technique: str = None,
    level: CompressionLevel = CompressionLevel.MEDIUM,
    preserve_quality: bool = True
) -> CompressionResult:
    """
    Main function to compress any prompt with the specified technique and level
    """
    return prompt_compressor.compress_prompt(prompt, technique, level, preserve_quality)

def compress_for_api(prompt: str, max_tokens: int = 4000) -> str:
    """
    Compress prompt to fit within API token limits
    """
    # Rough estimate: 1 token ≈ 4 characters
    max_chars = max_tokens * 4
    
    if len(prompt) <= max_chars:
        return prompt
    
    # Only medium compression is available
    result = compress_prompt(prompt, level=CompressionLevel.MEDIUM)
    return result.compressed_prompt

def batch_compress_prompts(prompts: List[Dict]) -> List[CompressionResult]:
    """
    Compress multiple prompts in batch
    """
    results = []
    for prompt_data in prompts:
        prompt = prompt_data["prompt"]
        technique = prompt_data.get("technique")
        level = CompressionLevel(prompt_data.get("level", "medium"))
        
        result = compress_prompt(prompt, technique, level)
        results.append(result)
    
    return results

def get_optimal_compression_level(prompt: str, target_length: int) -> CompressionLevel:
    """
    Determine optimal compression level to reach target length.
    Since only MEDIUM is available, always return that.
    """
    return CompressionLevel.MEDIUM

# Export key classes and functions
__all__ = [
    "PromptCompressor",
    "CompressionLevel", 
    "CompressionResult",
    "compress_prompt",
    "compress_for_api",
    "batch_compress_prompts",
    "get_optimal_compression_level",
    "prompt_compressor"
]
