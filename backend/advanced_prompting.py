#!/usr/bin/env python3
"""
Advanced Prompt Engineering Techniques Module
Implements: Contrastive, Few-Shot, ReAct, Auto-CoT, and Program-of-Thought (PoT) prompting
With integrated text-to-prompt compression capabilities
"""

import re
import json
import random
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

# Import compression functionality
try:
    from prompt_compression import (
        compress_prompt, compress_for_api, CompressionLevel, 
        CompressionResult, prompt_compressor
    )
    COMPRESSION_AVAILABLE = True
except ImportError:
    COMPRESSION_AVAILABLE = False
    print("Warning: Prompt compression module not available")

class AdvancedPromptingEngine:
    """
    Advanced prompting engine implementing sophisticated prompt engineering strategies
    """
    
    def __init__(self):
        self.few_shot_examples = self._load_few_shot_examples()
        self.reasoning_patterns = self._load_reasoning_patterns()
        self.compression_enabled = COMPRESSION_AVAILABLE
        self.default_compression_level = CompressionLevel.MEDIUM if COMPRESSION_AVAILABLE else None
        
    def _load_few_shot_examples(self) -> Dict[str, List[Dict]]:
        """Load domain-specific few-shot examples"""
        return {
            "python_programming": [
                {
                    "input": "How do I create a list comprehension in Python?",
                    "output": "List comprehensions provide a concise way to create lists:\n\n```python\n# Basic syntax: [expression for item in iterable]\nsquares = [x**2 for x in range(10)]\n# Result: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]\n\n# With condition\neven_squares = [x**2 for x in range(10) if x % 2 == 0]\n# Result: [0, 4, 16, 36, 64]\n```"
                },
                {
                    "input": "How do I handle exceptions in Python?",
                    "output": "Use try-except blocks to handle exceptions gracefully:\n\n```python\ntry:\n    result = 10 / int(input('Enter number: '))\n    print(f'Result: {result}')\nexcept ValueError:\n    print('Invalid input - not a number')\nexcept ZeroDivisionError:\n    print('Cannot divide by zero')\nexcept Exception as e:\n    print(f'Unexpected error: {e}')\nelse:\n    print('Operation successful')\nfinally:\n    print('Cleanup code here')\n```"
                }
            ],
            "machine_learning": [
                {
                    "input": "What's the difference between overfitting and underfitting?",
                    "output": "**Overfitting vs Underfitting:**\n\n**Overfitting:**\n- Model learns training data too well (memorizes)\n- High training accuracy, low validation accuracy\n- Solution: Regularization, more data, simpler model\n\n**Underfitting:**\n- Model too simple to capture patterns\n- Low training and validation accuracy\n- Solution: More complex model, more features\n\n**Sweet Spot:** Model that generalizes well to unseen data"
                }
            ],
            "data_analysis": [
                {
                    "input": "How do I clean missing data in pandas?",
                    "output": "Several strategies for handling missing data:\n\n```python\nimport pandas as pd\n\n# Check missing data\ndf.isnull().sum()\n\n# Drop rows with any missing values\ndf.dropna()\n\n# Fill with mean/median/mode\ndf['column'].fillna(df['column'].mean())\n\n# Forward/backward fill\ndf.fillna(method='ffill')  # Forward fill\ndf.fillna(method='bfill')  # Backward fill\n\n# Fill with custom value\ndf.fillna({'column1': 0, 'column2': 'Unknown'})\n```"
                }
            ]
        }
    
    def _load_reasoning_patterns(self) -> Dict[str, str]:
        """Load reasoning patterns for ReAct prompting"""
        return {
            "observation": "I observe that",
            "thought": "I think that",
            "action": "I will",
            "reflection": "Looking back at this",
            "plan": "My plan is to",
            "conclusion": "Therefore"
        }
    
    def contrastive_prompting(self, question: str, context: str = "", mode: str = "Explain", tone: str = "Concise") -> str:
        """
        Contrastive prompting: Show what to do vs what not to do
        """
        
        contrastive_template = f"""You are an expert AI assistant. I will ask you a question, and I want you to provide a comprehensive answer using contrastive examples.

**INSTRUCTION**: For each main point, show both:
✅ **GOOD APPROACH** - What TO do and why it works
❌ **POOR APPROACH** - What NOT to do and why it fails

{f"**CONTEXT**: {context[:800]}..." if context else ""}

**QUESTION**: {question}

**RESPONSE FORMAT** ({tone.lower()} {mode.lower()}):
Provide your answer using this structure:

## Main Concept 1: [Topic]
✅ **Good Approach**: [Explanation of correct method with reasoning]
❌ **Poor Approach**: [Explanation of wrong method and why it fails]

## Main Concept 2: [Topic]  
✅ **Good Approach**: [Explanation of correct method with reasoning]
❌ **Poor Approach**: [Explanation of wrong method and why it fails]

[Continue for additional concepts...]

**BEST PRACTICES SUMMARY**: [Key takeaways highlighting dos and don'ts]

Your detailed contrastive response:
"""
        return contrastive_template
    
    def few_shot_prompting(self, question: str, context: str = "", mode: str = "Explain", tone: str = "Concise") -> str:
        """
        Few-shot prompting: Provide relevant examples before asking for response
        """
        
        # Determine domain from question content
        question_lower = question.lower()
        domain = "general"
        
        if any(word in question_lower for word in ["python", "code", "programming", "function", "variable", "loop"]):
            domain = "python_programming"
        elif any(word in question_lower for word in ["machine learning", "ml", "model", "training", "neural", "overfitting"]):
            domain = "machine_learning"
        elif any(word in question_lower for word in ["data", "pandas", "analysis", "dataset", "clean", "missing"]):
            domain = "data_analysis"
        
        # Get examples for the domain
        examples = self.few_shot_examples.get(domain, [])
        
        # Build few-shot prompt
        examples_text = ""
        if examples:
            examples_text = "\n**EXAMPLES** (for reference style and depth):\n\n"
            for i, example in enumerate(examples[:2], 1):  # Use max 2 examples
                examples_text += f"**Example {i}:**\n"
                examples_text += f"Question: {example['input']}\n"
                examples_text += f"Answer: {example['output']}\n\n"
        
        few_shot_template = f"""You are an expert AI assistant. I will provide examples of high-quality responses, then ask you a new question.

{examples_text}

{f"**CONTEXT FROM DOCUMENTS**: {context[:800]}..." if context else ""}

**NEW QUESTION**: {question}

**YOUR TASK**: 
Follow the style, depth, and structure demonstrated in the examples above. Provide a comprehensive, well-structured answer that:
1. Breaks down the problem systematically like the examples
2. Provides concrete, actionable solutions
3. Includes code examples where relevant (following example format)
4. Explains the reasoning behind recommendations
5. Maintains the {tone.lower()} tone while being thorough
6. Uses the {mode.lower()} approach effectively

Your response (following the example patterns):
"""
        return few_shot_template
    
    def react_prompting(self, question: str, context: str = "", mode: str = "Explain", tone: str = "Concise") -> str:
        """
        ReAct prompting: Reasoning + Acting in an interleaved manner
        """
        
        react_template = f"""You are an AI assistant that thinks step by step and can take actions to solve problems.

**Available Actions**: analyze_context, search_knowledge, code_example, verify_solution, synthesize_answer

**Question**: {question}
{f"**Context**: {context[:600]}..." if context else ""}

**Task**: Use the ReAct framework to {mode.lower()} this question with a {tone.lower()} approach.

**Format**:
```
Thought: [Your reasoning about what you need to do]
Action: [The action you want to take from available actions]
Observation: [What you observe from taking that action]
Thought: [Your reasoning about the observation]
Action: [Next action based on your reasoning]
Observation: [What you observe]
... (continue until you have enough information)
Thought: [Final reasoning synthesis]
Answer: [Your comprehensive final answer]
```

**Start your ReAct process**:

Thought: I need to analyze this question and determine what information and actions are needed to provide a comprehensive {mode.lower()} response.

Action: analyze_context

Observation: Looking at the question "{question}", I can see this is asking about...

[Continue the ReAct process to build your complete response]
"""
        return react_template
    
    def auto_cot_prompting(self, question: str, context: str = "", mode: str = "Explain", tone: str = "Concise") -> str:
        """
        Auto-CoT: Automatically generate chain-of-thought reasoning
        """
        
        auto_cot_template = f"""You are an expert problem solver. For the given question, you will automatically generate a comprehensive step-by-step reasoning process.

**QUESTION**: {question}
{f"**CONTEXT**: {context[:700]}..." if context else ""}

**INSTRUCTIONS**: 
Generate an automatic chain-of-thought that follows this structured reasoning process:

**AUTO-GENERATED REASONING CHAIN**:

**Step 1: Problem Decomposition**
- What exactly is being asked?
- What are the key components I need to address?
- What type of response does this require? ({mode.lower()} approach)

**Step 2: Knowledge Activation**  
- What relevant knowledge do I have about this topic?
- What concepts, principles, or facts apply here?
- Are there any important relationships or dependencies?

**Step 3: Context Integration**
- How does any provided context relate to the question?
- What additional information can I derive or infer?
- What gaps in information do I need to address?

**Step 4: Solution Development**
- What is my systematic approach to answering this?
- What logical steps should I follow?
- How can I ensure completeness and accuracy?

**Step 5: Verification and Synthesis**
- Does my reasoning chain make logical sense?
- Have I addressed all aspects of the question?
- Is my conclusion well-supported by the reasoning?

**DETAILED STEP-BY-STEP RESPONSE** ({tone.lower()} {mode.lower()}):

Following the reasoning chain above, here is my comprehensive answer:

[Your detailed response following the auto-generated reasoning process]
"""
        return auto_cot_template
    
    def program_of_thought_prompting(self, question: str, context: str = "", mode: str = "Explain", tone: str = "Concise") -> str:
        """
        Program-of-Thought (PoT): Generate executable code to solve problems
        """
        
        pot_template = f"""You are an AI that solves problems by writing and explaining executable code.

**QUESTION**: {question}
{f"**CONTEXT**: {context[:600]}..." if context else ""}

**PROGRAM-OF-THOUGHT APPROACH** ({tone.lower()} {mode.lower()}):

## 1. Problem Analysis
```python
# Define the problem in computational terms
problem_statement = \"\"\"
{question}
\"\"\"

# Break down into computational steps
steps = [
    "Step 1: [Identify what needs to be computed]",
    "Step 2: [Determine data structures needed]", 
    "Step 3: [Design algorithm approach]",
    "Step 4: [Implement solution]",
    "Step 5: [Verify and test]"
]
```

## 2. Solution Implementation
```python
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any

def solve_problem():
    \"\"\"
    Main function to solve: {question}
    \"\"\"
    
    # Step 1: Initialize variables and data structures
    # [Your initialization code here]
    
    # Step 2: Core logic implementation
    # [Your main algorithm here]
    
    # Step 3: Process and validate results
    # [Your processing code here]
    
    # Step 4: Return formatted results
    return result

# Execute the solution
if __name__ == "__main__":
    result = solve_problem()
    print(f"Solution: {{result}}")
    
    # Additional analysis or visualization if needed
    # [Additional code here]
```

## 3. Code Explanation
[Detailed explanation of each part of the code and the reasoning behind the approach]

## 4. Testing and Validation
```python
# Test cases to verify the solution
def test_solution():
    # Test case 1: [Description]
    test_input_1 = # [Test data]
    expected_1 = # [Expected result]
    assert solve_problem(test_input_1) == expected_1
    
    # Test case 2: [Description] 
    # [More test cases]
    
    print("All tests passed!")

test_solution()
```

## 5. Alternative Approaches and Optimizations
```python
# Alternative implementation or optimizations
def optimized_solution():
    # [Alternative approach with better performance/readability]
    pass

# Performance comparison if relevant
# [Benchmarking code if applicable]
```

## 6. Practical Usage Example
```python
# Real-world usage example
example_usage = \"\"\"
[Show how to use the solution in practice]
\"\"\"
```

**COMPLETE EXECUTABLE SOLUTION**:
[Provide the final, complete, runnable code solution]
"""
        return pot_template
    
    def detect_best_technique(self, question: str, context: str = "") -> str:
        """
        Automatically detect which prompting technique would be most effective
        """
        question_lower = question.lower()
        
        # Check for programming/code-related questions
        if any(word in question_lower for word in ["code", "program", "algorithm", "implement", "function", "script"]):
            return "program_of_thought"
        
        # Check for comparison/contrast needs
        if any(word in question_lower for word in ["difference", "compare", "contrast", "vs", "versus", "better", "worse"]):
            return "contrastive"
            
        # Check for complex reasoning needs
        if any(word in question_lower for word in ["analyze", "evaluate", "assess", "determine", "investigate"]):
            return "react"
            
        # Check for learning/tutorial needs
        if any(word in question_lower for word in ["how to", "tutorial", "guide", "learn", "teach", "example"]):
            return "few_shot"
            
        # Default to auto-cot for complex explanations
        return "auto_cot"
    
    
    def apply_technique(self, technique: str, question: str, context: str = "", mode: str = "Explain", tone: str = "Concise") -> str:
        """
        Apply the specified prompting technique with optional compression
        """
        technique_map = {
            "contrastive": self.contrastive_prompting,
            "few_shot": self.few_shot_prompting,
            "react": self.react_prompting,
            "auto_cot": self.auto_cot_prompting,
            "program_of_thought": self.program_of_thought_prompting
        }
        
        if technique in technique_map:
            prompt = technique_map[technique](question, context, mode, tone)
        else:
            # Fallback to auto-cot
            prompt = self.auto_cot_prompting(question, context, mode, tone)
        
        return prompt

    
    
    def compress_all_techniques(self, question: str, context: str = "", mode: str = "Explain", 
                               tone: str = "Concise", compression_level: str = "medium") -> Dict[str, Dict]:
        """Generate compressed versions of all techniques for comparison"""
        if not self.compression_enabled:
            return {"error": "Compression not available"}
        
        techniques = ["contrastive", "few_shot", "react", "auto_cot", "program_of_thought"]
        results = {}
        
        for technique in techniques:
            try:
                # Generate original prompt
                original_prompt = self.apply_technique(technique, question, context, mode, tone, 
                                                     use_compression=False)
                
                # Generate compressed prompt
                compressed_prompt = self.apply_technique(technique, question, context, mode, tone, 
                                                       use_compression=True, compression_level=compression_level)
                
                # Get compression result for analysis
                compression_level_enum = CompressionLevel(compression_level)
                compression_result = compress_prompt(original_prompt, technique, compression_level_enum)
                
                results[technique] = {
                    "original_prompt": original_prompt,
                    "compressed_prompt": compressed_prompt,
                    "original_length": len(original_prompt),
                    "compressed_length": len(compressed_prompt),
                    "compression_ratio": compression_result.compression_ratio,
                    "quality_score": compression_result.quality_score,
                    "savings_chars": len(original_prompt) - len(compressed_prompt)
                }
                
            except Exception as e:
                results[technique] = {"error": str(e)}
        
        return results

# Global instance for easy access
advanced_prompting_engine = AdvancedPromptingEngine()

# Helper functions for integration
def get_advanced_prompt(question: str, context: str = "", mode: str = "Explain", tone: str = "Concise", 
                       technique: Optional[str] = None) -> str:
    """
    Main function to get an advanced prompt using specified or auto-detected technique
    """
    if technique is None:
        technique = advanced_prompting_engine.detect_best_technique(question, context)
    
    return advanced_prompting_engine.apply_technique(technique, question, context, mode, tone)

def list_available_techniques() -> List[str]:
    """
    Return list of available prompting techniques
    """
    return ["contrastive", "few_shot", "react", "auto_cot", "program_of_thought"]

def get_technique_description(technique: str) -> str:
    """
    Get description of a specific technique
    """
    descriptions = {
        "contrastive": "Shows good vs bad examples, highlighting what to do and what to avoid",
        "few_shot": "Provides relevant examples before the task to guide response style and quality",
        "react": "Uses interleaved reasoning and action steps to solve problems systematically", 
        "auto_cot": "Automatically generates chain-of-thought reasoning for complex problems",
        "program_of_thought": "Solves problems through executable code with detailed explanations"
    }
    return descriptions.get(technique, "Unknown technique")

def refine_technique_name(technique: str) -> str:
    """
    Map internal technique identifiers to original display names
    """
    name_mapping = {
        "contrastive": "contrastive",
        "few_shot": "few_shot", 
        "react": "react",
        "auto_cot": "auto_cot",
        "program_of_thought": "program_of_thought"
    }
    return name_mapping.get(technique, technique)
