"""Enhanced Fishbone Diagram Generator with Deep Root Cause Analysis.

BEGINNER'S GUIDE:
----------------
This tool creates a comprehensive "Fishbone Diagram" (Ishikawa Diagram) for 
systematic problem analysis. Think of it as a detective tool that:

1. IDENTIFIES THE PROBLEM (Effect): What went wrong?
2. CATEGORIZES POTENTIAL CAUSES (6M Framework):
   - Man (People): Human factors, skills, training issues
   - Machine: Equipment, tools, technology failures
   - Method: Process, procedures, workflow problems
   - Material: Raw materials, supplies, input quality
   - Measurement: Metrics, KPIs, monitoring gaps
   - Environment: External factors, conditions, culture

3. DIGS DEEPER (Root Cause Analysis): For each cause, asks "Why?" to find
   the underlying reasons - getting to the true root of problems.

EXAMPLE OUTPUT STRUCTURE:
Effect: "Release delays in mobile app"
├── Machine: ["Server crashes"] 
│   └── Why? ["Insufficient memory", "No load balancing"]
├── Method: ["Poor testing"]
│   └── Why? ["Rushed timelines", "No test automation"]
└── (continues for all categories...)

SETUP & USAGE:
1. Install: pip install langgraph langchain-openai python-dotenv
2. Create .env file with: OPENAI_API_KEY=sk-your-key-here
3. Run: python fishbone_enhanced.py
4. Interactive mode: Enter effects, type 'quit' to exit
5. Output: JSON files with complete root cause analysis
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# Load environment variables
load_dotenv()

# Configuration constants
MODEL = "gpt-5-mini"
MAX_CAUSES_PER_CATEGORY = 3
MAX_ROOT_CAUSES_PER_CAUSE = 3
CATEGORIES_6M = [
    "Man (People)",
    "Machine", 
    "Method", 
    "Material", 
    "Measurement", 
    "Environment"
]


class FishboneState(TypedDict):
    """State for enhanced Fishbone with root causes."""
    effect: str
    categories: List[str]
    causes: Dict[str, List[str]]
    root_causes: Dict[str, Dict[str, List[str]]]
    metadata: Dict[str, str]


class FishboneAnalyzer:
    """Enhanced Fishbone Diagram Generator with continuous input support."""
    
    def __init__(self, model: str = MODEL, temperature: float = 0):
        """Initialize the Fishbone Analyzer.
        
        Args:
            model: OpenAI model to use
            temperature: Temperature for LLM responses (0 = deterministic)
        """
        self.model = model
        self.temperature = temperature
        self.llm = self._create_llm()
        self.workflow = self._build_workflow()
    
    def _create_llm(self) -> ChatOpenAI:
        """Create and configure the LLM client."""
        return ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=800,
            timeout=30
        )
    
    def _build_causes_prompt(self, effect: str, categories: List[str]) -> List[Dict[str, str]]:
        """Build prompt for initial cause analysis.
        
        Args:
            effect: The problem/effect to analyze
            categories: List of categories to analyze
            
        Returns:
            List of message dictionaries for the LLM
        """
        system_message = (
            f"You are a Root Cause Analysis expert. Return only JSON. "
            f"Maximum {MAX_CAUSES_PER_CATEGORY} causes per category. "
            f"Each cause should be 5 words or less."
        )
        
        user_message = (
            f'Effect: {effect}\n'
            f'Categories: {", ".join(categories)}\n\n'
            f'Return JSON in this exact format:\n'
            f'{{"effect": "{effect}", '
            f'"causes": {{"Category": ["cause1", "cause2"]}}, '
            f'"metadata": {{"method": "Fishbone"}}}}'
        )
        
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    
    def _build_root_causes_prompt(self, causes: Dict[str, List[str]]) -> List[Dict[str, str]]:
        """Build batched prompt for root cause analysis.
        
        Args:
            causes: Dictionary of categories and their causes
            
        Returns:
            List of message dictionaries for the LLM
        """
        system_message = (
            f"Perform deep root cause analysis. Return only JSON. "
            f"Provide {MAX_ROOT_CAUSES_PER_CAUSE} reasons per cause. "
            f"Each reason should be 8 words or less."
        )
        
        # Build batched request for all causes
        cause_items = []
        for category, cause_list in causes.items():
            for cause in cause_list:
                cause_items.append(f'"{category}:{cause}"')
        
        user_message = (
            f'Analyze why these causes occur:\n'
            f'[{", ".join(cause_items)}]\n\n'
            f'Return JSON format: '
            f'{{"Category:cause": ["why1", "why2", "why3"]}}'
        )
        
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    
    def _parse_json_safely(self, text: str, default: Optional[Dict] = None) -> Dict:
        """Safely parse JSON response from LLM.
        
        Args:
            text: Raw text response from LLM
            default: Default value if parsing fails
            
        Returns:
            Parsed JSON dictionary or default value
        """
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return default or {
                "error": "JSON parsing failed",
                "raw_text": text[:100]
            }
    
    def _clean_and_limit_causes(
        self, 
        data: Dict, 
        categories: List[str], 
        max_items: int
    ) -> Dict[str, List[str]]:
        """Clean and limit the number of causes per category.
        
        Args:
            data: Raw data from LLM response
            categories: List of valid categories
            max_items: Maximum items per category
            
        Returns:
            Cleaned dictionary of causes
        """
        if not isinstance(data, dict):
            return {cat: [] for cat in categories}
        
        causes = data.get("causes", {})
        cleaned_causes = {}
        
        for category in categories:
            category_causes = causes.get(category, [])
            
            # Ensure it's a list
            if not isinstance(category_causes, list):
                category_causes = [str(category_causes)] if category_causes else []
            
            # Clean and limit
            cleaned_causes[category] = [
                cause.strip() 
                for cause in category_causes 
                if cause and cause.strip()
            ][:max_items]
        
        return cleaned_causes
    
    def _analyze_causes_node(self, state: FishboneState) -> FishboneState:
        """Generate initial causes for the effect."""
        prompt = self._build_causes_prompt(state["effect"], state["categories"])
        response = self.llm.invoke(prompt)
        data = self._parse_json_safely(response.content)
        
        state["causes"] = self._clean_and_limit_causes(
            data, 
            state["categories"], 
            MAX_CAUSES_PER_CATEGORY
        )
        state["metadata"] = {
            "method": "Fishbone",
            "model": self.model,
            "timestamp": datetime.now().isoformat()
        }
        
        return state
    
    def _analyze_root_causes_node(self, state: FishboneState) -> FishboneState:
        """Analyze root causes for all identified causes."""
        if not any(state["causes"].values()):
            state["root_causes"] = {}
            return state
        
        prompt = self._build_root_causes_prompt(state["causes"])
        response = self.llm.invoke(prompt)
        data = self._parse_json_safely(response.content, {})
        
        # Parse batched response and organize by category
        state["root_causes"] = {}
        for category, cause_list in state["causes"].items():
            state["root_causes"][category] = {}
            
            for cause in cause_list:
                key = f"{category}:{cause}"
                
                if key in data and isinstance(data[key], list):
                    state["root_causes"][category][cause] = [
                        reason.strip() 
                        for reason in data[key][:MAX_ROOT_CAUSES_PER_CAUSE]
                        if reason and reason.strip()
                    ]
                else:
                    state["root_causes"][category][cause] = ["Analysis pending"]
        
        return state
    
    def _build_workflow(self) -> StateGraph:
        """Build the analysis workflow graph."""
        graph = StateGraph(FishboneState)
        
        # Add nodes
        graph.add_node("analyze_causes", self._analyze_causes_node)
        graph.add_node("analyze_root_causes", self._analyze_root_causes_node)
        
        # Define workflow
        graph.set_entry_point("analyze_causes")
        graph.add_edge("analyze_causes", "analyze_root_causes")
        graph.add_edge("analyze_root_causes", END)
        
        return graph.compile()
    
    def analyze_effect(
        self, 
        effect: str, 
        categories: Optional[List[str]] = None,
        output_file: Optional[str] = None
    ) -> Dict:
        """Perform complete Fishbone analysis for a given effect.
        
        Args:
            effect: The problem/effect to analyze
            categories: List of categories (defaults to 6M framework)
            output_file: Optional output file path
            
        Returns:
            Complete analysis results
        """
        if not effect.strip():
            raise ValueError("Effect cannot be empty")
        
        # Use default categories if none provided
        if categories is None:
            categories = CATEGORIES_6M.copy()
        
        # Execute workflow
        result = self.workflow.invoke({
            "effect": effect.strip(),
            "categories": categories,
            "causes": {},
            "root_causes": {},
            "metadata": {}
        })
        
        # Prepare output
        output_data = {
            "effect": result["effect"],
            "causes": result["causes"],
            "root_causes": result["root_causes"],
            "metadata": result["metadata"]
        }
        
        # Save to file if specified
        if output_file:
            self._save_to_file(output_data, output_file)
        
        return output_data
    
    def _save_to_file(self, data: Dict, file_path: str) -> None:
        """Save analysis results to a JSON file.
        
        Args:
            data: Analysis results to save
            file_path: Path to save the file
        """
        try:
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=2, ensure_ascii=False)
            print(f"Analysis saved to: {file_path}")
        except IOError as error:
            print(f"Error saving file: {error}")
    
    def display_results(self, results: Dict) -> None:
        """Display analysis results in a formatted way.
        
        Args:
            results: Analysis results to display
        """
        print("\n" + "="*80)
        print(f"FISHBONE ANALYSIS: {results['effect']}")
        print("="*80)
        
        for category, causes in results["causes"].items():
            if causes:
                print(f"\n📁 {category}:")
                for cause in causes:
                    print(f"   ├── {cause}")
                    
                    # Show root causes if available
                    root_causes = results["root_causes"].get(category, {}).get(cause, [])
                    for i, root_cause in enumerate(root_causes):
                        connector = "└──" if i == len(root_causes) - 1 else "├──"
                        print(f"   │   {connector} Why? {root_cause}")
        
        print(f"\nAnalysis completed at: {results['metadata'].get('timestamp', 'Unknown')}")
        print("-"*80)


def get_user_input() -> str:
    """Get effect input from user with validation."""
    while True:
        effect = input("\nEnter the effect/problem to analyze (or 'quit' to exit): ").strip()
        
        if effect.lower() in ['quit', 'exit', 'q']:
            return ""
        
        if effect:
            return effect
        
        print("Please enter a valid effect/problem description.")


def main():
    """Main function for interactive Fishbone analysis."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set OPENAI_API_KEY in your .env file")
        sys.exit(1)
    
    print("Enhanced Fishbone Diagram Generator")
    print("=" * 50)
    print("This tool helps you perform systematic root cause analysis.")
    print("Enter effects/problems to analyze, or 'quit' to exit.")
    
    # Initialize analyzer
    try:
        analyzer = FishboneAnalyzer()
    except Exception as error:
        print(f"Error initializing analyzer: {error}")
        sys.exit(1)
    
    analysis_count = 0
    
    # Main interaction loop
    while True:
        effect = get_user_input()
        
        if not effect:  # User wants to quit
            break
        
        try:
            print(f"\nAnalyzing: {effect}")
            print("Please wait...")
            
            # Generate timestamp-based filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"fishbone_analysis_{timestamp}.json"
            
            # Perform analysis
            results = analyzer.analyze_effect(effect, output_file=output_file)
            
            # Display results
            analyzer.display_results(results)
            
            analysis_count += 1
            
        except KeyboardInterrupt:
            print("\n\nAnalysis interrupted by user.")
            break
        except Exception as error:
            print(f"Error during analysis: {error}")
            continue
    
    print(f"\nSession completed. Total analyses performed: {analysis_count}")
    print("Thank you for using the Fishbone Diagram Generator!")


if __name__ == "__main__":
    main()