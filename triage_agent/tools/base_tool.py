"""
tools/base_tool.py

This module defines the abstract base interface that all agent tools must implement.

Design Philosophy:
-----------------
Tools are the hands and eyes of agents - they perform concrete actions and gather
information. By enforcing a common interface, we achieve:

1. Interchangeability: Agents can use any tool through the same interface
2. Testability: Can mock tools easily, test tools in isolation
3. Composability: Can build tool chains and pipelines
4. Extensibility: New tools plug in without changing agent code

The Tool interface is intentionally minimal:
- Exposes a `name` for identification
- Provides a `run()` method that operates on AgentState
- Returns results in a flexible format (Any type)

This simplicity enables maximum flexibility while enforcing just enough
structure to make tools predictable and reliable.

Tool Flow:
---------
1. Agent identifies need for tool (e.g., "need to parse resume")
2. Agent retrieves appropriate tool by name
3. Agent calls tool.run(state) 
4. Tool reads necessary data from state
5. Tool performs its operation
6. Tool returns results (may also update state)
7. Agent incorporates results into reasoning

Examples of tools that implement this interface:
- ResumeParserTool: Extracts structured data from resume documents
- WebSearchTool: Searches for information about candidates
- DatabaseTool: Queries candidate database
- ValidationTool: Validates data against schemas
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List
from agent.agent_state import AgentState


class Tool(ABC):
    """
    Abstract base class for all agent tools.
    
    This class establishes the contract that all tools must fulfill. Any class
    that inherits from Tool must implement the `run()` method.
    
    Philosophy:
    ----------
    Tools are stateless operators that transform or augment AgentState. They
    should be:
    - Pure-ish: Same inputs → same outputs (when possible)
    - Side-effect aware: If they have side effects (API calls, DB writes), 
      these should be explicit and logged
    - Error-tolerant: Should handle failures gracefully and report them
    
    Tools should NOT:
    - Make policy decisions (that's the agent's job)
    - Know about business rules (they're generic utilities)
    - Maintain internal state between runs (use AgentState instead)
    
    Attributes:
    ----------
    name : str
        Human-readable identifier for the tool. Used for:
        - Logging and debugging
        - Tool selection/routing
        - Error messages
        
        Convention: Use snake_case, be descriptive
        Examples: "resume_parser", "linkedin_scraper", "skills_validator"
    
    Class Design:
    ------------
    This is an Abstract Base Class (ABC) with an abstract method, which means:
    - Cannot instantiate Tool directly (must subclass)
    - All subclasses MUST implement run()
    - Python will raise TypeError if run() is not implemented
    
    This enforcement at the language level prevents incomplete tool implementations.
    """
    
    # Class-level attribute that must be set by subclasses
    # Type hint indicates this must be a string
    name: str
    
    @abstractmethod
    def run(self, state: AgentState) -> Any:
        """
        Execute the tool's primary operation.
        
        This is the main entry point for tool execution. The agent calls this
        method, passing in the current state, and expects either:
        1. A return value containing results
        2. Modifications to the state object (passed by reference)
        3. Both
        
        Parameters:
        ----------
        state : AgentState
            The current agent state containing all context needed for execution.
            Tools read from state to get inputs and may write to state to record
            results.
            
            Common patterns:
            - Read from state.data: Get candidate info, documents, etc.
            - Write to state.data: Store parsed results, enrichment data
            - Read from state.context: Get configuration, API keys, settings
            - Append to state.history: Log tool execution for audit trail
        
        Returns:
        -------
        Any
            Return type is intentionally flexible to accommodate different tools:
            
            - Structured data (dict, dataclass): Parsed resume, validation results
            - Primitive types (str, int, bool): Simple lookups, checks
            - Collections (list, set): Search results, candidate matches
            - None: Tool updates state in-place, no return needed
            - Custom objects: Domain-specific results
            
            Return type flexibility enables tools to be purpose-built without
            forcing them into a rigid output schema.
        
        Raises:
        ------
        ToolExecutionError (recommended):
            When tool encounters an error it cannot recover from.
            Subclasses should define domain-specific exceptions.
            
        General exceptions:
            Tools should catch and handle expected errors, but may raise
            unexpected exceptions to be caught by agent error handling.
        
        Implementation Guidelines:
        -------------------------
        When implementing run() in a subclass:
        
        1. VALIDATE INPUTS:
           Check that state contains required data before proceeding.
           Fail fast with clear error messages.
           
           Example:
           if "resume_text" not in state.data:
               raise ValueError("resume_text required in state.data")
        
        2. LOG EXECUTION:
           Record what the tool is doing for debugging and audit.
           
           Example:
           logger.info(f"{self.name} processing candidate {state.data.get('candidate_id')}")
        
        3. HANDLE ERRORS GRACEFULLY:
           Catch expected errors, log them, and either:
           - Return a safe default
           - Raise a descriptive exception
           - Update state with error information
           
           Example:
           try:
               result = external_api_call()
           except APITimeout:
               logger.error(f"{self.name} timeout")
               return {"error": "timeout", "partial_results": []}
        
        4. MAINTAIN STATE CONSISTENCY:
           If you modify state, ensure it remains valid.
           Don't leave state in a broken state if execution fails.
        
        5. DOCUMENT SIDE EFFECTS:
           If tool makes API calls, writes to DB, sends emails, etc.,
           document this in the tool's docstring.
        
        6. RETURN USEFUL DATA:
           Return data in a format that's easy for the agent to use.
           Include metadata like confidence scores, timestamps, sources.
        
        Examples:
        --------
        # Simple tool that returns a value
        class SkillsExtractor(Tool):
            name = "skills_extractor"
            
            def run(self, state: AgentState) -> List[str]:
                resume_text = state.data.get("resume_text", "")
                skills = self._extract_skills(resume_text)
                return skills
        
        # Tool that updates state in-place
        class ResumeEnricher(Tool):
            name = "resume_enricher"
            
            def run(self, state: AgentState) -> None:
                candidate_name = state.data["candidate_name"]
                linkedin_data = self._fetch_linkedin(candidate_name)
                state.data["linkedin_profile"] = linkedin_data
                # No return - state updated directly
        
        # Tool that does both
        class BackgroundChecker(Tool):
            name = "background_checker"
            
            def run(self, state: AgentState) -> Dict[str, Any]:
                results = self._check_background(state.data["candidate_id"])
                state.data["background_check"] = results
                return {
                    "passed": results["status"] == "clear",
                    "confidence": results["confidence"],
                    "issues": results.get("issues", [])
                }
        """
        pass  # Abstract method - must be implemented by subclasses
    
    # -------------------------------------------------------------------------
    # Optional helper methods that subclasses can override
    # -------------------------------------------------------------------------
    
    def validate_state(self, state: AgentState, required_keys: List[str]) -> None:
        """
        Helper method to validate that state contains required data.
        
        Subclasses can call this in run() to ensure prerequisites are met.
        
        Parameters:
        ----------
        state : AgentState
            State to validate
        required_keys : List[str]
            Keys that must exist in state.data
        
        Raises:
        ------
        ValueError: If any required key is missing
        
        Example:
        -------
        def run(self, state: AgentState) -> Any:
            self.validate_state(state, ["resume_text", "candidate_id"])
            # Now safe to access state.data["resume_text"]
        """
        missing_keys = [key for key in required_keys if key not in state.data]
        if missing_keys:
            raise ValueError(
                f"{self.name} requires {missing_keys} in state.data, but they are missing"
            )
    
    def get_config(self, state: AgentState, key: str, default: Any = None) -> Any:
        """
        Helper to safely retrieve configuration from state.
        
        Parameters:
        ----------
        state : AgentState
            State containing configuration
        key : str
            Configuration key to retrieve
        default : Any
            Default value if key not found
        
        Returns:
        -------
        Configuration value or default
        
        Example:
        -------
        def run(self, state: AgentState) -> Any:
            timeout = self.get_config(state, "api_timeout", default=30)
            max_retries = self.get_config(state, "max_retries", default=3)
        """
        return state.context.get(key, default) if hasattr(state, 'context') else default
    
    def __str__(self) -> str:
        """
        String representation for logging and debugging.
        
        Returns the tool's name for easy identification in logs.
        """
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __repr__(self) -> str:
        """
        Developer-friendly representation.
        
        Shows class name and tool name for debugging.
        """
        return f"<{self.__class__.__name__} name='{self.name}'>"


# -----------------------------------------------------------------------------
# Optional: Base exception class for tool errors
# -----------------------------------------------------------------------------

class ToolExecutionError(Exception):
    """
    Base exception for tool execution failures.
    
    Subclasses can define more specific exceptions:
    - ToolValidationError: Invalid inputs
    - ToolTimeoutError: Operation timed out
    - ToolAPIError: External API failure
    
    Attributes:
    ----------
    tool_name : str
        Name of the tool that raised the error
    message : str
        Error description
    details : Dict[str, Any]
        Additional context about the error
    """
    
    def __init__(
        self,
        tool_name: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize tool execution error.
        
        Parameters:
        ----------
        tool_name : str
            Name of the tool that failed
        message : str
            Human-readable error message
        details : Optional[Dict[str, Any]]
            Additional error context
        """
        self.tool_name = tool_name
        self.message = message
        self.details = details or {}
        
        # Construct full error message
        full_message = f"[{tool_name}] {message}"
        if details:
            full_message += f" | Details: {details}"
        
        super().__init__(full_message)


# -----------------------------------------------------------------------------
# Design Notes
# -----------------------------------------------------------------------------
#
# Why use ABC instead of duck typing?
# -----------------------------------
# 1. Explicit contract: Makes expectations clear
# 2. Early error detection: Catches incomplete implementations at class definition
# 3. IDE support: Better autocomplete and type checking
# 4. Documentation: Abstract methods are self-documenting
#
# Why allow Any return type?
# --------------------------
# 1. Flexibility: Different tools have different output types
# 2. Evolution: Can change return types without breaking interface
# 3. Simplicity: Avoids complex generic type parameters
#
# However, in production:
# - Consider using TypeVar for more type safety
# - Document expected return types in docstrings
# - Use type hints in concrete implementations
#
# Why pass entire state instead of specific parameters?
# -----------------------------------------------------
# 1. Extensibility: Tools can access any state data they need
# 2. Context preservation: Tools have full context for decision-making
# 3. Uniformity: All tools follow same calling convention
# 4. Evolution: Can add state fields without changing tool signatures
#
# Trade-off:
# - Makes dependencies less explicit
# - Mitigate by documenting required state keys in tool docstrings
#
# Alternative designs considered:
# ------------------------------
# 1. Typed return values: Tool[InputT, OutputT]
#    Rejected: Too complex, reduces flexibility
#
# 2. Separate read/write methods: read_state(), write_state()
#    Rejected: Overly prescriptive, some tools don't fit this pattern
#
# 3. Async support: async def run()
#    Future consideration: Could add AsyncTool subclass if needed
#
