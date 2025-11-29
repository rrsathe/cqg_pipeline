"""
Abstract LLM client for provider-agnostic API calls.

Uses langchain_groq ChatGroq for Groq models with structured output support.
"""

import logging
from typing import Optional, Type, TypeVar, List, Dict, Any
import os

from pydantic import BaseModel

from .config import Config

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class LLMClient:
    """
    LLM client with schema validation using langchain_groq.
    
    Features:
    - Direct Groq API integration via ChatGroq
    - Pydantic schema enforcement with structured output
    - Automatic retry on failures
    - Offline mode support
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None
    ):
        """
        Initialize LLM client.
        
        Args:
            model: Model identifier (e.g., "groq/llama-3.3-70b-versatile")
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
            max_retries: Number of retries for failures
        """
        self.model = model or Config.LLM_MODEL
        self.temperature = temperature or Config.LLM_TEMPERATURE
        self.max_tokens = max_tokens or Config.LLM_MAX_TOKENS
        self.timeout = timeout or Config.LLM_TIMEOUT
        self.max_retries = max_retries or Config.LLM_MAX_RETRIES
        
        # Offline/disabled mode short-circuit
        self.disabled = bool(Config.DISABLE_LLM)

        if self.disabled:
            logger.info("📴 LLM disabled (offline mode)")
            self.client = None
            return

        # Import dependencies
        try:
            from langchain_groq import ChatGroq

            # Extract model name (remove groq/ prefix if present)
            model_name = self.model.replace("groq/", "")
            
            # Create ChatGroq client
            self.client = ChatGroq(
                model=model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                max_retries=self.max_retries,
                groq_api_key=Config.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
            )

            logger.info(f"✅ LLM client initialized: {model_name}")

        except ImportError as e:
            logger.error(f"Required dependencies not installed: {e}")
            logger.error("Install with: pip install langchain-groq")
            raise
    
    def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Optional[T]:
        """
        Generate structured output conforming to Pydantic schema.
        
        Args:
            prompt: User prompt
            response_model: Pydantic model class for output
            system_prompt: Optional system message
            **kwargs: Additional arguments (ignored for compatibility)
        
        Returns:
            Pydantic model instance or None on failure
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        if self.disabled:
            logger.debug("LLM disabled: generate_structured returning None")
            return None

        try:
            # Use structured output with ChatGroq
            structured_llm = self.client.with_structured_output(response_model)
            response = structured_llm.invoke(messages)
            return response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            logger.debug(f"Prompt: {prompt[:200]}...")
            return None
    
    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Generate unstructured text response.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system message
            **kwargs: Additional arguments (ignored for compatibility)
        
        Returns:
            Generated text or None on failure
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        if self.disabled:
            logger.debug("LLM disabled: generate_text returning None")
            return None

        try:
            response = self.client.invoke(messages)
            
            if response and hasattr(response, 'content'):
                return response.content
            
            return None
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            logger.debug(f"Prompt: {prompt[:200]}...")
            return None
    
    def generate_batch(
        self,
        prompts: List[str],
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> List[Optional[T]]:
        """
        Generate structured outputs for batch of prompts.
        
        Args:
            prompts: List of user prompts
            response_model: Pydantic model class
            system_prompt: Optional system message
            **kwargs: Additional arguments
        
        Returns:
            List of Pydantic model instances (None for failures)
        """
        results = []
        
        for prompt in prompts:
            result = self.generate_structured(
                prompt=prompt,
                response_model=response_model,
                system_prompt=system_prompt,
                **kwargs
            )
            results.append(result)
        
        return results
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate API cost for token usage.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        
        Returns:
            Estimated cost in USD
        
        Note: Pricing hardcoded for common models, update as needed
        """
        # Pricing per 1M tokens (as of Nov 2025)
        pricing = {
            "gpt-4o": {"input": 5.0, "output": 15.0},
            "gpt-4o-mini": {"input": 0.15, "output": 0.6},
            "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
            "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
            "gemini/gemini-1.5-flash": {"input": 0.075, "output": 0.3},
            "groq/llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
        }
        
        model_pricing = pricing.get(self.model, {"input": 0.0, "output": 0.0})
        
        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
        
        return input_cost + output_cost
    
    def get_config(self) -> Dict[str, Any]:
        """Get current LLM configuration."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "max_retries": self.max_retries
        }
