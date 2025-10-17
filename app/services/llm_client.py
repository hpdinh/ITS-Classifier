import httpx
from typing import List, Dict, Union, Literal
from core.config import LLMSettings, PromptSettings

class LLMClient:
    """
    Asynchronous client for interacting with a Large Language Model (LLM) API.

    Provides methods for:
      - Chat completions
      - Embedding generation
      - Rephrasing service desk inputs (tickets, calls, classifications)

    Attributes:
        cfg (LLMSettings): Configuration settings for the LLM API.
        client (httpx.AsyncClient): Persistent async HTTP client.
        ticket_rephrase_prompt (str): Prompt template for rephrasing tickets.
        call_rephrase_prompt (str): Prompt template for rephrasing call transcripts.
        classification_rephrase_prompt (str): Prompt template for rephrasing classification queries.
    """
    def __init__(self, settings: LLMSettings = LLMSettings(), prompts: PromptSettings = PromptSettings()):
        """
        Initialize the LLMClient with API settings and prompt templates.

        Args:
            settings (LLMSettings): Configuration for the LLM (API key, URL, model, etc.).
            prompts (PromptSettings, optional): Prompt configuration object containing prompt templates.
        """
        self.cfg = settings
        self.client = httpx.AsyncClient(timeout=60.0)  # persistent client
        self.ticket_rephrase_prompt = prompts.get('ticket_rephrase')
        self.call_rephrase_prompt = prompts.get('call_rephrase')
        self.classification_rephrase_prompt = prompts.get('classification_rephrase')

    def headers(self) -> Dict[str, str]:
        """
        Build authorization and content headers for API requests.

        Returns:
            Dict[str, str]: HTTP headers including Authorization and Content-Type.
        """
        return {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }

    async def chat(self, messages: List[Dict[str, str]], **overrides) -> str:
        """
        Send a chat completion request to the LLM API.

        Args:
            messages (List[Dict[str, str]]): A list of messages, each with "role" and "content".
            **overrides: Optional keyword arguments to override model parameters such as
                temperature, max_tokens, etc.

        Returns:
            str: The assistant's response text.
        """
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            **self.cfg.default_kwargs,
            **overrides,
        }
        r = await self.client.post(
            f"{self.cfg.base_url}/v1/chat/completions",
            headers=self.headers(),
            json=payload,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    async def embed(
        self,
        texts: Union[str, List[str]],
        model="tgpt-embeddings",
        dim=2560,
    ) -> List[List[float]]:
        """
        Generate vector embeddings for text(s).

        Args:
            texts (Union[str, List[str]]): A single string or a list of strings to embed.
            model (str, optional): Embedding model name. Defaults to "tgpt-embeddings".
            dim (int, optional): Embedding dimensionality. Defaults to 2560.

        Returns:
            List[List[float]]: A list of embeddings corresponding to each input text.
        """
        if isinstance(texts, str):
            texts = [texts]
        r = await self.client.post(
            f"{self.cfg.base_url}/v1/embeddings",
            headers=self.headers(),
            json={"model": model, "input": texts, "dimensions": dim},
        )
        r.raise_for_status()
        data = r.json()
        return [d["embedding"] for d in data["data"]]

    async def rephrase(self, ticket_text: str, mode: Literal['ticket', 'call', 'classification'] = 'ticket') -> str:
        """
        Rephrase text for different service desk tasks.

        Args:
            ticket_text (str): Input text to rephrase (e.g., a ticket description).
            mode (Literal['ticket', 'call', 'classification'], optional): 
                The rephrase mode:
                  - 'ticket': Rephrase a support ticket.
                  - 'call': Rephrase call notes.
                  - 'classification': Rephrase text for classification.
                Defaults to 'ticket'.

        Returns:
            str: The rephrased text.
        """
        if mode=='ticket':
            request = self.ticket_rephrase_prompt.format(ticket_text=ticket_text)
        elif mode=='call':
            request = self.call_rephrase_prompt.format(ticket_text=ticket_text)
        else:
            request = self.classification_rephrase_prompt.format(ticket_text=ticket_text)
        messages = [
            {"role": "system", "content": "You are a rephraser for various service desk tasks, such as before a query or a classification request."},
            {"role": "user", "content": request},
        ]
        return await self.chat(
            messages,
            model="gpt-oss-120b",
            temperature=0.0,
            max_tokens=1024,
        )

    async def aclose(self):
        """
        Gracefully close the underlying async HTTP client.

        Should be called during application shutdown to release resources.
        """
        await self.client.aclose()
