# services/classifier_service.py
import asyncio, httpx
import logging
import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from core.config import ClassifierSettings

logger = logging.getLogger(__name__)

class ClassifierService:
    """
    Service for text classification using either a remote API or a local Hugging Face model.

    Depending on the configuration, the service can:
      - Send requests to a remote classifier API (API mode).
      - Load and run a local transformer model for sequence classification (local mode).

    Attributes:
        settings (ClassifierSettings): Configuration for the classifier (paths, API URL, tokens, etc.).
        id2label (Dict[Union[str, int], str]): Mapping of label IDs to human-readable labels.
        client (Optional[httpx.AsyncClient]): HTTP client used in API mode.
        model (Optional[torch.nn.Module]): Transformer model used in local mode.
        tokenizer (Optional[AutoTokenizer]): Tokenizer for preparing input text in local mode.
        device (Optional[torch.device]): Device where the model is loaded (CPU or GPU).
    """
    def __init__(self, settings: ClassifierSettings = ClassifierSettings()):
        """
        Initialize the classifier service.

        Args:
            settings (ClassifierSettings, optional): Configuration for classifier mode,
                including label mapping, API endpoints, or local model paths.
                Defaults to a new ClassifierSettings() instance.

        Raises:
            Exception: If the classifier fails to initialize in local mode.
        """
        self.settings = settings
        self.id2label = settings.load_label_mapping()

        if self.settings.use_api:
            logger.info(f"Classifier set to API mode: {self.settings.classifier_api}")
            self.client = httpx.AsyncClient(timeout=10.0)
            self.model = None
            self.tokenizer = None
            self.device = None
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(settings.tokenizer_path, use_fast=False)
                self.model = AutoModelForSequenceClassification.from_pretrained(settings.model_path)
                self.model.to(self.device)
                self.model.eval()
                self.client = None
                logger.info(f"Classifier loaded locally with {len(self.id2label)} labels on {self.device}")
            except Exception as e:
                logger.error(f"Error initializing classifier: {e}")
                raise

    def get_labels(self) -> Dict[str, str]:
        """
        Get the mapping of label IDs to label names.

        Returns:
            Dict[str, str]: A dictionary where keys are label IDs (str or int) and values are label names.
        """
        return self.id2label

    async def predict(self, text: str, top_k: int = 3) -> List[str]:
        """
        Predict the most likely labels for the given input text.

        Args:
            text (str): Input text to classify.
            top_k (int, optional): Number of top predictions to return. Defaults to 3.

        Returns:
            List[str]: A list of predicted label names ranked by confidence.
        """
        # ---------- API MODE ----------
        if self.settings.use_api:
            headers = {
                "Authorization": f"Bearer {self.settings.classifier_token}",
                "Content-Type": "application/json"
            }
            payload = {"text": text, "top_k": top_k}

            resp = await self.client.post(
                self.settings.classifier_api,
                json=payload,
                headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("predictions", [])

        # ---------- LOCAL MODE ----------
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.model(**inputs)
            )
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        topk = torch.topk(probs, top_k)

        predictions = [
            self.id2label[str(idx.item())] if isinstance(next(iter(self.id2label.keys())), str)
            else self.id2label[int(idx.item())]
            for idx in topk.indices
        ]
        return predictions

    async def aclose(self):
        """
        Gracefully close the HTTP client if running in API mode.

        Should be called during application shutdown to free resources.
        """
        if self.client:
            await self.client.aclose()