import logging
from typing import Dict, List

from atlassian import Confluence
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.config import ServiceNowGroupsSettings, ClassifierSettings

logger = logging.getLogger(__name__)


class SnowAssignmentGroupsRetriever:
    """
    Retriever for the "SNOW Assignment Groups" Confluence page.

    Provides two main capabilities:
      - Keyword-based TF-IDF search against assignment group table rows.
      - Mapping classifier predictions to assignment group rows using
        exact/substring/embedding similarity matching.

    Attributes:
        settings (ServiceNowGroupsSettings): Configuration for Confluence connection.
        _client (Confluence): Atlassian Confluence client instance.
        _embedder (SentenceTransformer): Model used for computing embeddings
            when resolving classifier predictions to assignment groups.
        passages (List[str]): Human-readable text snippets built from table rows.
        rows (List[Dict[str, str]]): Parsed assignment group table rows, each with
            "assignment_group", "team_info", and "escalations".
        vectorizer (TfidfVectorizer): TF-IDF model used for keyword search.
        matrix (scipy.sparse matrix): TF-IDF feature matrix built from passages.
        prediction_cache (Dict[str, Dict[str, str]]): Cached mapping from classifier
            predictions to assignment group rows.
    """

    def __init__(self, settings: ServiceNowGroupsSettings = ServiceNowGroupsSettings(), 
                 classifier_settings: ClassifierSettings = ClassifierSettings()):
        """
        Initialize the retriever, loading the Confluence page content and building search indices.

        Args:
            settings (ServiceNowGroupsSettings, optional): Configuration with Confluence
                credentials, space key, and page title. Defaults to ServiceNowGroupsSettings().
            classifier_settings (ClassifierSettings, optional): Settings for classifier label
                mapping, used to build prediction-to-group cache. Defaults to ClassifierSettings().
        """
        self.settings = settings
        self._client = self._init_client()
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Load the page table rows once
        self.passages, self.rows = self._load_assignment_groups()

        # TF-IDF model for keyword search
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(self.passages or [""])

        # Prediction cache for classifier → assignment group lookup
        pred_dict = classifier_settings.load_label_mapping()
        self.prediction_cache = self._build_prediction_cache(pred_dict)

    def _init_client(self) -> Confluence:
        """
        Initialize a Confluence client.

        Returns:
            Confluence: Configured Confluence client instance.

        Raises:
            RuntimeError: If Confluence credentials are not set.
        """
        if not (self.settings.url and self.settings.email and self.settings.api_token):
            raise RuntimeError("Confluence credentials not set.")
        return Confluence(
            url=self.settings.url,
            username=self.settings.email,
            password=self.settings.api_token,
            cloud=True,
        )

    def _parse_first_table(self, html: str) -> List[Dict[str, str]]:
        """
        Parse the first table in the given HTML into structured rows.

        Args:
            html (str): HTML string containing table markup.

        Returns:
            List[Dict[str, str]]: Parsed rows with assignment group info.
        """
        soup = BeautifulSoup(html, "lxml-xml")
        tables = soup.find_all("table")
        if not tables:
            return []

        rows: List[Dict[str, str]] = []
        for tr in tables[0].find_all("tr")[1:]:
            cols = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
            if len(cols) >= 3:
                rows.append(
                    {
                        "assignment_group": cols[0],
                        "team_info": cols[1],
                        "escalations": cols[2],
                    }
                )
        return rows

    def _load_assignment_groups(self) -> tuple[List[str], List[Dict[str, str]]]:
        """
        Fetch and parse the SNOW Assignment Groups Confluence page.

        Returns:
            tuple[List[str], List[Dict[str, str]]]:
                - passages: Text snippets suitable for keyword search.
                - rows: Parsed assignment group rows.
        """
        page = self._client.get_page_by_title(
            space=self.settings.space_key,
            title=self.settings.page_title,
            expand="body.storage",
        )
        if not page or "body" not in page or "storage" not in page["body"]:
            logger.error(
                "SNOW Assignment Groups page not found: space=%s title=%s",
                self.settings.space_key,
                self.settings.page_title,
            )
            return [], []

        html = page["body"]["storage"]["value"]
        rows = self._parse_first_table(html)
        passages = [
            f"Assignment Group: {r['assignment_group']}. "
            f"Team Info: {r['team_info']}. "
            f"Escalations: {r['escalations']}"
            for r in rows
        ]
        return passages, rows

    def _build_prediction_cache(self, pred_dict: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        """
        Build a cache mapping classifier predictions to assignment group rows.

        Matching is done via:
          1. Exact match
          2. Substring match
          3. Embedding similarity (fallback)

        Args:
            pred_dict (Dict[str, str]): Mapping from label IDs to label names.

        Returns:
            Dict[str, Dict[str, str]]: Mapping from prediction string to row dict.
        """
        if not self.rows:
            return {}

        row_name_map = {r["assignment_group"].strip().lower(): r for r in self.rows}
        row_names_norm = list(row_name_map.keys())
        row_embeds = self._embedder.encode(row_names_norm, convert_to_tensor=True)

        cache: Dict[str, Dict[str, str]] = {}
        all_predictions = [f"ITS-{pred_dict[str(i)]}" for i in range(len(pred_dict))]

        for pred in all_predictions:
            norm_pred = pred.strip().lower()

            if norm_pred in row_name_map:  # exact match
                cache[pred] = row_name_map[norm_pred]
                continue

            submatches = [
                r for name, r in row_name_map.items()
                if norm_pred in name or name in norm_pred
            ]
            if submatches:  # substring
                best = min(submatches, key=lambda r: abs(len(r["assignment_group"]) - len(pred)))
                cache[pred] = best
                continue

            # embedding similarity
            pred_embed = self._embedder.encode(norm_pred, convert_to_tensor=True)
            sims = util.pytorch_cos_sim(pred_embed, row_embeds)[0]
            best_idx = int(sims.argmax())
            best_name = row_names_norm[best_idx]
            cache[pred] = row_name_map[best_name]

        return cache

    # ---------- Public methods ----------

    def keyword_search(self, query: str, k: int = 3) -> List[str]:
        """
        Retrieve assignment groups by keyword-based TF-IDF similarity.

        Args:
            query (str): User query string.
            k (int, optional): Number of top results to return. Defaults to 3.

        Returns:
            List[str]: Top-matching assignment group snippets.
        """
        if not self.passages:
            return []
        qv = self.vectorizer.transform([query or ""])
        sims = cosine_similarity(qv, self.matrix)[0]
        idxs = sims.argsort()[::-1][:k]
        return [self.passages[i] for i in idxs]

    def prediction_lookup(self, predictions: List[str]) -> List[str]:
        """
        Map classifier predictions to assignment group snippets.

        Args:
            predictions (List[str]): Classifier-predicted labels (with or without "ITS-" prefix).

        Returns:
            List[str]: Snippets describing the matched assignment groups.
        """
        results, seen = [], set()
        for pred in predictions:
            key = pred if pred.startswith("ITS-") else f"ITS-{pred}"
            row = self.prediction_cache.get(key)
            if row and row["assignment_group"] not in seen:
                snippet = (
                    f"Assignment Group: {row['assignment_group']}. "
                    f"Team Info: {row['team_info']}. "
                    f"Escalations: {row['escalations']}"
                )
                results.append(snippet)
                seen.add(row["assignment_group"])
        return results
