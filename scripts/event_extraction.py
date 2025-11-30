# src/event_extraction.py

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

class EventExtractor:
    """
    Performs:
    - Named Entity Recognition (NER) using BERT
    - Event trigger detection (rule-based)
    - Event classification
    """

    def __init__(self):
        # Load HuggingFace NER pipeline
        self.ner_pipeline = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            tokenizer="dslim/bert-base-NER",
            aggregation_strategy="simple"
        )

        # List of verbs grouped by event type
        self.event_lexicon = {
            "conflict": [
                "attacked", "killed", "bombed", "injured", "clashed",
                "fired", "shot", "exploded", "ambushed"
            ],
            "politics": [
                "announced", "approved", "voted", "elected", "appointed",
                "resigned", "negotiated", "signed"
            ],
            "legal": [
                "arrested", "charged", "sentenced", "investigated",
                "sued"
            ],
            "economy": [
                "launched", "acquired", "approved", "invested",
                "collapsed", "expanded"
            ],
            "disaster": [
                "flooded", "burned", "collapsed", "struck",
                "damaged"
            ]
        }

        # Flatten triggers for fast matching
        self.all_triggers = {
            verb: event_type
            for event_type, verbs in self.event_lexicon.items()
            for verb in verbs
        }
        # Put into single dict: verb → event_type
        self.trigger_to_event = {}
        for event_type, verbs in self.event_lexicon.items():
            for v in verbs:
                self.trigger_to_event[v] = event_type

    # -----------------------------
    # NER
    # -----------------------------
    def extract_entities(self, text: str):
        """
        Uses BERT NER model to extract named entities.
        Returns a dict grouped by entity type.
        """
        ent = self.ner_pipeline(text)

        entities = {}
        for item in ent:
            label = item["entity_group"]
            word = item["word"]

            if label not in entities:
                entities[label] = []

            entities[label].append(word)

        return entities

    # -----------------------------
    # Event Trigger Detection
    # -----------------------------
    def find_event_trigger(self, text: str):
        """
        Find the first occurring event verb in the article.
        Returns:
            trigger_word, event_type
        If no trigger found → (None, None)
        """
        lower_text = text.lower().split()

        for word in lower_text:
            if word in self.trigger_to_event:
                return word, self.trigger_to_event[word]

        return None, None

    # -----------------------------
    # Full Event Extraction
    # -----------------------------
    def extract_event(self, text: str):
        """
        Performs NER + event trigger classification.
        Returns ONE event per article.
        """
        # 1. Named Entities
        entities = self.extract_entities(text)

        # 2. Event Trigger
        trigger, event_type = self.find_event_trigger(text)

        # If no trigger → default event type
        if trigger is None:
            event_type = "unknown"

        return {
            "event_type": event_type,
            "trigger": trigger,
            "entities": entities
        }
