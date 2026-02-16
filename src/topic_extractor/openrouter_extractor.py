"""
OpenRouter-based topic extraction using cheap SLM models.
"""
import json
import re
from typing import List, Dict
from openai import OpenAI

from .base import BaseTopicExtractor


# OpenRouter API base URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterTopicExtractor(BaseTopicExtractor):
    """Topic extraction using OpenRouter API with cheap SLM models."""
    
    def __init__(self, config):
        """
        Initialize OpenRouter topic extractor.
        
        Args:
            config: OpenRouterTopicConfig object
        """
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=OPENROUTER_BASE_URL
        )
        self.model = config.model
        self.batch_size = config.batch_size
    
    def extract_topics(self, texts: List[str], message_indices: List[int]) -> Dict[int, List[str]]:
        """
        Extract topics from ALL messages at once.
        
        Sends all messages together so the model can see the full conversation
        and connect topics across the entire discussion.
        
        Args:
            texts: List of message texts (in order)
            message_indices: List of message indices from the conversation
            
        Returns:
            Dict mapping message_index -> list of topics
        """
        if not texts:
            return {}
        
        return self._extract_all(texts, message_indices)
    
    def _extract_all(self, texts: List[str], message_indices: List[int]) -> Dict[int, List[str]]:
        """
        Extract topics from all messages using 2-step approach:
        1. Identify main topics from the conversation
        2. Map each message to relevant topics
        
        Args:
            texts: List of message texts
            message_indices: List of message indices from conversation
            
        Returns:
            Dict mapping message_index -> list of topics
        """

        messages_json = json.dumps([
            {"index": idx, "content": content} 
            for idx, content in zip(message_indices, texts)
        ], indent=2)
        
        # === STEP 1: Identify main topics ===
        step1_prompt = f"""You are analyzing a conversation between a human and an AI assistant to extract the main topics discussed.

CONVERSATION:
{messages_json}

TASK: Identify the 3-15 MOST IMPORTANT topics/subjects discussed in this conversation.

RULES:
- Extract ONLY the core topics, not every detail mentioned
- Maximum 15 topics - prioritize the most significant ones
- Topics should be specific and actionable
- Use lowercase with underscores for multi-word topics
- Merge related topics
- Topics should help someone search this conversation later

CRITICAL RULES:
- You MUST return EXACTLY 3-15 topics, NO MORE
- If you identify 20 topics, ruthlessly prioritize and keep only the top 15
- Merge similar topics
- DO NOT return more than 15 topics under any circumstances

BEFORE YOU RESPOND: Count your topics. If > 15, remove the least important ones.

Return ONLY a JSON array of topic strings.
Output Schema: ["<topic1>", "<topic2>", "<topic3>"]

JSON OUTPUT:"""
        
        try:
            response1 = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": step1_prompt}],
                temperature=0.6,
                max_tokens=250
            )
            
            content1 = response1.choices[0].message.content
            print(f"    Step 1 response: {content1}")
            
            # Parse topics from step 1
            topics = self._parse_topics_array(content1)
            if not topics:
                print(f"  Warning: Step 1 returned no valid topics, skipping topic extraction")
                return {idx: [] for idx in message_indices}
            
        except Exception as e:
            print(f"  Warning: Step 1 failed: {e}")
            return {idx: [] for idx in message_indices}
        
        # === STEP 2: Map messages to topics ===
        topics_json = json.dumps(topics)
        
        step2_prompt = f"""You are mapping messages to topics in a conversation.

TOPICS IDENTIFIED (use these EXACT topic names as keys. DO NOT invent new topics):
{topics_json}

MESSAGES (with indices):
{messages_json}

TASK: For each topic listed above, identify which messages (by their index number) discuss that topic.

RULES:
- Use the EXACT topic names from the list above as your JSON keys
- A message can relate to multiple topics
- A topic can have multiple messages
- Use the exact message index numbers from the input
- Only include messages that meaningfully discuss the topic
- Include messages where ideas were explored even if later rejected

CRITICAL: Your JSON keys must be the exact topic names from the topics list above. DO NOT invent new topics

Output Schema: {{"<topic_1>": [<message_index_1>, <message_index_2>, <message_index_3>], "<topic_2>": [<message_index_1>, <message_index_2>]}}

Return ONLY valid JSON with the same format as the output schema above. 
JSON OUTPUT:"""
        
        try:
            response2 = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": step2_prompt}],
                temperature=0.3,
                max_tokens=5000
            )
            
            content2 = response2.choices[0].message.content
            
            # Parse topic->messages mapping
            topic_to_messages = self._parse_topic_mapping(content2)
            
            # Print topics found in step 2 response
            step2_topics = list(topic_to_messages.keys())
            print(f"    Step 2 topics found: {step2_topics}")
            
            # Convert to message_index -> topics format
            result = self._convert_mapping(topic_to_messages, message_indices)
            print(f"    Parsed {len(result)} messages with topics")
            return result
            
        except Exception as e:
            print(f"  Warning: Step 2 failed: {e}")
            return {idx: [] for idx in message_indices}
    
    def _parse_topics_array(self, response: str) -> List[str]:
        """Parse the topics array from step 1 response."""
        try:
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                topics = json.loads(json_match.group())
                if isinstance(topics, list):
                    return [t.lower().strip().replace(' ', '_') for t in topics if t]
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"  Warning: Could not parse topics array: {e}")
        return []
    
    def _parse_topic_mapping(self, response: str) -> Dict[str, List[int]]:
        """Parse the topic->messages mapping from step 2 response."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                mapping = json.loads(json_match.group())
                if isinstance(mapping, dict):
                    result = {}
                    for topic, indices in mapping.items():
                        if isinstance(indices, list):
                            result[topic] = [int(i) for i in indices if str(i).isdigit()]
                        elif isinstance(indices, str) and indices.isdigit():
                            result[topic] = [int(indices)]
                    return result
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"  Warning: Could not parse topic mapping: {e}")
        return {}
    
    def _convert_mapping(self, topic_to_messages: Dict[str, List[int]], message_indices: List[int]) -> Dict[int, List[str]]:
        """Convert topic->messages mapping to message->topics mapping."""
        result = {idx: [] for idx in message_indices}
        
        for topic, msg_indices in topic_to_messages.items():
            normalized_topic = topic.lower().strip().replace(' ', '_')
            for msg_idx in msg_indices:
                if msg_idx in result:
                    if normalized_topic not in result[msg_idx]:
                        result[msg_idx].append(normalized_topic)
        
        return result
    
    def extract_topics_with_context(
        self, 
        texts: List[str], 
        message_indices: List[int],
        existing_topics: List[str]
    ) -> Dict[int, List[str]]:
        """
        Extract topics for new messages, considering existing topics in conversation.
        
        This is used during updates - passes existing topics so the SLM can either
        use them or create new ones for the new messages.
        
        Args:
            texts: List of new message texts
            message_indices: List of message indices from conversation
            existing_topics: List of topics already in the conversation
            
        Returns:
            Dict mapping message_index -> list of topics
        """
        if not texts:
            return {}
        
        return self._extract_with_context(texts, message_indices, existing_topics)
    
    def _extract_with_context(
        self, 
        texts: List[str], 
        message_indices: List[int], 
        existing_topics: List[str]
    ) -> Dict[int, List[str]]:
        """
        Extract topics with context of existing topics.
        
        Args:
            texts: List of new message texts
            message_indices: List of message indices from conversation
            existing_topics: List of topics already in conversation
            
        Returns:
            Dict mapping message_index -> list of topics
        """
        messages_content = []
        for text, msg_idx in zip(texts, message_indices):
            messages_content.append(f"[msg_{msg_idx}] {text}")
        
        combined_text = "\n\n".join(messages_content)
        existing_topics_str = ", ".join(existing_topics) if existing_topics else "none yet"
        
        prompt = f"""You are updating topic labels for NEW messages in an ongoing conversation.

EXISTING TOPICS in this conversation: {existing_topics_str}

For each NEW message below:
1. Use existing topics if relevant
2. Add new topics if the message introduces something new
3. Use empty array if no topics apply

Return a JSON object mapping the ORIGINAL message indices to their topics:
{{"10": ["existing_topic" OR "new_topic"], "11": [], "12": ["topic"]}}

Rules:
- Use the ORIGINAL message index as the key (e.g., "10", "11", "12" from [msg_10], [msg_11], [msg_12])
- Topics should be lowercase, 1-2 words
- A message can have MULTIPLE topics
- A topic can be on MULTIPLE messages (many-to-many relationship)
- Prefer reusing existing topics when applicable

NEW Messages:
{combined_text}

Return ONLY the JSON object, nothing else:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            return self._parse_response(content)
            
        except Exception as e:
            print(f"  Warning: Topic extraction with context failed: {e}")
            return {idx: [] for idx in message_indices}
    
    def _parse_response(self, response: str) -> Dict[int, List[str]]:
        """
        Parse LLM response to get dict mapping message_index -> topics.
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Dict mapping message_index -> list of topics
        """
        result = {}
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                topic_map = json.loads(json_match.group())
                
                if isinstance(topic_map, dict):
                    for idx, topics in topic_map.items():
                        try:
                            msg_idx = int(idx)
                            if isinstance(topics, list):
                                result[msg_idx] = [t.lower().strip() for t in topics if t]
                            elif isinstance(topics, str) and topics:
                                result[msg_idx] = [topics.lower().strip()]
                            else:
                                result[msg_idx] = []
                        except (ValueError, TypeError):
                            continue
                    
                    return result
                    
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"  Warning: Could not parse topic extraction response: {e}")
        
        return result
    
    def extract_topics_sync(self, text: str) -> List[str]:
        """
        Extract topics from a single text.
        
        Args:
            text: Message text
            
        Returns:
            List of topics
        """
        prompt = f"""Extract the main topics from this message.
Return a JSON array of topics, or empty array if no clear topics.
Topics should be lowercase, 1-2 words.

Message: {text[:1000]}

Return only a JSON array, nothing else:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            content = response.choices[0].message.content
            
            try:
                topics = json.loads(content)
                if isinstance(topics, list):
                    return topics
            except json.JSONDecodeError:
                pass
            
            return []
            
        except Exception as e:
            print(f"  Warning: Topic extraction failed: {e}")
            return []
