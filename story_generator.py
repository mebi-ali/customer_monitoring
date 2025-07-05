# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# class StoryGenerator:
#     def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.1"):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             torch_dtype=torch.float16,
#             device_map="auto"
#         )
        
#     def generate(self, features):
#         prompt = self._build_prompt(features)
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=300,
#                 do_sample=True,
#                 temperature=0.7,
#                 top_p=0.9
#             )
            
#         story = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return story.split("STORY:")[-1].strip()
    
#     def _build_prompt(self, features):
#         return f"""
#         Generate a creative speculative story about this person based on these attributes:
#         - Gender: {features['gender']}
#         - Age: {features['age']}
#         - Emotion: {features['emotion']}
#         - Clothing: {', '.join(features['clothing'])}
#         - Dominant color: {features['color']}
        
#         The story should be 3-5 paragraphs long and imagine a possible life scenario.
        
#         STORY:
#         """


# from jinja2 import Template
# import requests
# import time
# from typing import Dict, Optional

# class StoryGenerator:
#     def __init__(self, model: str = "mistral", host: str = "http://localhost:11434", max_retries: int = 3):
#         self.model = model
#         self.api_url = f"{host}/api/generate"
#         self.max_retries = max_retries
#         self.timeout = 45  # Increased timeout for local inference

#         self.template = Template("""
# [INST] You are observing a person in a public space. Generate a creative character sketch based on:

# Basic Attributes:
# - Age: {{ age }}
# - Gender: {{ gender }}
# - Emotional State: {{ emotion }}
# - Clothing Color: {{ clothing_color }}

# Guidelines:
# 1. Write 3 concise paragraphs (120-150 words total)
# 2. First paragraph: Describe their apparent lifestyle
# 3. Second paragraph: Speculate on their values/eco-consciousness
# 4. Third paragraph: Note any interesting contradictions
# 5. Maintain neutral but insightful tone
# 6. Avoid stereotypes - be thoughtful and original

# Example Structure:
# "[Name/description] appears to... They likely... However..."

# Output ONLY the story text. [/INST]
# """)

#     def _check_ollama_connection(self) -> bool:
#         """Verify Ollama server is running"""
#         try:
#             health_check = requests.get(
#                 f"{self.api_url.replace('/generate', '/tags')}",
#                 timeout=10
#             )
#             return health_check.status_code == 200
#         except requests.exceptions.RequestException:
#             return False

#     def build_prompt(self, attributes: Dict) -> str:
#         """Construct the prompt from extracted features"""
#         gender_data = attributes.get("gender", {})
#         primary_gender = max(gender_data.items(), key=lambda x: x[1])[0] if isinstance(gender_data, dict) else gender_data

#         context = {
#             "age": attributes.get("age", "unknown"),
#             "gender": primary_gender.lower(),
#             "emotion": attributes.get("emotion", "neutral"),
#             "clothing_color": attributes.get("colors", ["unknown"])[0] if attributes.get("colors") else "unknown"
#         }
#         return self.template.render(**context)

#     def generate_story(self, attributes: Dict) -> str:
#         """Generate story with retry logic"""
#         if not self._check_ollama_connection():
#             return self._fallback_story("Ollama server not available")

#         prompt = self.build_prompt(attributes)
#         for attempt in range(self.max_retries):
#             try:
#                 response = requests.post(
#                     self.api_url,
#                     json={
#                         "model": self.model,
#                         "prompt": prompt,
#                         "stream": False,
#                         # "options": {
#                         #     "temperature": 0.7,
#                         #     "top_p": 0.85,
#                         #     "num_ctx": 2048
#                         # }
#                     },
#                     timeout=self.timeout
#                 )
#                 print(f"\nRaw response:\n{response}\n")

                
#                 response.raise_for_status()
#                 data = response.json()
                
#                 story = data.get("response", "").strip()
                
#                 if story and len(story.split()) > 30:  # Basic length check
#                     return story
                
#                 raise ValueError("Incomplete story generated")

#             except Exception as e:
#                 print(f"[Attempt {attempt + 1}] Story generation issue: {str(e)}")
#                 if attempt < self.max_retries - 1:
#                     time.sleep(2 ** attempt)  # Exponential backoff
#                 continue

#         return self._fallback_story("Maximum retries exceeded")

#     def _fallback_story(self, reason: str) -> str:
#         print(f"Using fallback story: {reason}")
#         return (
#             "A contemplative figure moves through the space, their story quietly contained. "
#             "Perhaps they're thinking about their environmental impact, or maybe they're "
#             "considering how their daily choices align with their values. There's always "
#             "more beneath the surface than what first appears."
#         )


# # Sample attributes for testing
# attributes = {
#     "age": "30s",
#     "gender": {"female": 0.95, "male": 0.05},
#     "emotion": "thoughtful",
#     "colors": ["green", "brown"]
# }

# generator = StoryGenerator()

# story = generator.generate_story(attributes)
# print("\n--- Generated Story ---\n")
# print(story)


from pyexpat import features
from jinja2 import Template
import requests
from typing import Dict

class StoryGenerator:
    def __init__(self, model: str = "mistral", host: str = "http://localhost:11434"):
        self.model = model
        self.api_url = f"{host}/api/generate"
        self.timeout = (5, 300)  # (connect timeout, read timeout)

        self.template = Template("""
            [INST] You are observing a person in a public space. Generate a creative character sketch based on:

            Basic Attributes:
            - Age: {{ age }}
            - Gender: {{ gender }}
            - Emotional State: {{ emotion }}
            - Clothings: {{ clothing_list }}

            Guidelines:
            1. Write 3 concise paragraphs (120-150 words total)
            2. First paragraph: Describe their apparent lifestyle
            3. Second paragraph: Speculate on their values/eco-consciousness
            4. Third paragraph: Note any interesting contradictions
            5. Maintain neutral but insightful tone
            6. Avoid stereotypes - be thoughtful and original

            Example Structure:
            "[Name/description] appears to... They likely... However..."

            Output ONLY the story text. [/INST]
        """)
    def _check_ollama_connection(self) -> bool:
        """Verify Ollama server is running"""
        try:
            response = requests.get(f"{self.api_url.replace('/generate', '/tags')}", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def build_prompt(self, attributes: Dict) -> str:
        attributes['clothing_list'] = ', '.join(attributes.get('clothing', []))
        return self.template.render(**dict(attributes))

    def generate_story(self, attributes: Dict) -> str:
        if not self._check_ollama_connection():
            return self._fallback_story("Ollama server not available")

        prompt = self.build_prompt(attributes)
        # print(f"\nPrompt:\n{prompt}\n")  # Optional: Debug log

        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=self.timeout
            )
            print(f"\nRaw response:\n{response}\n")

            response.raise_for_status()
            
            data = response.json()
            story = data.get("response", "").strip()

            if story and len(story.split()) > 30:
                return story
            return self._fallback_story("Story too short or empty")

        except Exception as e:
            print(f"Story generation error: {str(e)}")
            return self._fallback_story("Exception occurred")

    def _fallback_story(self, reason: str) -> str:
        print(f"Using fallback story: {reason}")
        return (
            "A contemplative figure moves through the space, their story quietly contained. "
            "Perhaps they're thinking about their environmental impact, or maybe they're "
            "considering how their daily choices align with their values. There's always "
            "more beneath the surface than what first appears."
        )


# # Sample attributes for testing
# attributes = {
#     "age": "30s",
#     "gender": {"female": 0.95, "male": 0.05},
#     "emotion": "thoughtful",
#     "colors": ["green", "brown"]
# }

# generator = StoryGenerator()
# story = generator.generate_story(attributes)
# print("\n--- Generated Story ---\n")
# print(story)



# import requests
# import json
# from typing import Dict

# class StoryGenerator:
#     def __init__(self, ollama_host="http://localhost:11434"):
#         self.ollama_host = ollama_host
#         self.model_name = "mistral"
        
#         # Verify connection to Ollama
#         try:
#             response = requests.get(f"{self.ollama_host}/api/tags")
#             if response.status_code != 200:
#                 raise ConnectionError("Could not connect to Ollama server")
#             print("Connected to Ollama successfully")
#         except Exception as e:
#             raise ConnectionError(f"Ollama connection failed: {str(e)}")

#     def generate(self, features: Dict) -> str:
#         prompt = self._build_prompt(features)
        
#         try:
#             response = requests.post(
#                 f"{self.ollama_host}/api/generate",
#                 json={
#                     "model": self.model_name,
#                     "prompt": prompt,
#                     "stream": False,
#                     "options": {
#                         "temperature": 0.7,
#                         "top_p": 0.9,
#                         "max_tokens": 500
#                     }
#                 }
#             )
            
#             if response.status_code == 200:
#                 result = json.loads(response.text)
#                 return result.get("response", "").strip()
#             else:
#                 raise Exception(f"Generation failed: {response.text}")
                
#         except Exception as e:
#             print(f"Error in story generation: {str(e)}")
#             return "Could not generate story due to error."

#     def _build_prompt(self, features: Dict) -> str:
#         return f"""
#         [INST] Generate a creative speculative story about this person based on these attributes:
#         - Gender: {features.get('gender', 'unknown')}
#         - Age: {features.get('age', 'unknown')}
#         - Emotion: {features.get('emotion', 'neutral')}
#         - Clothing: {', '.join(features.get('clothing', []))}
#         - Dominant color: {features.get('color', 'unknown')}

#         The story should be 3-5 paragraphs long and imagine a possible life scenario.
#         Make it creative but plausible. Focus on narrative storytelling.
#         [/INST]

#         Story:
#         """