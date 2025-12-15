# utils/inference.py
import os
import base64
from groq import Groq

GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

class ModelWrapper:
    def __init__(self):
        print("Connecting to Groq API...")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise Exception("GROQ_API_KEY not found in environment variables. Get one at https://console.groq.com")
        
        self.client = Groq(api_key=api_key)
        print(f"✅ Connected to Groq API. Using model: {GROQ_MODEL}")

    def generate(self, prompt: str, image=None, max_tokens: int = 512) -> str:
        """
        Generate text using Groq API. Optionally accepts an image.
        image can be: file path, base64 string, or data URL
        """
        try:
            messages = []
            
            if image is not None:
                image_b64 = self._process_image(image)
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                })
            else:
                messages.append({
                    "role": "user",
                    "content": prompt
                })
            
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            return response.choices[0].message.content
                
        except Exception as e:
            print("Error during generation:", e)
            raise e

    def _process_image(self, image):
        """Convert image to base64 string for Groq"""
        if isinstance(image, str):
            if image.startswith('data:'):
                if ',' in image:
                    b64_data = image.split(',', 1)[1]
                    b64_data = b64_data.replace('\n', '').replace('\r', '').replace(' ', '')
                    return b64_data
                else:
                    raise ValueError("Invalid data URL format - missing comma separator")
            elif os.path.exists(image):
                with open(image, 'rb') as f:
                    return base64.b64encode(f.read()).decode('utf-8')
            else:
                b64_data = image.replace('\n', '').replace('\r', '').replace(' ', '')
                try:
                    base64.b64decode(b64_data)
                    return b64_data
                except Exception:
                    raise ValueError("Invalid base64 string provided")
        elif isinstance(image, bytes):
            return base64.b64encode(image).decode('utf-8')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

# Create a module-level model instance (loaded on import)
model = None

def get_model():
    global model
    if model is None:
        model = ModelWrapper()
    return model
