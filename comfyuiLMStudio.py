import random
import re
import json
from typing import Optional, List, Dict, Any

import httpx
import numpy as np
import base64
from io import BytesIO
from server import PromptServer
from aiohttp import web
from pprint import pprint
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import os


# Helper function to format conversation history for readability
def format_conversation_history(messages: List[Dict[str, Any]]) -> str:
    formatted_history = []
    for message in messages:
        role = message.get("role", "unknown").capitalize()
        content = message.get("content", "")
        
        # Handle content that might be a list (e.g., for vision models)
        if isinstance(content, list):
            text_parts = [part["text"] for part in content if part.get("type") == "text"]
            content = "\n".join(text_parts) + " (contains images)" if text_parts else "(contains images)"

        formatted_history.append(f"--- {role} ---\n{content}\n")
    return "\n".join(formatted_history).strip()


# This endpoint will fetch models from LM Studio's API
@PromptServer.instance.routes.post("/lmstudio/get_models")
async def get_models_endpoint(request):
    data = await request.json()

    url = data.get("url")

    try:
        # LM Studio's models endpoint is typically /v1/models
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{url}/v1/models")
            response.raise_for_status() # Raise an exception for HTTP errors

        models_data = response.json().get('data', [])
        # LM Studio returns models with an 'id' field
        models = [model['id'] for model in models_data if 'id' in model]
        return web.json_response(models)
    except httpx.RequestError as e:
        print(f"Error fetching models from LM Studio: {e}")
        return web.json_response({"error": f"Could not connect to LM Studio: {e}"}, status=500)
    except json.JSONDecodeError:
        print("Error decoding JSON response from LM Studio.")
        return web.json_response({"error": "Invalid JSON response from LM Studio"}, status=500)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return web.json_response({"error": f"An unexpected error occurred: {e}"}, status=500)


class LMStudioVision:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        seed = random.randint(1, 2 ** 31)
        return {
            "required": {
                "images": ("IMAGE",),
                "query": ("STRING", {
                    "multiline": True,
                    "default": "describe the image"
                }),
                "debug": (["enable", "disable"],),
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://localhost:1234" # Default LM Studio URL
                }),
                "model": ((), {}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.05}),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2 ** 31, "step": 1}),
            },
            "optional": {
                "previous_conv": ("STRING", {"forceInput": True, "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",) # Added read_conv output
    RETURN_NAMES = ("description", "previous_conv", "read_conv",) # Added read_conv output name
    FUNCTION = "lmstudio_vision"
    CATEGORY = "LM Studio"

    def lmstudio_vision(self, images, query, debug, url, model, temperature, seed, previous_conv=None):
        messages: List[Dict[str, Any]] = []

        if previous_conv:
            try:
                history = json.loads(previous_conv)
                if isinstance(history, list):
                    messages.extend(history)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse previous_conv JSON: {previous_conv}")

        images_b64 = []
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_bytes = base64.b64encode(buffered.getvalue())
            images_b64.append(str(img_bytes, 'utf-8'))

        user_content = [{"type": "text", "text": query}] + [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}} for img_b64 in images_b64]
        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "seed": seed,
        }

        if debug == "enable":
            print(f"""[LM Studio Vision]
request query params:

- query: {query}
- url: {url}
- model: {model}
- temperature: {temperature}
- seed: {seed}
- messages: {messages}
""")

        try:
            with httpx.Client() as client:
                response = client.post(f"{url}/v1/chat/completions", json=payload, timeout=600.0)
                response.raise_for_status()
            
            lmstudio_response = response.json()

            if debug == "enable":
                print("[LM Studio Vision]\nResponse:\n")
                pprint(lmstudio_response)

            description = lmstudio_response['choices'][0]['message']['content']
            messages.append({"role": "assistant", "content": description}) # Add assistant response to history
            
            return (description, json.dumps(messages), format_conversation_history(messages),) # Return raw JSON and formatted history
        except httpx.RequestError as e:
            raise Exception(f"LM Studio Vision API request failed: {e}")
        except json.JSONDecodeError:
            raise Exception("LM Studio Vision API returned invalid JSON.")
        except KeyError as e:
            raise Exception(f"LM Studio Vision API response missing expected key: {e}. Full response: {lmstudio_response}")


class LMStudioGenerate:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        seed = random.randint(1, 2 ** 31)
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "What is Art?"
                }),
                "debug": (["enable", "disable"],),
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://localhost:1234" # Default LM Studio URL
                }),
                "model": ((), {}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.05}),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2 ** 31, "step": 1}),
                "filter_thinking": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "previous_conv": ("STRING", {"forceInput": True, "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",) # Added read_conv output
    RETURN_NAMES = ("response", "previous_conv", "read_conv",) # Added read_conv output name
    FUNCTION = "lmstudio_generate"
    CATEGORY = "LM Studio"

    def lmstudio_generate(self, prompt, debug, url, model, temperature, seed, filter_thinking, previous_conv=None):
        messages: List[Dict[str, Any]] = []

        if previous_conv:
            try:
                history = json.loads(previous_conv)
                if isinstance(history, list):
                    messages.extend(history)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse previous_conv JSON: {previous_conv}")

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "seed": seed,
        }

        if debug == "enable":
            print(f"""[LM Studio Generate]
request query params:

- prompt: {prompt}
- url: {url}
- model: {model}
- temperature: {temperature}
- seed: {seed}
- messages: {messages}
            """)

        try:
            with httpx.Client() as client:
                response = client.post(f"{url}/v1/chat/completions", json=payload, timeout=600.0)
                response.raise_for_status()
            
            lmstudio_response = response.json()

            if debug == "enable":
                print("[LM Studio Generate]\nResponse:\n")
                pprint(lmstudio_response)
            
            lmstudio_response_text = lmstudio_response['choices'][0]['message']['content']
            if filter_thinking:
                lmstudio_response_text = re.sub(r"<(?:think|thinking)>.*?</(?:think|thinking)>\s*", "", lmstudio_response_text, flags=re.DOTALL | re.IGNORECASE).strip()

            messages.append({"role": "assistant", "content": lmstudio_response_text}) # Add assistant response to history

            return (lmstudio_response_text, json.dumps(messages), format_conversation_history(messages),) # Return raw JSON and formatted history
        except httpx.RequestError as e:
            raise Exception(f"LM Studio Generate API request failed: {e}")
        except json.JSONDecodeError:
            raise Exception("LM Studio Generate API returned invalid JSON.")
        except KeyError as e:
            raise Exception(f"LM Studio Generate API response missing expected key: {e}. Full response: {lmstudio_response}")


class LMStudioGenerateAdvance:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        seed = random.randint(1, 2 ** 31)
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "What is Art?"
                }),
                "debug": ("BOOLEAN", {"default": False}),
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://localhost:1234" # Default LM Studio URL
                }),
                "model": ((), {}),
                "system": ("STRING", {
                    "multiline": True,
                    "default": "You are an art expert, gracefully describing your knowledge in art domain.",
                    "title": "system"
                }),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2 ** 31, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0, "max": 1, "step": 0.05}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100, "step": 1}),
                "max_tokens": ("INT", {"default": -1, "min": -1, "max": 4096, "step": 1}),
                "filter_thinking": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "previous_conv": ("STRING", {"forceInput": True, "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",) # Added read_conv output
    RETURN_NAMES = ("response", "previous_conv", "read_conv",) # Added read_conv output name
    FUNCTION = "lmstudio_generate_advance"
    CATEGORY = "LM Studio"

    def lmstudio_generate_advance(self, prompt, debug, url, model, system, seed, temperature, top_p, top_k, max_tokens,
                                        filter_thinking, previous_conv=None):

        messages: List[Dict[str, Any]] = []

        if previous_conv:
            try:
                history = json.loads(previous_conv)
                if isinstance(history, list):
                    messages.extend(history)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse previous_conv JSON: {previous_conv}")

        if system:
            if not messages or messages[0].get("role") != "system" or messages[0].get("content") != system:
                messages.insert(0, {"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
        }
        if max_tokens != -1:
            payload["max_tokens"] = max_tokens


        if debug:
            print(f"""[LM Studio Generate Advance]
request query params:

- prompt: {prompt}
- url: {url}
- model: {model}
- system: {system}
- messages: {messages}
- temperature: {temperature}
- top_p: {top_p}
- top_k: {top_k}
- max_tokens: {max_tokens}
- seed: {seed}
""")

        try:
            with httpx.Client() as client:
                response = client.post(f"{url}/v1/chat/completions", json=payload, timeout=600.0)
                response.raise_for_status()
            
            lmstudio_response = response.json()

            if debug:
                print("[LM Studio Generate Advance]\nResponse:\n")
                pprint(lmstudio_response)

            response_content = lmstudio_response['choices'][0]['message']['content']
            if filter_thinking:
                response_content = re.sub(r"<(?:think|thinking)>.*?</(?:think|thinking)>\s*", "", response_content, flags=re.DOTALL | re.IGNORECASE).strip()

            messages.append({"role": "assistant", "content": response_content})
            
            return (response_content, json.dumps(messages), format_conversation_history(messages),) # Return raw JSON and formatted history
        except httpx.RequestError as e:
            raise Exception(f"LM Studio Generate Advance API request failed: {e}")
        except json.JSONDecodeError:
            raise Exception("LM Studio Generate Advance API returned invalid JSON.")
        except KeyError as e:
            raise Exception(f"LM Studio Generate Advance API response missing expected key: {e}. Full response: {lmstudio_response}")


class LMStudioOptionsV2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        seed = random.randint(1, 2 ** 31)
        return {
            "required": {
                "enable_temperature": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.05}),

                "enable_top_p": ("BOOLEAN", {"default": False}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0, "max": 1, "step": 0.05}),

                "enable_seed": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2 ** 31, "step": 1}),

                "enable_max_tokens": ("BOOLEAN", {"default": False}),
                "max_tokens": ("INT", {"default": -1, "min": -1, "max": 4096, "step": 1}),

                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "previous_conv": ("STRING", {"forceInput": True, "multiline": True}),
            }
        }

    RETURN_TYPES = ("LMSTUDIO_OPTIONS", "STRING", "STRING",) # Added read_conv output
    RETURN_NAMES = ("options", "previous_conv", "read_conv",) # Added read_conv output name
    FUNCTION = "lmstudio_options"
    CATEGORY = "LM Studio"

    def lmstudio_options(self, previous_conv=None, **kargs):
        # Pass through previous_conv, or an empty JSON string if not provided
        raw_conv = previous_conv if previous_conv is not None else json.dumps([])
        formatted_conv = format_conversation_history(json.loads(raw_conv))

        if kargs['debug']:
            print("--- LM Studio options v2 dump\n")
            pprint(kargs)
            print("---------------------------------------------------------")

        return (kargs, raw_conv, formatted_conv,)


class LMStudioConnectivityV2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://localhost:1234"
                }),
                "model": ((), {}),
            },
            "optional": {
                "previous_conv": ("STRING", {"forceInput": True, "multiline": True}),
            }
        }

    RETURN_TYPES = ("LMSTUDIO_CONNECTIVITY", "STRING", "STRING",) # Added read_conv output
    RETURN_NAMES = ("connection", "previous_conv", "read_conv",) # Added read_conv output name
    FUNCTION = "lmstudio_connectivity"
    CATEGORY = "LM Studio"

    def lmstudio_connectivity(self, url, model, previous_conv=None):
        data = {
            "url": url,
            "model": model,
        }
        # Pass through previous_conv, or an empty JSON string if not provided
        raw_conv = previous_conv if previous_conv is not None else json.dumps([])
        formatted_conv = format_conversation_history(json.loads(raw_conv))

        return (data, raw_conv, formatted_conv,)


class LMStudioGenerateV2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "system": ("STRING", {
                    "multiline": True,
                    "default": "You are an AI artist."
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "What is art?"
                }),
                "filter_thinking": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "connectivity": ("LMSTUDIO_CONNECTIVITY", {"forceInput": False},),
                "options": ("LMSTUDIO_OPTIONS", {"forceInput": False},),
                "images": ("IMAGE", {"forceInput": False},),
                "previous_conv": ("STRING", {"forceInput": False, "multiline": True}),
                "meta": ("LMSTUDIO_META", {"forceInput": False},),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "LMSTUDIO_META", "STRING",) # Added read_conv output
    RETURN_NAMES = ("result", "previous_conv", "meta", "read_conv",) # Added read_conv output name
    FUNCTION = "lmstudio_generate_v2"
    CATEGORY = "LM Studio"

    def get_request_options(self, options_dict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extracts enabled options from the options dictionary."""
        request_options = {}
        if options_dict is None:
            return request_options

        option_map = {
            "enable_temperature": "temperature",
            "enable_top_p": "top_p",
            "enable_seed": "seed",
            "enable_max_tokens": "max_tokens",
        }

        for enable_key, actual_key in option_map.items():
            if options_dict.get(enable_key, False):
                value = options_dict.get(actual_key)
                if actual_key == "max_tokens" and value == -1:
                    request_options[actual_key] = None
                else:
                    request_options[actual_key] = value
        return request_options

    def lmstudio_generate_v2(self, system, prompt, filter_thinking, previous_conv=None, options=None, connectivity=None, images=None, meta=None):

        current_connectivity = connectivity
        current_options = options
        
        if meta is not None:
            if connectivity is None and "connectivity" in meta:
                current_connectivity = meta["connectivity"]
            if options is None and "options" in meta:
                current_options = meta["options"]
        else:
            meta = {}

        if current_connectivity is None:
            raise Exception("Required input 'connectivity' or 'connectivity' in 'meta'.")

        url = current_connectivity['url']
        model = current_connectivity['model']
        
        debug_print = True if current_options is not None and current_options.get('debug', False) else False

        messages: List[Dict[str, Any]] = []

        if previous_conv:
            try:
                history = json.loads(previous_conv)
                if isinstance(history, list):
                    messages.extend(history)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse previous_conv JSON: {previous_conv}")

        if system:
            if not messages or messages[0].get("role") != "system" or messages[0].get("content") != system:
                messages.insert(0, {"role": "system", "content": system})

        if images is not None:
            image_content_parts = []
            for (batch_number, image) in enumerate(images):
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_bytes = base64.b64encode(buffered.getvalue())
                image_content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{str(img_bytes, 'utf-8')}"}})
            
            user_content = [{"type": "text", "text": prompt}] + image_content_parts
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": prompt})

        request_options = self.get_request_options(current_options)

        payload = {
            "model": model,
            "messages": messages,
            **request_options
        }

        if debug_print:
            print(f"""
--- LM Studio generate v2 request: 

url: {url}
model: {model}
system: {system}
prompt: {prompt}
images: {0 if images is None else len(images)}
messages: {messages}
options: {request_options}
---------------------------------------------------------
""")

        try:
            with httpx.Client() as client:
                response = client.post(f"{url}/v1/chat/completions", json=payload, timeout=600.0)
                response.raise_for_status()
            
            lmstudio_response = response.json()

            if debug_print:
                print("\n--- LM Studio generate v2 response:")
                pprint(lmstudio_response)
                print("---------------------------------------------------------")

            response_content = lmstudio_response['choices'][0]['message']['content']
            if filter_thinking:
                response_content = re.sub(r"<(?:think|thinking)>.*?</(?:think|thinking)>\s*", "", response_content, flags=re.DOTALL | re.IGNORECASE).strip()

            messages.append({"role": "assistant", "content": response_content})
            
            meta["connectivity"] = current_connectivity
            meta["options"] = current_options

            return (response_content, json.dumps(messages), meta, format_conversation_history(messages),) # Return raw JSON, meta, and formatted history
        except httpx.RequestError as e:
            raise Exception(f"LM Studio Generate V2 API request failed: {e}")
        except json.JSONDecodeError:
            raise Exception("LM Studio Generate V2 API returned invalid JSON.")
        except KeyError as e:
            raise Exception(f"LM Studio Generate V2 API response missing expected key: {e}. Full response: {lmstudio_response}")


class LMStudioSequentialPrompt:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        seed = random.randint(1, 2 ** 31)
        return {
            "required": {
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://localhost:1234"
                }),
                "model": ((), {}),
                "prompt_templates": ("STRING", {
                    "multiline": True,
                    "default": "Step 1: Summarize the following text: 'The quick brown fox jumps over the lazy dog.'\n---\nStep 2: Explain the meaning of 'quick brown fox'.",
                    "title": "Prompt Templates (separated by delimiter)"
                }),
                "delimiter": ("STRING", {
                    "multiline": False,
                    "default": "---",
                    "title": "Delimiter for Prompt Templates"
                }),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.05}),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2 ** 31, "step": 1}),
                "send_as_chat": ("BOOLEAN", {"default": False, "label_on": "Send Current Prompt Only", "label_off": "Send Full History"}),
                "debug": ("BOOLEAN", {"default": False}),
                "filter_thinking": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "previous_conv": ("STRING", {"forceInput": True, "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",) # Added read_conv output
    RETURN_NAMES = ("final_response", "previous_conv", "read_conv",) # Added read_conv output name
    FUNCTION = "lmstudio_sequential_prompt"
    CATEGORY = "LM Studio/Sequential"

    def lmstudio_sequential_prompt(self, url, model, prompt_templates, delimiter, temperature, seed, send_as_chat, debug, filter_thinking, previous_conv=None):
        messages: List[Dict[str, Any]] = []
        final_response_content = ""

        if previous_conv:
            try:
                history = json.loads(previous_conv)
                if isinstance(history, list):
                    messages.extend(history)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse previous_conv JSON: {previous_conv}")

        templates = [t.strip() for t in prompt_templates.split(delimiter) if t.strip()]

        for i, template in enumerate(templates):
            user_prompt = template
            
            messages.append({"role": "user", "content": user_prompt})

            messages_to_send = []
            if send_as_chat:
                messages_to_send = [{"role": "user", "content": user_prompt}]
            else:
                messages_to_send = list(messages)

            payload = {
                "model": model,
                "messages": messages_to_send,
                "temperature": temperature,
                "seed": seed,
            }

            if debug:
                print(f"""[LM Studio Sequential Prompt - Step {i+1}]
request query params:

- url: {url}
- model: {model}
- current_user_prompt: {user_prompt}
- messages_sent_to_api: {messages_to_send}
- temperature: {temperature}
- seed: {seed}
- send_as_chat (only current prompt): {send_as_chat}
""")

            try:
                with httpx.Client() as client:
                    response = client.post(f"{url}/v1/chat/completions", json=payload, timeout=600.0)
                    response.raise_for_status()
                
                lmstudio_response = response.json()

                if debug:
                    print(f"[LM Studio Sequential Prompt - Step {i+1}]\nResponse:\n")
                    pprint(lmstudio_response)

                assistant_response_content = lmstudio_response['choices'][0]['message']['content']
                if filter_thinking:
                    assistant_response_content = re.sub(r"<(?:think|thinking)>.*?</(?:think|thinking)>\s*", "", assistant_response_content, flags=re.DOTALL | re.IGNORECASE).strip()

                messages.append({"role": "assistant", "content": assistant_response_content})
                final_response_content = assistant_response_content

            except httpx.RequestError as e:
                raise Exception(f"LM Studio Sequential Prompt API request failed at step {i+1}: {e}")
            except json.JSONDecodeError:
                raise Exception(f"LM Studio Sequential Prompt API returned invalid JSON at step {i+1}.")
            except KeyError as e:
                raise Exception(f"LM Studio Sequential Prompt API response missing expected key: {e}. Full response: {lmstudio_response}")
        
        return (final_response_content, json.dumps(messages), format_conversation_history(messages),)


class LMStudioSequentialPromptAdvanced:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        seed = random.randint(1, 2 ** 31)
        return {
            "required": {
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://localhost:1234" # Default LM Studio URL
                }),
                "model": ((), {}),
                "system": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant.",
                    "title": "system"
                }),
                "prompt_templates": ("STRING", {
                    "multiline": True,
                    "default": "Step 1: Summarize the following text: 'The quick brown fox jumps over the lazy dog.'\n---\nStep 2: Explain the meaning of 'quick brown fox'.",
                    "title": "Prompt Templates (separated by delimiter)"
                }),
                "delimiter": ("STRING", {
                    "multiline": False,
                    "default": "---",
                    "title": "Delimiter for Prompt Templates"
                }),
                "debug": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2 ** 31, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0, "max": 1, "step": 0.05}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100, "step": 1}),
                "max_tokens": ("INT", {"default": -1, "min": -1, "max": 4096, "step": 1}),
                "send_as_chat": ("BOOLEAN", {"default": False, "label_on": "Send Current Prompt Only", "label_off": "Send Full History"}),
                "filter_thinking": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "previous_conv": ("STRING", {"forceInput": True, "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",) # Added read_conv output
    RETURN_NAMES = ("final_response", "previous_conv", "read_conv",) # Added read_conv output name
    FUNCTION = "lmstudio_sequential_prompt_advanced"
    CATEGORY = "LM Studio/Sequential"

    def lmstudio_sequential_prompt_advanced(self, url, model, system, prompt_templates, delimiter, debug, seed, temperature, top_p, top_k, max_tokens, send_as_chat, filter_thinking, previous_conv=None):
        messages: List[Dict[str, Any]] = []
        final_response_content = ""

        if previous_conv:
            try:
                history = json.loads(previous_conv)
                if isinstance(history, list):
                    messages.extend(history)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse previous_conv JSON: {previous_conv}")

        if system:
            if not messages or messages[0].get("role") != "system" or messages[0].get("content") != system:
                messages.insert(0, {"role": "system", "content": system})

        templates = [t.strip() for t in prompt_templates.split(delimiter) if t.strip()]

        for i, template in enumerate(templates):
            user_prompt = template
            
            messages.append({"role": "user", "content": user_prompt})

            messages_to_send = []
            if send_as_chat:
                if i == 0 and system:
                    messages_to_send = [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}]
                else:
                    messages_to_send = [{"role": "user", "content": user_prompt}]
            else:
                messages_to_send = list(messages)

            payload = {
                "model": model,
                "messages": messages_to_send,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "seed": seed,
            }
            if max_tokens != -1:
                payload["max_tokens"] = max_tokens

            if debug:
                print(f"""[LM Studio Sequential Prompt Advanced - Step {i+1}]
request query params:

- url: {url}
- model: {model}
- current_user_prompt: {user_prompt}
- messages_sent_to_api: {messages_to_send}
- temperature: {temperature}
- top_p: {top_p}
- top_k: {top_k}
- max_tokens: {max_tokens}
- seed: {seed}
- send_as_chat (only current prompt): {send_as_chat}
""")

            try:
                with httpx.Client() as client:
                    response = client.post(f"{url}/v1/chat/completions", json=payload, timeout=600.0)
                    response.raise_for_status()
                
                lmstudio_response = response.json()

                if debug:
                    print(f"[LM Studio Sequential Prompt Advanced - Step {i+1}]\nResponse:\n")
                    pprint(lmstudio_response)

                assistant_response_content = lmstudio_response['choices'][0]['message']['content']
                if filter_thinking:
                    assistant_response_content = re.sub(r"<(?:think|thinking)>.*?</(?:think|thinking)>\s*", "", assistant_response_content, flags=re.DOTALL | re.IGNORECASE).strip()

                messages.append({"role": "assistant", "content": assistant_response_content})
                final_response_content = assistant_response_content

            except httpx.RequestError as e:
                raise Exception(f"LM Studio Sequential Prompt Advanced API request failed at step {i+1}: {e}")
            except json.JSONDecodeError:
                raise Exception(f"LM Studio Sequential Prompt Advanced API returned invalid JSON at step {i+1}.")
            except KeyError as e:
                raise Exception(f"LM Studio Sequential Prompt Advanced API response missing expected key: {e}. Full response: {lmstudio_response}")
        
        return (final_response_content, json.dumps(messages), format_conversation_history(messages),)


NODE_CLASS_MAPPINGS = {
    "LMStudioVision": LMStudioVision,
    "LMStudioGenerate": LMStudioGenerate,
    "LMStudioGenerateAdvance": LMStudioGenerateAdvance,
    "LMStudioOptionsV2": LMStudioOptionsV2,
    "LMStudioConnectivityV2": LMStudioConnectivityV2,
    "LMStudioGenerateV2": LMStudioGenerateV2,
    "LMStudioSequentialPrompt": LMStudioSequentialPrompt,
    "LMStudioSequentialPromptAdvanced": LMStudioSequentialPromptAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LMStudioVision": "LM Studio Vision",
    "LMStudioGenerate": "LM Studio Generate",
    "LMStudioGenerateAdvance": "LM Studio Generate Advance",
    "LMStudioOptionsV2": "LM Studio Options V2",
    "LMStudioConnectivityV2": "LM Studio Connectivity V2",
    "LMStudioGenerateV2": "LM Studio Generate V2",
    "LMStudioSequentialPrompt": "LM Studio Sequential Prompt",
    "LMStudioSequentialPromptAdvanced": "LM Studio Sequential Prompt Advanced",
}
