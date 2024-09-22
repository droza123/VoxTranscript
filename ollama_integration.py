import os
import logging
import requests
import asyncio
import json
import aiohttp
from asyncio import TimeoutError as AsyncTimeoutError
from transformers import AutoTokenizer
from subprocess_context import silent_subprocess

class OllamaIntegration:
    def __init__(self, base_url="http://localhost:11434", model="llama3.1:70b"):
        self.logger = logging.getLogger(__name__)
        self.base_url = base_url
        self.generate_endpoint = f"{self.base_url}/api/generate"
        self.tags_endpoint = f"{self.base_url}/api/tags"
        self.set_model(model)
        self.default_context_window = 4096
        self.tokenizer = self.init_tokenizer()

    def set_model(self, model):
        self.model = model
        self.logger.info(f"Set Ollama model to: {self.model}")

    def get_models(self):
        try:
            response = requests.get(self.tags_endpoint)
            response.raise_for_status()
            models = response.json().get('models', [])
            return [model['name'] for model in models]
        except requests.RequestException as e:
            self.logger.error(f"Error fetching Ollama models: {str(e)}")
            return []

    def init_tokenizer(self):
        tokenizer_path = os.path.join("models", "llama_tokenizer")
        os.makedirs(tokenizer_path, exist_ok=True)
        
        try:
            if os.path.exists(os.path.join(tokenizer_path, "tokenizer_config.json")):
                # Load existing config
                with open(os.path.join(tokenizer_path, "tokenizer_config.json"), 'r') as f:
                    config = json.load(f)
                
                # Update model_max_length if it's less than 128000. Debug log the value first
                self.logger.info(f"model_max_length: {config.get('model_max_length', 0)}")
                if config.get('model_max_length', 0) < 128000:
                    config['model_max_length'] = 128000
                    
                    # Save updated config
                    with open(os.path.join(tokenizer_path, "tokenizer_config.json"), 'w') as f:
                        json.dump(config, f, indent=2)
                    self.logger.info("Updated model_max_length in tokenizer config.")
                
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
                self.logger.info("Loaded Llama tokenizer from local files.")
            else:
                tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer", use_fast=True)
                if tokenizer.model_max_length < 128000:
                    tokenizer.model_max_length = 128000
                tokenizer.save_pretrained(tokenizer_path)
                self.logger.info("Downloaded and saved Llama tokenizer to local files.")
            
            return tokenizer
        except Exception as e:
            self.logger.error(f"Error initializing Llama tokenizer: {str(e)}")
            return self.fallback_tokenizer()

    def fallback_tokenizer(self):
        self.logger.info("Using fallback tokenizer")
        try:
            return AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        except Exception as e:
            self.logger.error(f"Error initializing fallback tokenizer: {str(e)}")
            return None

    def count_tokens(self, text: str) -> int:
        if self.tokenizer is None:
            self.logger.warning("No tokenizer available. Using character-based estimation.")
            return len(text) // 4  # Rough estimate: 1 token ≈ 4 characters
        return len(self.tokenizer.encode(text))


    async def ensure_ollama_running(self):
        try:
            response = await asyncio.to_thread(requests.get, f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                self.logger.info("Ollama is already running.")
                return True
        except (requests.RequestException, asyncio.TimeoutError) as e:
            self.logger.info(f"Ollama is not running. Error: {str(e)}")

        try:
            with silent_subprocess():
                process = await asyncio.create_subprocess_shell(
                    "ollama serve",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

            for _ in range(30):  # 30 second timeout
                try:
                    response = await asyncio.to_thread(requests.get, f"{self.base_url}/api/tags")
                    if response.status_code == 200:
                        self.logger.info("Ollama started successfully.")
                        return True
                except requests.RequestException:
                    await asyncio.sleep(1)

            self.logger.error("Failed to start Ollama within the timeout period.")
            return False

        except Exception as e:
            self.logger.error(f"Error starting Ollama: {str(e)}")
            return False

    async def generate_summary(self, transcript, includes_nuestro_padre):
        if not await self.ensure_ollama_running():
            self.logger.error("Failed to start Ollama. Unable to generate summary.")
            return None

        base_system_prompt = (
            "Eres un asistente de IA especializado en crear resúmenes objetivos de transcripciones para el archivo histórico de la Legión de Cristo. "
            "Sigue estas instrucciones para estructurar el resumen:\n"
            "1. Comienza identificando el género de la transcripción (por ejemplo, entrevista, conferencia, homilía, etc.) y el tema o temas principales.\n"
            "2. Utiliza un formato como: 'La transcripción es de [género] que trata sobre [tema(s) principal(es)].'\n"
            "3. Si es posible, identifica y menciona el período de tiempo o fechas específicas relacionadas con el contenido.\n"
            "4. Proporciona un resumen detallado del contenido, organizándolo de manera jerárquica si es extenso:\n"
            "   - Identifica los puntos principales y subpuntos relevantes.\n"
            "   - Destaca figuras clave, lugares o eventos mencionados.\n"
            "   - Incluye información contextual importante para la comprensión histórica.\n"
            "5. Mantén un tono objetivo y conciso, evitando opiniones personales o interpretaciones subjetivas.\n"
            "6. Limita el resumen a aproximadamente 250-350 palabras, ajustando según la complejidad del contenido.\n"
            "Proporciona solo el contenido del resumen, sin frases introductorias adicionales, elementos conversacionales o comentarios finales."
        )
        
        nuestro_padre_context = (
            "Instrucción crítica sobre el uso de 'Nuestro Padre' en este contexto:\n"
            "1. La frase 'Nuestro Padre' o 'nuestro padre' en esta transcripción se refiere específicamente al "
            "P. Marcial Maciel, LC, fundador de los Legionarios de Cristo.\n"
            "2. Este uso es exclusivo de los miembros de los Legionarios de Cristo o del Regnum Christi "
            "cuando hablan sobre el P. Maciel.\n"
            "3. En este contexto, 'Nuestro Padre' NO se refiere a:\n"
            "   - Dios Padre o cualquier referencia religiosa general\n"
            "   - Un padre biológico o cualquier forma de paternidad natural\n"
            "4. Si la frase 'Nuestro Padre' o 'nuestro padre' aparece en la transcripción, asume que:\n"
            "   - El hablante es probablemente un miembro de la Legión de Cristo o del Regnum Christi o alguien cercano a ellos\n"
            "   - Están hablando específicamente sobre el P. Marcial Maciel\n"
            "5. Esta interpretación solo aplica cuando la frase aparece en la transcripción.\n"
            "Interpreta y resume el contenido de la transcripción teniendo en cuenta esta contextualización, "
            "sin necesidad de explicar explícitamente el significado de 'Nuestro Padre' en el resumen."
        )

        system_prompt = base_system_prompt + (" " + nuestro_padre_context if includes_nuestro_padre else "")

        self.logger.info(f"System prompt being sent to Ollama: {system_prompt}")
        
        user_prompt = (
            "Resume la siguiente transcripción siguiendo las instrucciones proporcionadas: \n\n"
            f"{transcript}"
        )

        system_tokens = self.count_tokens(system_prompt)
        user_tokens = self.count_tokens(user_prompt)
        total_tokens = system_tokens + user_tokens

        self.logger.info(f"System tokens: {system_tokens}")
        self.logger.info(f"User tokens: {user_tokens}")
        self.logger.info(f"Total tokens: {total_tokens}")

        context_window = max(self.default_context_window, total_tokens + 1000)  # Add buffer

        self.logger.info(f"Using context window: {context_window}")

        payload = {
            "model": self.model,
            "prompt": user_prompt,
            "system": system_prompt,
            "options": {
                "num_ctx": context_window,
            },
            "stream": False,
            "keep_alive": 0
        }

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(requests.post, self.generate_endpoint, json=payload),
                timeout=10800  # 180-minute timeout
            )
            response.raise_for_status()
            result = response.json()
            return result['response']
        except AsyncTimeoutError:
            self.logger.error("Timeout while generating summary")
            return None
        except requests.RequestException as e:
            self.logger.error(f"Error communicating with Ollama: {str(e)}")
            return None
        except json.JSONDecodeError:
            self.logger.error("Failed to decode JSON response from Ollama")
            return None

    async def is_ollama_available(self):
        try:
            response = await asyncio.to_thread(requests.get, f"{self.base_url}/api/tags")
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    async def unload_model(self):
        try:
            self.logger.info(f"Attempting to unload Ollama model: {self.model}")
            
            payload = {
                "model": self.model,
                "messages": [],
                "keep_alive": 0
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.chat_endpoint, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get('done') and result.get('done_reason') == 'unload':
                            self.logger.info(f"Successfully unloaded model: {self.model}")
                        else:
                            self.logger.warning(f"Unexpected response when unloading model {self.model}: {result}")
                    else:
                        self.logger.error(f"Error unloading model {self.model}. Status: {response.status}")
        except Exception as e:
            self.logger.error(f"Exception while unloading model {self.model}: {str(e)}")
        finally:
            self.logger.info("Completed unload_model method")

# Usage example:
# async def main():
#     ollama = OllamaIntegration()
#     if await ollama.is_ollama_available():
#         summary = await ollama.generate_summary("Your transcript text here")
#         if summary:
#             print(summary)
#         else:
#             print("Failed to generate summary")
#     else:
#         print("Ollama is not available")
#
# asyncio.run(main())