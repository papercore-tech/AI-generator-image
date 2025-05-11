
import os
os.environ['FLAX_LAZY_RNG'] = 'no'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import torch
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import diffusers
import transformers
import huggingface_hub
import nltk
import threading
import time
from tkinter import ttk
    
# Проверка версий
print(">>> diffusers version:", diffusers.__version__)
print(">>> diffusers path:", diffusers.__file__)
print(">>> transformers version:", transformers.__version__)
print(">>> huggingface_hub version:", huggingface_hub.__version__)

# Загрузка NLTK токенов
nltk.download('punkt')

from translatepy import Translator

# Устройство
if not torch.cuda.is_available():
    print("⚠️ CUDA недоступна. Модель будет использовать CPU, что гораздо медленнее.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📦 Используемое устройство: {device}")

translator = Translator()

def translate_to_english(prompt: str) -> str:
    try:
        translation = translator.translate(prompt, "english")
        print(f"[DEBUG] Перевод: {translation.result}")
        return translation.result
    except Exception as e:
        print(f"[ERROR] Ошибка перевода: {e}")
        return prompt  # fallback


# GPT-2 для оценки сложности
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt2_model.eval()

# Stable Diffusion
from diffusers import StableDiffusionPipeline

print("🚀 Загрузка Stable Diffusion...")
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    low_cpu_mem_usage=False
).to(device)

# Оценка сложности текста
def evaluate_complexity(prompt):
    inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = gpt2_model(inputs, labels=inputs)
        return outputs.loss.item()

def generate_image(prompt):
    translated_prompt = translate_to_english(prompt)
    print(f"[DEBUG] Переведённый запрос: {translated_prompt}")

    complexity = evaluate_complexity(translated_prompt)
    print(f"[DEBUG] Сложность: {complexity:.2f}")

    image = pipe(translated_prompt, guidance_scale=7.5, num_inference_steps=50).images[0]

    os.makedirs("output", exist_ok=True)
    image.save("output/generated_image.png")
    return image

# Tkinter GUI
class ImageGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stable Diffusion Generator")
        self.root.geometry("800x600")

        self.label = tk.Label(root, text="Введите описание изображения:")
        self.label.pack(pady=10)

        self.entry = tk.Entry(root, width=70)
        self.entry.pack(pady=10)

        self.generate_button = tk.Button(root, text="Генерировать", command=self.generate)
        self.generate_button.pack(pady=20)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        self.progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="indeterminate")
        self.progress.pack(pady=10)

        self.status_label = tk.Label(root, text="")
        self.status_label.pack(pady=5)

    def generate(self):
        prompt = self.entry.get().strip()
        if not prompt:
            messagebox.showwarning("Ошибка", "Пожалуйста, введите описание.")
            return
        threading.Thread(target=self._run_generation, args=(prompt,)).start()

    def _run_generation(self, prompt):
        try:
            self.progress["mode"] = "indeterminate"
            self.progress.start(10)
            self.status_label.config(text="⏳ Генерация изображения...")

            img = generate_image(prompt)

            self.progress.stop()
            self.display_image(img)
            self.status_label.config(text="✅ Готово!")
            self.progress["value"] = 100

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.progress.stop()
            self.status_label.config(text="❌ Ошибка при генерации")
            messagebox.showerror("Ошибка генерации", str(e))

    def display_image(self, img):
        img = img.resize((512, 512))
        photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=photo)
        self.image_label.image = photo

# Запуск
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageGeneratorApp(root)
    root.mainloop()
