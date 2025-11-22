import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from playwright.sync_api import sync_playwright
import os

# Configure SSL to use Zscaler certificate
cert_path = os.path.join(os.path.dirname(__file__), "ZscalerCert")
os.environ['REQUESTS_CA_BUNDLE'] = cert_path
os.environ['SSL_CERT_FILE'] = cert_path
os.environ['CURL_CA_BUNDLE'] = cert_path


# 1. Capture the HTML Page as an Image
def capture_screenshot(url, output_path="page.png"):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        # specific viewport size helps normalize coordinates
        page.set_viewport_size({"width": 1024, "height": 1024})
        page.screenshot(path=output_path)
        browser.close()
    return output_path


# 2. Load the Model (Qwen2-VL is excellent for grounding)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",  # Use 7B or 72B for higher accuracy
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")


# 3. Define the Detection Function
def find_button_coordinates(image_path, button_description):
    image = Image.open(image_path)

    # Qwen specific prompting for detection
    prompt = f"Find the bounding box of the {button_description}."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Prepare inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = processor.image_processor(images=image, return_tensors="pt")
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Generate Output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return output_text


# --- Execution ---
screenshot_path = capture_screenshot("https://www.google.com")
result = find_button_coordinates(screenshot_path, "Google Search button")

print(f"Model Output: {result}")