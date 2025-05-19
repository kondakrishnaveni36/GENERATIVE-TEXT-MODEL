!pip install transformers torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def preprocess_input(text):
    text = text.strip()
    text = ' '.join(text.split())
    return text

def generate_text(prompt, max_length=100, temperature=1.0, top_k=50):
    prompt = preprocess_input(prompt)
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def batch_generate(prompts, max_length=100, temperature=1.0, top_k=50):
    results = []
    for i, prompt in enumerate(prompts):
        print(f"\nGenerating for Prompt {i+1}: {prompt}")
        output = generate_text(prompt, max_length, temperature, top_k)
        print(f"\nGenerated Text:\n{output}\n")
        results.append(output)
    return results

def evaluate_text(text):
    print("\nGenerated Text:\n")
    print(text)
    try:
        rating = int(input("\nRate the quality (1 to 5): "))
        if 1 <= rating <= 5:
            print(f"Thank you! You rated this: {rating}/5")
        else:
            print("Rating out of range.")
    except ValueError:
        print("Invalid input. Please enter a number.")

def user_interface():
    mode = input("Choose mode: (1) Single Prompt or (2) Batch Mode: ")

    if mode == '1':
        prompt = input("Enter your prompt: ")
        max_length = int(input("Max length of generated text: "))
        temperature = float(input("Temperature (e.g., 0.7): "))
        top_k = int(input("Top_k (e.g., 50): "))

        result = generate_text(prompt, max_length, temperature, top_k)
        evaluate_text(result)

    elif mode == '2':
        n = int(input("How many prompts do you want to enter? "))
        prompts = []
        for i in range(n):
            prompt = input(f"Enter prompt {i+1}: ")
            prompts.append(prompt)

        max_length = int(input("Max length of generated text: "))
        temperature = float(input("Temperature (e.g., 0.7): "))
        top_k = int(input("Top_k (e.g., 50): "))

        results = batch_generate(prompts, max_length, temperature, top_k)
        for output in results:
            evaluate_text(output)
    else:
        print("Invalid option.")
      user_interface()
