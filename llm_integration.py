from transformers import pipeline

def get_gemini_response(prompt, question):
    max_input_length = 1024 - 150  # Reserve space for the generated tokens
    combined_input = (prompt + "\n\n" + question)[:max_input_length]  # Truncate if necessary

    generator = pipeline('text-generation', model='gpt2')
    response = generator(combined_input, max_new_tokens=150, truncation=True)
    return response[0]['generated_text']

if __name__ == "__main__":
    prompt = "You are an expert in generating TensorFlow code. Follow the structure and examples provided."
    question = "Create a simple neural network for image classification."
    response = get_gemini_response(prompt, question)
    print(response)
