import openai
import time
import json
import argparse
import os 
import tiktoken

# Initialize the OpenAI client with your API key
openai.api_key = os.getenv('OPENAI_API_KEY')
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def model_call(user_message, system_message='', model_id='gpt-3.5-turbo'):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    completion = openai.ChatCompletion.create(
        model=model_id,
        messages=messages
    )
    return completion.choices[0].message.content

def benchmark_prompts(filename, model_id):
    """
    Read prompts from the file and benchmark OpenAI API calls using the provided model.
    
    Returns: A list of tuples containing (time_taken, response_length).
    """
    results = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines:
        prompt = line.strip()

        start_time = time.time()
        response = model_call(prompt, model_id=model_id)
        end_time = time.time()

        time_taken = end_time - start_time
        response_length = len(encoding.encode(response))
        
        results.append((time_taken, response_length))
        print(time_taken, response_length)
        print(response)

    return results

def save_results_to_file(results, model_id, filename="benchmark.json"):
    """
    Save the benchmark results to a JSON file.
    """
    data = {}

    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write('{}')

    with open(filename, 'r') as f:
        data = json.load(f)

    if model_id in data:
        data[model_id].extend(results)
    else:
        data[model_id] = results

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark OpenAI models.')
    parser.add_argument('--model-id', type=str, required=True, help='Model ID to use for benchmarking.')
    parser.add_argument('--prompts-file', type=str, default='prompts.txt', help='File containing the prompts to benchmark.')

    args = parser.parse_args()

    input_filename = args.prompts_file

    if not input_filename:
        print("Please provide a prompts file using --prompts_file argument.")
        exit(1)
    
    benchmarks = benchmark_prompts(input_filename, args.model_id)
    
    # Save results to a file
    save_results_to_file(benchmarks, args.model_id)
    
    print(f"Benchmark results saved to benchmark.json for model: {args.model_id}!")

