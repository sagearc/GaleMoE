# vllm serve Qwen/Qwen3-32B --no-enable-prefix-caching --enforce-eager --max-model-len 16384


import json
import asyncio
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
import pandas as pd

# --- UPDATED CONFIG ---
# Point this to your A100 server address
client = AsyncOpenAI(
    base_url="http://localhost:8000/v1", 
    api_key="vllm-runs-locally"
)
MODEL_NAME = "Qwen/Qwen3-32B" # Must match the --model flag used to start vLLM
MAX_RETRIES = 5
MAX_CONCURRENT_REQUESTS = 100  # Limit concurrent requests to avoid overwhelming the server

# --- DEFINE PYDANTIC SCHEMAS ---
class TopicDetail(BaseModel):
    domain: str = Field(description="The broad, general category.")
    specific_niche: str = Field(description="The highly granular, specific unifying theme. Must be as precise and detailed as possible.")

class NeuronTopic(BaseModel):
    neuron_id: int = Field(description="The exact ID of the neuron from the input data.")
    topics: list[TopicDetail] = Field(min_length=2, max_length=2, description="Exactly TWO topic classifications for these titles.")

async def get_topics_from_vllm(row, semaphore):
    """Process a single neuron row and return its topic classification."""
    neuron_id = int(row['neuron_id'])
    
    async with semaphore:
        try:
            # Get all columns except neuron_id (top1, top2, ..., top20)
            title_cols = [col for col in row.index if col.startswith('top')]
            titles = [row[col] for col in title_cols if pd.notna(row[col])]
            
            # Unbiased prompt - let the model find patterns naturally
            prompt = f"""Analyze this neuron and return ONLY valid JSON.

Neuron ID: {neuron_id}
Titles: {json.dumps(titles, ensure_ascii=False)}

Identify TWO distinct themes:
- 'domain': broad, general category
- 'specific_niche': highly specific, granular unifying theme"""

            for attempt in range(MAX_RETRIES):
                try:
                    # Add timeout to prevent hanging
                    response = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=[{"role": "user", "content": prompt}],
                            extra_body={
                                "structured_outputs": {
                                    "json": NeuronTopic.model_json_schema()
                                }
                            },
                            temperature=0.1
                        ),
                        timeout=60.0  # 60 second timeout per request
                    )
                    
                    content = response.choices[0].message.content.strip()
                    
                    # Extract JSON if wrapped in text
                    if not content.startswith('{'):
                        start = content.find('{')
                        end = content.rfind('}') + 1
                        if start != -1 and end > start:
                            content = content[start:end]
                    
                    parsed_data = NeuronTopic.model_validate_json(content)
                    result = parsed_data.model_dump()
                    assert result['neuron_id'] == neuron_id, f"Neuron ID mismatch in response. Expected {neuron_id}, got {result['neuron_id']}"
                    return result
                    
                except asyncio.TimeoutError:
                    print(f"Neuron {neuron_id} - Attempt {attempt + 1} timed out. Retrying...")
                    if attempt == MAX_RETRIES - 1:
                        print(f"Neuron {neuron_id} - FAILED after {MAX_RETRIES} timeouts")
                        return None
                except Exception as e:
                    print(f"Neuron {neuron_id} - Attempt {attempt + 1} failed: {e}")
                    if attempt == MAX_RETRIES - 1:
                        print(f"Neuron {neuron_id} - FAILED after {MAX_RETRIES} attempts")
                        return None
        except Exception as e:
            print(f"Neuron {neuron_id} - Unexpected error: {e}")
            return None


async def main():
    import sys
    import os
    from tqdm.asyncio import tqdm_asyncio
    
    # Default CSV path or accept from command line
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = 'neuron_top20_titles_layer16_expert1_w3.csv'
    
    # Create output directory for individual results
    output_dir = csv_path.replace('.csv', '_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Check which neurons are already processed
    processed_ids = set()
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.startswith('neuron_') and filename.endswith('.json'):
                neuron_id = int(filename.replace('neuron_', '').replace('.json', ''))
                processed_ids.add(neuron_id)
    
    if processed_ids:
        print(f"Found {len(processed_ids)} already processed neurons. Resuming...")
    
    # Read CSV file
    df = pd.read_csv(csv_path)  # Remove .head() for full processing
    
    # Filter out already processed rows
    df_to_process = df[~df['neuron_id'].isin(processed_ids)]
    
    print(f"Loaded {len(df)} neurons from {csv_path}")
    print(f"Rows remaining to process: {len(df_to_process)}")
    print(f"Processing with max {MAX_CONCURRENT_REQUESTS} concurrent requests...\n")
    
    if len(df_to_process) == 0:
        print("All neurons already processed!")
        return
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async def process_and_save(row):
        result = await get_topics_from_vllm(row, semaphore)
        if result is not None:
            # Save to individual file
            neuron_file = os.path.join(output_dir, f"neuron_{result['neuron_id']}.json")
            with open(neuron_file, 'w') as f:
                json.dump(result, f, indent=2)
        return result
    
    # Process all rows with controlled concurrency
    tasks = [process_and_save(row) for idx, row in df_to_process.iterrows()]
    results = await tqdm_asyncio.gather(*tasks, desc="Processing neurons")
    
    # Count failures
    failed_count = sum(1 for r in results if r is None)
    total_processed = len(processed_ids) + len(results) - failed_count
    
    print(f"\n{'='*60}")
    print(f"COMPLETE! Results saved to {output_dir}/")
    print(f"Total processed: {total_processed}/{len(df)}")
    print(f"This session: {len(results) - failed_count}/{len(results)}")
    if failed_count > 0:
        print(f"Failed this session: {failed_count}")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())
