# vllm serve Qwen/Qwen3-32B-AWQ --quantization awq --no-enable-prefix-caching --enforce-eager

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
MODEL_NAME = "Qwen/Qwen3-32B-AWQ" # Must match the --model flag used to start vLLM
MAX_RETRIES = 5
MAX_CONCURRENT_REQUESTS = 50  # Limit concurrent requests to avoid overwhelming the server

# --- DEFINE PYDANTIC SCHEMAS ---
class TopicDetail(BaseModel):
    domain: str = Field(description="The broad, general category.")
    specific_niche: str = Field(description="The highly granular, specific unifying theme. Must be as precise and detailed as possible.")

class NeuronTopic(BaseModel):
    neuron_id: int = Field(description="The exact ID of the neuron from the input data.")
    topics: list[TopicDetail] = Field(description="Exactly TWO topic classifications for these titles.")

async def get_topics_from_vllm(row, semaphore):
    """Process a single neuron row and return its topic classification."""
    async with semaphore:
        neuron_id = int(row['neuron_id'])
        
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
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    extra_body={
                        "structured_outputs": {
                            "json": NeuronTopic.model_json_schema()
                        }
                    },
                    temperature=0.1
                )
                
                content = response.choices[0].message.content.strip()
                
                # Extract JSON if wrapped in text
                if not content.startswith('{'):
                    start = content.find('{')
                    end = content.rfind('}') + 1
                    if start != -1 and end > start:
                        content = content[start:end]
                
                parsed_data = NeuronTopic.model_validate_json(content)
                return parsed_data.model_dump()
                
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                else:
                    print(f"All {MAX_RETRIES} attempts failed. Last error: {e}")
                    raise

async def main():
    import sys
    from tqdm.asyncio import tqdm_asyncio
    
    # Default CSV path or accept from command line
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = 'neuron_top20_titles_layer16_expert0_w3.csv'
    
    # Read CSV file
    df = pd.read_csv(csv_path).head(200)  # Remove .head() for full processing
    
    print(f"Loaded {len(df)} neurons from {csv_path}")
    print(f"Processing with max {MAX_CONCURRENT_REQUESTS} concurrent requests...\n")
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # Process all rows with controlled concurrency
    tasks = [get_topics_from_vllm(row, semaphore) for idx, row in df.iterrows()]
    results = await tqdm_asyncio.gather(*tasks, desc="Processing neurons")
    
    # Save results to JSON file
    output_file = csv_path.replace('.csv', '_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"COMPLETE! Results saved to {output_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())
