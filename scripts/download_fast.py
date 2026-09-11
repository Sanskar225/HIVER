import os
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

URL = "https://huggingface.co/datasets/TNE-AI/customer-support-on-twitter-conversation/resolve/main/data/train-00000-of-00001.parquet"
OUTPUT_FILE = "data/raw/twcs_full.parquet"
NUM_CHUNKS = 12

def download_chunk(start, end, chunk_id):
    headers = {"Range": f"bytes={start}-{end}"}
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        resp = client.get(URL, headers=headers)
        if resp.status_code not in (200, 206):
            raise RuntimeError(f"Chunk {chunk_id} failed with status {resp.status_code}")
        return chunk_id, resp.content

def main():
    os.makedirs("data/raw", exist_ok=True)
    print("Fetching total file size...")
    with httpx.Client(follow_redirects=True) as client:
        head = client.head(URL)
        total_size = int(head.headers["content-length"])
    print(f"Total size: {total_size / 1024 / 1024:.2f} MB")

    chunk_size = total_size // NUM_CHUNKS
    ranges = []
    for i in range(NUM_CHUNKS):
        start = i * chunk_size
        end = total_size - 1 if i == NUM_CHUNKS - 1 else (i + 1) * chunk_size - 1
        ranges.append((start, end, i))

    t0 = time.time()
    results = {}
    print(f"Starting parallel download with {NUM_CHUNKS} threads...")

    with ThreadPoolExecutor(max_workers=NUM_CHUNKS) as executor:
        futures = [executor.submit(download_chunk, start, end, i) for start, end, i in ranges]
        for f in as_completed(futures):
            chunk_id, content = f.result()
            results[chunk_id] = content
            print(f"Chunk {chunk_id + 1}/{NUM_CHUNKS} downloaded ({len(content)/1024/1024:.1f} MB)")

    print("Assembling chunks into final file...")
    with open(OUTPUT_FILE, "wb") as out:
        for i in range(NUM_CHUNKS):
            out.write(results[i])

    dt = time.time() - t0
    final_size = os.path.getsize(OUTPUT_FILE)
    print(f"Downloaded {final_size / 1024 / 1024:.2f} MB in {dt:.1f}s ({final_size / 1024 / 1024 / dt:.2f} MB/s)")

if __name__ == "__main__":
    main()
