"""
Danbooru Character Fetcher

Fetches character data from Danbooru API and generates book1.xlsx.

Usage:
1. Edit DANBOORU_API_KEY below with your Danbooru API key
2. Run: python app/fetch_danbooru_characters.py

Features:
- Resumes from existing book1.xlsx (skips already-fetched characters)
- Fetches 20 posts per character tag
- Extracts top 20 associated tags by frequency
- Excludes 'solo' and meta tags
- Rate-limited API requests (1 second delay)
- Retry logic with exponential backoff
"""

import requests
import pandas as pd
import time
from collections import Counter
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================
# USER CONFIGURATION - EDIT THIS SECTION
# ============================================
DANBOORU_API_KEY = "your_api_key_here"  # Get from https://danbooru.donmai.us/user/edit

# Fetch settings
POSTS_PER_CHARACTER = 20
TOP_N_TAGS = 20
RATE_LIMIT_DELAY = 1.0  # Seconds between requests
MAX_RETRIES = 3

# ============================================
# PATHS
# ============================================
TAG_LIST_PATH = os.path.join(os.path.dirname(__file__), "tag_list.txt")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "book1.xlsx")


# ============================================
# API FUNCTIONS
# ============================================
def fetch_posts_for_tag(tag):
    """
    Fetch posts from Danbooru API for a specific character tag.
    Returns list of posts (each post is a dict).
    """
    api_url = "https://danbooru.donmai.us/posts.json"
    
    # Build query parameters
    params = {
        "tags": tag,
        "limit": POSTS_PER_CHARACTER
    }
    
    # Add API key for authentication (better rate limits)
    headers = {}
    if DANBOORU_API_KEY and DANBOORU_API_KEY != "your_api_key_here":
        # Danbooru uses Basic Auth with API key as username, password empty
        headers = {
            "Authorization": f"Bearer {DANBOORU_API_KEY}"
        }
    
    # Retry logic with exponential backoff
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(api_url, params=params, headers=headers, timeout=30)
            
            # Check for rate limiting
            if response.status_code == 429:
                wait_time = (2 ** attempt) * RATE_LIMIT_DELAY
                print(f"    [RATE LIMIT] Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = (2 ** attempt) * RATE_LIMIT_DELAY
                print(f"    [RETRY] Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
                print(f"    [RETRY] Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"    [ERROR] Failed after {MAX_RETRIES} attempts: {e}")
                return None
    
    return None


def extract_and_sort_tags(posts, character_tag):
    """
    Extract all tags from posts, count frequency, and return top N tags sorted by frequency.
    Excludes 'solo' and meta tags.
    """
    all_tags = []
    
    for post in posts:
        # Extract tag strings
        tag_string = post.get("tag_string", "")
        tags = tag_string.split()
        
        # Extract artist tags
        artist_string = post.get("tag_string_artist", "")
        artist_tags = artist_string.split()
        
        # Extract copyright tags
        copyright_string = post.get("tag_string_copyright", "")
        copyright_tags = copyright_string.split()
        
        # Extract meta tags (we'll exclude these)
        meta_string = post.get("tag_string_meta", "")
        meta_tags = meta_string.split()
        
        # Combine all tag lists for analysis
        all_tags.extend(tags)
        all_tags.extend(artist_tags)
        all_tags.extend(copyright_tags)
    
    # Count tag frequency
    tag_counts = Counter(all_tags)
    
    # Filter out unwanted tags
    unwanted_tags = {
        'solo',
        character_tag,  # Don't include the character tag itself in columns
    }
    
    # Meta tag patterns to exclude
    meta_prefixes = ('rating:', 'score:', 'user:', 'date:', 'fav:', 'pool:')
    
    filtered_counts = {}
    for tag, count in tag_counts.items():
        if tag in unwanted_tags:
            continue
        if tag.startswith(meta_prefixes):
            continue
        
        filtered_counts[tag] = count
    
    # Sort by frequency (descending) and take top N
    top_tags = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)[:TOP_N_TAGS]
    
    # Return just the tag strings (not the counts)
    return [tag for tag, count in top_tags]


def load_tag_list():
    """
    Load character tags from tag_list.txt.
    Returns list of tags (one per line).
    """
    if not os.path.exists(TAG_LIST_PATH):
        print(f"[ERROR] tag_list.txt not found: {TAG_LIST_PATH}")
        sys.exit(1)
    
    with open(TAG_LIST_PATH, 'r', encoding='utf-8') as f:
        tags = [line.strip() for line in f if line.strip()]
    
    print(f"[INFO] Loaded {len(tags)} character tags from tag_list.txt")
    return tags


def load_existing_data():
    """
    Load existing book1.xlsx to resume from where we left off.
    Returns dict of {character_tag: existing_tags_list} or None if file doesn't exist.
    """
    if not os.path.exists(OUTPUT_PATH):
        return None
    
    try:
        df = pd.read_excel(OUTPUT_PATH, header=None)
        
        # Convert to dict: {character_tag: [tags...]}
        existing_data = {}
        for _, row in df.iterrows():
            if pd.notna(row.iloc[0]):
                char_tag = str(row.iloc[0]).strip()
                # Get all non-null tags from this row
                tags = [str(val).strip() for val in row.iloc[1:] if pd.notna(val)]
                existing_data[char_tag] = tags
        
        print(f"[INFO] Loaded {len(existing_data)} existing characters from book1.xlsx")
        return existing_data
        
    except Exception as e:
        print(f"[WARNING] Could not load existing book1.xlsx: {e}")
        print("[WARNING] Will start fresh (overwrite existing file)")
        return None


def save_to_excel(data_dict):
    """
    Save data to book1.xlsx (no header row).
    data_dict: {character_tag: [tags...]}
    """
    # Convert to list of lists for DataFrame
    rows = []
    for char_tag, tags in data_dict.items():
        row = [char_tag] + tags
        rows.append(row)
    
    # Create DataFrame with no header
    df = pd.DataFrame(rows)
    
    # Save to Excel
    df.to_excel(OUTPUT_PATH, index=False, header=False)
    print(f"[INFO] Saved {len(data_dict)} characters to book1.xlsx")


def fetch_all_characters():
    """
    Main function: Fetch all characters from tag_list.txt.
    Resumes from existing book1.xlsx if it exists.
    """
    print("="*60)
    print("Danbooru Character Fetcher")
    print("="*60)
    
    # Check API key
    if DANBOORU_API_KEY == "your_api_key_here":
        print("\n[WARNING] No API key configured!")
        print("[WARNING] Please edit DANBOORU_API_KEY in this script.")
        print("[WARNING] Continuing with public API (lower rate limits)...\n")
    else:
        print(f"\n[INFO] Using API key: {DANBOORU_API_KEY[:10]}...{DANBOORU_API_KEY[-4:]}")
    
    # Load tags
    tags = load_tag_list()
    total_tags = len(tags)
    
    # Load existing data (for resume)
    existing_data = load_existing_data() or {}
    
    # Calculate progress
    fetched_count = 0
    skipped_count = len(existing_data)
    error_count = 0
    
    print(f"\n[INFO] Total tags to process: {total_tags}")
    print(f"[INFO] Already fetched: {skipped_count}")
    print(f"[INFO] Remaining: {total_tags - skipped_count}")
    print(f"[INFO] Fetching {POSTS_PER_CHARACTER} posts per character...")
    print(f"[INFO] Extracting top {TOP_N_TAGS} tags per character...")
    print("="*60)
    
    # Process each tag
    for idx, tag in enumerate(tags):
        tag_num = idx + 1
        
        # Skip if already in existing data
        if tag in existing_data:
            continue
        
        # Show progress
        print(f"[{tag_num}/{total_tags}] Fetching: {tag}")
        
        # Fetch posts
        posts = fetch_posts_for_tag(tag)
        
        if posts is None:
            print(f"    [ERROR] Failed to fetch posts for {tag}")
            error_count += 1
            # Save empty row (just character name)
            existing_data[tag] = []
        else:
            # Extract and sort tags
            sorted_tags = extract_and_sort_tags(posts, tag)
            
            print(f"    [OK] Found {len(posts)} posts, extracted {len(sorted_tags)} tags")
            
            # Store in data dict
            existing_data[tag] = sorted_tags
            fetched_count += 1
        
        # Save progress every 10 characters
        if fetched_count % 10 == 0:
            save_to_excel(existing_data)
            print(f"    [PROGRESS] Saved {len(existing_data)} characters to book1.xlsx")
        
        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)
    
    # Final save
    save_to_excel(existing_data)
    
    # Summary
    print("\n" + "="*60)
    print("FETCH COMPLETE")
    print("="*60)
    print(f"Total tags processed: {total_tags}")
    print(f"Already fetched: {skipped_count}")
    print(f"Newly fetched: {fetched_count}")
    print(f"Errors: {error_count}")
    print(f"Success rate: {(fetched_count + skipped_count) / total_tags * 100:.1f}%")
    print(f"\nOutput file: {OUTPUT_PATH}")
    print(f"Next step: Run 'python app/import_danbooru_characters.py' to import to database")


if __name__ == "__main__":
    fetch_all_characters()
