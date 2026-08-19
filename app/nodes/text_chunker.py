# WHAT: Groups and splits timestamped transcript segments into logical, sized text chunks
# WHY: LLMs have token limits and work best with concise context. Splitting the transcript into chunks
#      allows us to find and send only the most relevant parts of the video to the LLM during Q&A.
# INPUT: segments (List[Dict[str, Any]]) - the list of timestamped segments from Node 3
# OUTPUT: chunks (List[str]) - list of text chunks formatted with their time ranges (e.g. "[00:00:00 - 00:01:15] Hello and welcome...")
# WHO CALLS: app/graphs/video_processing_graph.py (as the fourth node in the pipeline, after transcriber)
# CONNECTED TO: Python standard libraries, LangChain text splitter concept

from typing import List, Dict, Any


def format_time(seconds: float) -> str:
    """
    What this does:
        Converts raw seconds (float) into a standard HH:MM:SS string format.
    
    Why it exists:
        Raw seconds (like 125.5) are hard for users to read. HH:MM:SS (00:02:05)
        is the standard format for video player timelines.
    
    Output:
        Returns formatted time string (e.g. "00:02:05")
    """
    # Use divmod to calculate hours, minutes, and seconds from total seconds
    mins, secs = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    return f"{hrs:02}:{mins:02}:{secs:02}"


def chunk_transcript(segments: List[Dict[str, Any]], chunk_size: int = 1000) -> List[str]:
    """
    What this does:
        Takes the raw transcript segments and groups them into larger text blocks (chunks)
        of approximately `chunk_size` characters, while prefixing each chunk with its timestamp range.

    Why it exists:
        Instead of indexing single sentences (too small) or the entire video (too large),
        we group adjacent sentences together into logical paragraphs. Adding the timestamp range
        allows the LLM to tell the user exactly where in the video a certain topic was discussed.

    Output:
        Returns a list of formatted strings.
        Example output:
        [
            "[00:00:00 - 00:01:15] Hello and welcome to the course. Today we talk about blockchain...",
            "[00:01:15 - 00:02:30] A blockchain is a distributed ledger database..."
        ]

    Raises:
        ValueError: if input segments is not a list, or chunk_size is invalid (<= 0)
    """

    # Step 1: Validate input arguments
    if not isinstance(segments, list):
        raise ValueError("Input 'segments' must be a list of dictionaries.")
        
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer greater than 0.")

    # If the transcript is completely empty, return an empty list immediately
    if not segments:
        return []

    chunks = []
    current_segment_group = []
    current_char_count = 0

    # Step 2: Loop through each segment to group them based on text length
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        # Add the segment to our current active group
        current_segment_group.append(seg)
        current_char_count += len(text)

        # Step 3: Once we reach the target chunk_size, package this group into a chunk
        if current_char_count >= chunk_size:
            # Determine the starting timestamp of the first segment in this chunk
            start_time = format_time(current_segment_group[0]["start"])
            # Determine the ending timestamp of the last segment in this chunk
            end_time = format_time(current_segment_group[-1]["end"])
            
            # Combine all the text from the segments in this group
            combined_text = " ".join([s["text"].strip() for s in current_segment_group])
            
            # Format the final chunk with timestamps at the beginning
            formatted_chunk = f"[{start_time} - {end_time}] {combined_text}"
            chunks.append(formatted_chunk)

            # Reset the group and character count for the next chunk
            current_segment_group = []
            current_char_count = 0

    # Step 4: Handle any remaining segments that didn't reach the chunk_size limit
    if current_segment_group:
        start_time = format_time(current_segment_group[0]["start"])
        end_time = format_time(current_segment_group[-1]["end"])
        combined_text = " ".join([s["text"].strip() for s in current_segment_group])
        formatted_chunk = f"[{start_time} - {end_time}] {combined_text}"
        chunks.append(formatted_chunk)

    return chunks
