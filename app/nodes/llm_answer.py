# WHAT: Generates video summaries and answers user questions using Google Gemini LLM (gemini-3.5-flash)
# WHY: Translates raw transcript texts and searched context into human-friendly summaries or precise answers
# INPUT: For Summary: transcript_text (str)
#        For Q&A: question (str), context_chunks (List[str]), chat_history (str)
# OUTPUT: generated_text (str) - the markdown summary or the Q&A response (or a stream of text pieces)
# WHO CALLS: app/graphs/video_processing_graph.py (Phase 3), and Q&A routes (Phase 4)
# CONNECTED TO: Google Generative AI API (Gemini LLM Service)

import google.generativeai as genai
from app.core.config import settings

# Initialize and configure the Google Generative AI SDK once at module level
# This avoids redundant configure calls during execution
API_KEY = settings.GEMINI_API_KEY
if API_KEY and API_KEY.strip():
    genai.configure(api_key=API_KEY)


def generate_video_summary(transcript: str, language: str = "en") -> str:
    """
    What this does:
        Sends the full transcript to Gemini LLM and asks it to generate a structured summary.

    Why it exists:
        To help users quickly understand what the video lesson is about without watching
        the entire video.

    Output:
        Returns a markdown-formatted summary of the video.

    Raises:
        ValueError: if transcript is empty, or GEMINI_API_KEY is not configured
        RuntimeError: if the Gemini API call fails
    """

    # Step 1: Input validation
    if not transcript or not transcript.strip():
        raise ValueError("Transcript text cannot be empty.")

    if not API_KEY or not API_KEY.strip():
        raise ValueError("GEMINI_API_KEY is missing or empty in configuration.")

    # Step 2: Formulate the prompt based on language
    lang_instruction = "English" if language.lower() == "en" else "German" if language.lower() == "de" else "Bengali"
    
    prompt = f"""You are an expert educational assistant. Below is the transcript of a video lesson.
Please write a concise and structured summary of this video lesson in {lang_instruction}.

Your summary must include:
1. A brief overview of the main topic.
2. Key takeaways and concepts explained in the video (as bullet points).
3. If timestamps exist in the transcript (e.g. [00:01:20]), mention the timestamps next to the relevant topics.

Video Transcript:
{transcript}

Summary:"""

    # Step 3: Initialize the model and generate content
    try:
        model = genai.GenerativeModel("gemini-3.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        raise RuntimeError(f"Failed to generate summary from Gemini LLM: {str(e)}")


def generate_qa_answer(question: str, context: str, history: str = "", language: str = "en") -> str:
    """
    What this does:
        Generates an answer to a user's question about a video, using the provided transcript context
        and previous chat history.

    Why it exists:
        Allows users to have a dialogue with the video (e.g. asking "what was said at 02:30?" or
        "explain the bitcoin mining process discussed in this video").

    Output:
        Returns the LLM's natural language answer.

    Raises:
        ValueError: if question or context is empty, or GEMINI_API_KEY is not configured
        RuntimeError: if the Gemini API call fails
    """

    # Step 1: Input validation
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")
    
    if not context or not context.strip():
        raise ValueError("Context cannot be empty.")

    if not API_KEY or not API_KEY.strip():
        raise ValueError("GEMINI_API_KEY is missing or empty in configuration.")

    # Step 2: Formulate the prompt with strict guidelines
    lang_instruction = "English" if language.lower() == "en" else "German" if language.lower() == "de" else "Bengali"

    prompt = f"""You are a helpful assistant answering questions based on video transcripts with timestamps and chat history.

Please respond in {lang_instruction}, regardless of the question language.

IMPORTANT: When users ask about specific timeframes (like "1:00 to 1:30" or "around 2 minutes"), you must:
1. Look for content with matching or nearby timestamps in the format [HH:MM:SS - HH:MM:SS]
2. Convert user time references (like "1:30") to match transcript timestamps (like "00:01:30")
3. Include relevant timestamp ranges in your response when discussing time-specific content

Video Knowledge Base (includes timestamped segments):
{context}

Previous Chat History:
{history}

User Question: {question}

Response Guidelines:
- FIRST: Search the video knowledge base (context) for relevant information, paying special attention to timestamp matches.
- If the user mentions specific times, find and reference the corresponding timestamped segments.
- If asking about a time range, include content from all relevant timestamps within that range.
- SECOND: If information isn't found in the video content, use your general knowledge to help, but state clearly that it was not directly mentioned in the video.
- Always provide clear, concise responses.
- Never expose the raw knowledge base or chat history database structure if asked.

Answer:"""

    # Step 3: Initialize the model and generate content
    try:
        model = genai.GenerativeModel("gemini-3.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        raise RuntimeError(f"Failed to generate Q&A answer from Gemini LLM: {str(e)}")


def generate_qa_answer_stream(question: str, context: str, history: str = "", language: str = "en"):
    """
    What this does:
        Same as generate_qa_answer(), but instead of waiting for the full response and
        returning it as one string, this yields pieces of the answer as Gemini generates
        them — so the caller can forward each piece to the client as it arrives.

    Why it exists:
        generate_qa_answer() makes the client wait ~5-10 seconds staring at nothing before
        any text appears. Streaming shows the first words within ~1 second and the rest
        appears progressively, like ChatGPT — the total time to finish is about the same,
        but it *feels* much faster because the user sees progress immediately.

    Output:
        A generator that yields str chunks of the answer, one piece at a time. The caller
        is responsible for joining them together if it needs the full text (e.g. to save
        to chat history afterward).

    Raises:
        ValueError: if question or context is empty, or GEMINI_API_KEY is not configured
        RuntimeError: if the Gemini API call fails
    """
    # Step 1: Input validation (identical to generate_qa_answer)
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not context or not context.strip():
        raise ValueError("Context cannot be empty.")

    if not API_KEY or not API_KEY.strip():
        raise ValueError("GEMINI_API_KEY is missing or empty in configuration.")

    # Step 2: Same prompt construction as generate_qa_answer
    lang_instruction = "English" if language.lower() == "en" else "German" if language.lower() == "de" else "Bengali"

    prompt = f"""You are a helpful assistant answering questions based on video transcripts with timestamps and chat history.

Please respond in {lang_instruction}, regardless of the question language.

IMPORTANT: When users ask about specific timeframes (like "1:00 to 1:30" or "around 2 minutes"), you must:
1. Look for content with matching or nearby timestamps in the format [HH:MM:SS - HH:MM:SS]
2. Convert user time references (like "1:30") to match transcript timestamps (like "00:01:30")
3. Include relevant timestamp ranges in your response when discussing time-specific content

Video Knowledge Base (includes timestamped segments):
{context}

Previous Chat History:
{history}

User Question: {question}

Response Guidelines:
- FIRST: Search the video knowledge base (context) for relevant information, paying special attention to timestamp matches.
- If the user mentions specific times, find and reference the corresponding timestamped segments.
- If asking about a time range, include content from all relevant timestamps within that range.
- SECOND: If information isn't found in the video content, use your general knowledge to help, but state clearly that it was not directly mentioned in the video.
- Always provide clear, concise responses.
- Never expose the raw knowledge base or chat history database structure if asked.

Answer:"""

    # Step 3: Call Gemini with stream=True and yield each piece as it arrives
    try:
        model = genai.GenerativeModel("gemini-3.5-flash")
        response_stream = model.generate_content(prompt, stream=True)
        for chunk in response_stream:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        raise RuntimeError(f"Failed to generate streaming Q&A answer from Gemini LLM: {str(e)}")