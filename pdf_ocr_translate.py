#!/usr/bin/env python
"""
PDF OCR and Translation Tool
----------------------------
This tool extracts text from academic PDFs using Mistral OCR
and translates the content using LLM services while preserving formatting.
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import openai
from dotenv import load_dotenv
from mistralai import DocumentURLChunk, Mistral
from mistralai.models import OCRResponse
from tqdm import tqdm
from openai import OpenAI
import requests


def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    """Replace image placeholders in markdown with actual image paths."""
    for img_name, img_path in images_dict.items():
        markdown_str = markdown_str.replace(
            f"![{img_name}]({img_name})", f"![{img_name}]({img_path})"
        )
    return markdown_str


def save_ocr_results(ocr_response: OCRResponse, output_dir: str) -> str:
    """Save OCR results (markdown text and images) to the specified directory."""
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    all_markdowns = []
    for page in ocr_response.pages:
        # Save images
        page_images = {}
        for img in page.images:
            img_data = base64.b64decode(img.image_base64.split(",")[1])
            img_path = os.path.join(images_dir, f"{img.id}.png")
            with open(img_path, "wb") as f:
                f.write(img_data)
            page_images[img.id] = f"images/{img.id}.png"

        # Process markdown content
        page_markdown = replace_images_in_markdown(page.markdown, page_images)
        all_markdowns.append(page_markdown)

    # Save complete markdown
    markdown_path = os.path.join(output_dir, "original.md")
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_markdowns))

    return markdown_path


def process_pdf(
    pdf_path: str, mistral_api_key: str, output_dir: Optional[str] = None
) -> str:
    """
    Process a PDF file using Mistral OCR API.

    Args:
        pdf_path: Path to the PDF file
        mistral_api_key: Mistral API key
        output_dir: Custom output directory (optional)

    Returns:
        Path to the generated markdown file
    """
    # Initialize client
    client = Mistral(api_key=mistral_api_key)

    # Confirm PDF file exists
    pdf_file = Path(pdf_path)
    if not pdf_file.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Create output directory name
    if output_dir is None:
        output_dir = f"results_{pdf_file.stem}"

    print(f"Processing PDF: {pdf_file.name}")

    # Upload and process PDF
    uploaded_file = client.files.upload(
        file={
            "file_name": pdf_file.stem,
            "content": pdf_file.read_bytes(),
        },
        purpose="ocr",
    )

    signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
    print("Running OCR processing...")
    pdf_response = client.ocr.process(
        document=DocumentURLChunk(document_url=signed_url.url),
        model="mistral-ocr-latest",
        include_image_base64=True,
    )

    # Save results
    markdown_path = save_ocr_results(pdf_response, output_dir)
    print(f"OCR processing complete. Results saved to: {output_dir}")

    return markdown_path


def split_markdown_into_chunks(
    markdown_path: str, max_tokens: int = 4000, by_paragraph: bool = True
) -> List[str]:
    """
    Split a markdown file into chunks for translation.

    This function preserves paragraphs and content structure while ensuring
    chunks don't exceed the maximum token limit for API calls.
    Sections with heading "References" (or similar) will be skipped.

    Args:
        markdown_path: Path to the markdown file
        max_tokens: Maximum tokens per chunk (not used if by_paragraph is True)
        by_paragraph: If True, split by paragraphs regardless of token count

    Returns:
        List of markdown chunks
    """
    with open(markdown_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by paragraphs to preserve structure
    paragraphs = content.split("\n\n")

    # Filter out empty paragraphs
    paragraphs = [p for p in paragraphs if p.strip()]
    
    # Filter out references sections (typically starts with "# References" or similar)
    in_references_section = False
    filtered_paragraphs = []
    for p in paragraphs:
        # Check if this paragraph is a "References" heading (or similar)
        if p.strip().startswith("# Reference") or p.strip() == "# References" or p.strip() == "# Bibliography":
            in_references_section = True
            continue
        
        # If we're in a references section, skip this paragraph
        if in_references_section:
            continue
            
        # Otherwise, include it
        filtered_paragraphs.append(p)

    paragraphs = filtered_paragraphs

    if by_paragraph:
        # Return each paragraph as a separate chunk
        return paragraphs
    else:
        # Use token-based chunking
        chunks = []
        current_chunk = []
        current_token_count = 0

        for para in paragraphs:
            # Rough estimation of tokens (characters/4 is a common approximation)
            para_token_count = len(para) // 4

            if current_token_count + para_token_count > max_tokens and current_chunk:
                # Current chunk would exceed token limit, save it and start a new one
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_token_count = para_token_count
            else:
                # Add paragraph to current chunk
                current_chunk.append(para)
                current_token_count += para_token_count

        # Add the last chunk if there's anything
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks


async def translate_text_openai_async(
    text: str,
    target_language: str,
    model: str = "gpt-4o",
    base_url: Optional[str] = None,
    max_retries: int = 100,
    bilingual: bool = True,
    client: Optional[OpenAI] = None,
    verify_ssl: bool = True,
) -> str:
    """
    Translate text using OpenAI's API (async version).

    Args:
        text: Text to translate
        target_language: Target language (e.g., 'English', 'Chinese')
        model: OpenAI model to use
        base_url: Optional custom API endpoint URL
        max_retries: Maximum number of retry attempts for failed requests
        bilingual: If True, output will contain both original text and translation
        client: Optional pre-initialized OpenAI client
        verify_ssl: Whether to verify SSL certificates

    Returns:
        Translated text
    """
    # Skip translation for content that doesn't need translation (images, code blocks, formulas)
    # Check if the content is an image, a code block, or a LaTeX formula
    if (
        text.strip().startswith("![") and text.strip().endswith(")")  # Image
        or (text.strip().startswith("```") and text.strip().endswith("```"))  # Code block
        or (text.strip().startswith("$$") and text.strip().endswith("$$"))  # LaTeX formula block
        or (text.strip().startswith("$") and text.strip().endswith("$") and not text.count("$") > 2)  # Inline LaTeX
    ):
        return text  # Return original content without translation
    
    # Always ask the model to only generate the translation
    prompt = f"""Translate the following academic content into {target_language}. 
Maintain the original formatting including headers, lists, tables, and code blocks.
Do not translate code content, variable names, or technical terms that should remain in their original form.
Preserve all markdown syntax including links, images, and formatting.

Content to translate:

{text}
"""

    # Use provided client or create new one
    if client is None:
        # Configure client with SSL verification option
        import httpx

        http_client = httpx.Client(verify=verify_ssl)
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
            http_client=http_client,
        )

    retry_count = 0
    while retry_count <= max_retries:
        try:
            # Attempt the API call
            completion = await asyncio.to_thread(
                client.chat.completions.create,
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional academic translator specializing in technical content.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )

            translation = completion.choices[0].message.content
            
            # Handle bilingual output format here in the code instead of in the LLM prompt
            if bilingual:
                # First the original text, then the translation
                return f"{text}\n\n{translation}"
            else:
                return translation

        except Exception as e:
            retry_count += 1

            # Extract and print detailed error information
            print(
                f"\n============ ERROR DETAILS (Attempt {retry_count}/{max_retries}) ============"
            )
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")

            # Print underlying cause if available
            if hasattr(e, "__cause__") and e.__cause__ is not None:
                print(f"Caused by: {type(e.__cause__).__name__}: {str(e.__cause__)}")

            # Provide error details
            if hasattr(e, "response") and hasattr(e.response, "status_code"):
                print(f"API Error: Status code {e.response.status_code}")
                if hasattr(e.response, "text"):
                    print(f"Response content: {e.response.text}")

            # Log connection-related issues
            if "Connection" in str(e):
                print(f"Connection issue: {e}")
                print(f"Attempted to connect to: {base_url or 'Default OpenAI API'}")
                print(
                    "Please check your internet connection and API endpoint configuration."
                )
                if "SSL" in str(e) or "certificate" in str(e).lower():
                    print(
                        "SSL certificate verification failed. Try using --no-verify-ssl option."
                    )

            print("============================================================\n")

            # If this was the last retry, re-raise the exception
            if retry_count > max_retries:
                raise

            # Calculate backoff time with jitter (between 2-5 seconds)
            backoff_time = 2 + random.random() * 3
            print(
                f"⚠️ Request failed. Retrying ({retry_count}/{max_retries}) after {backoff_time:.1f} seconds..."
            )
            await asyncio.sleep(backoff_time)


class RateLimiter:
    """Rate limiter for API requests to prevent exceeding API rate limits."""

    def __init__(self, requests_per_minute: int = 1000):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum number of requests per minute
        """
        self.requests_per_minute = requests_per_minute
        self.interval = 60.0 / requests_per_minute  # Time between requests
        self.last_request_time = 0
        self.lock = asyncio.Lock()
        self.tokens = requests_per_minute  # Token bucket
        self.last_refill = time.time()
        self.max_tokens = requests_per_minute

    async def acquire(self):
        """Acquire permission to make a request."""
        async with self.lock:
            now = time.time()

            # Refill tokens based on time elapsed
            time_passed = now - self.last_refill
            new_tokens = time_passed * (self.requests_per_minute / 60.0)
            self.tokens = min(self.max_tokens, self.tokens + new_tokens)
            self.last_refill = now

            if self.tokens < 1:
                # Not enough tokens, calculate wait time
                wait_time = (1 - self.tokens) * self.interval
                await asyncio.sleep(wait_time)
                self.tokens = 1  # Now we have 1 token

            # Consume one token
            self.tokens -= 1


async def translate_markdown_async(
    markdown_path: str,
    target_language: str,
    api_provider: str = "openai",
    model: str = None,
    api_key: str = None,
    base_url: Optional[str] = None,
    max_retries: int = 100,
    requests_per_minute: int = 1000,  # Default to maximum 1000 requests per minute
    bilingual: bool = True,
    max_concurrency: int = 10,  # Maximum number of concurrent requests
    verify_ssl: bool = True,  # Whether to verify SSL certificates
) -> str:
    """
    Translate the content of a markdown file asynchronously.

    Args:
        markdown_path: Path to the markdown file
        target_language: Target language (e.g., 'English', 'Chinese')
        api_provider: Translation API provider ('openai' only)
        model: Model to use for translation
        api_key: API key for the provider
        base_url: Optional custom API endpoint URL for OpenAI-compatible providers
        max_retries: Maximum number of retry attempts for failed API requests
        requests_per_minute: Maximum number of API requests per minute
        bilingual: If True, output will contain both original text and translation
        max_concurrency: Maximum number of concurrent requests
        verify_ssl: Whether to verify SSL certificates

    Returns:
        Path to the translated markdown file
    """
    # Set default models
    if model is None:
        model = os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4o")

    # Create output file path
    output_dir = os.path.dirname(markdown_path)
    output_file = os.path.join(output_dir, f"translated_{target_language.lower()}.md")

    # Split markdown into chunks (by paragraph)
    chunks = split_markdown_into_chunks(markdown_path, by_paragraph=True)
    print(f"Markdown split into {len(chunks)} paragraphs for translation")

    # Initialize rate limiter
    rate_limiter = RateLimiter(requests_per_minute)

    # Initialize translation client (reuse for all requests)
    # Configure client with SSL verification option
    import httpx

    http_client = httpx.Client(verify=verify_ssl)
    client = OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        base_url=base_url,
        http_client=http_client,
    )

    # Prepare semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrency)

    async def translate_chunk(i, chunk):
        # Skip empty chunks
        if not chunk.strip():
            return i, ""

        # Acquire semaphore to limit concurrency
        async with semaphore:
            # Acquire permission from rate limiter
            await rate_limiter.acquire()

            try:
                translated = await translate_text_openai_async(
                    chunk,
                    target_language,
                    model,
                    base_url,
                    max_retries=max_retries,
                    bilingual=bilingual,
                    client=client,
                    verify_ssl=verify_ssl,
                )
                return i, translated
            except Exception as e:
                print(f"Error translating chunk {i}: {e}")
                # Return original text if translation failed
                return i, chunk

    # Create and gather translation tasks
    tasks = []

    for i, chunk in tqdm(
        enumerate(chunks), desc="Preparing translation tasks", total=len(chunks)
    ):
        tasks.append(translate_chunk(i, chunk))

    print(
        f"Starting concurrent translation with max {max_concurrency} parallel requests..."
    )
    print(f"Rate limit: {requests_per_minute} requests per minute")
    print(f"SSL verification: {'Enabled' if verify_ssl else 'Disabled'}")

    # Use tqdm to show progress
    progress_bar = tqdm(total=len(chunks), desc=f"Translating to {target_language}")

    # Track the completed tasks and results
    results = {}
    pending = set(asyncio.create_task(task) for task in tasks)

    while pending:
        # Wait for some tasks to complete
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

        # Process completed tasks
        for task in done:
            i, translated = await task
            results[i] = translated
            progress_bar.update(1)

    progress_bar.close()

    # Combine translated chunks in original order
    translated_chunks = [results[i] for i in range(len(chunks))]
    translated_content = "\n\n".join(translated_chunks)

    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(translated_content)

    print(f"Translation complete. Translated file saved to: {output_file}")
    return output_file


def translate_markdown(
    markdown_path: str,
    target_language: str,
    api_provider: str = "openai",
    model: str = None,
    api_key: str = None,
    base_url: Optional[str] = None,
    max_retries: int = 100,
    requests_per_minute: int = 1000,  # Default to maximum 1000 requests per minute
    bilingual: bool = True,
    max_concurrency: int = 10,  # Maximum number of concurrent requests
    verify_ssl: bool = True,  # Whether to verify SSL certificates
) -> str:
    """
    Translate the content of a markdown file.

    Args:
        markdown_path: Path to the markdown file
        target_language: Target language (e.g., 'English', 'Chinese')
        api_provider: Translation API provider ('openai' only)
        model: Model to use for translation
        api_key: API key for the provider
        base_url: Optional custom API endpoint URL for OpenAI-compatible providers
        max_retries: Maximum number of retry attempts for failed API requests
        requests_per_minute: Maximum number of API requests per minute
        bilingual: If True, output will contain both original text and translation
        max_concurrency: Maximum number of concurrent requests
        verify_ssl: Whether to verify SSL certificates

    Returns:
        Path to the translated markdown file
    """
    return asyncio.run(
        translate_markdown_async(
            markdown_path,
            target_language,
            api_provider,
            model,
            api_key,
            base_url,
            max_retries,
            requests_per_minute,
            bilingual,
            max_concurrency,
            verify_ssl,
        )
    )


def main():
    """Parse command-line arguments and run PDF OCR and translation."""
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="PDF OCR and Translation Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "pdf_path", help="Path to the PDF file to process"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (defaults to pdf_filename_results)",
    )
    parser.add_argument(
        "--target-language",
        type=str,
        default="Chinese",
        help="Target language for translation (e.g., 'English', 'Chinese')",
    )
    parser.add_argument(
        "--ocr-only",
        action="store_true",
        help="Only perform OCR, skip translation",
    )
    parser.add_argument(
        "--mistral-api-key",
        type=str,
        help="Mistral API key (can also be set via MISTRAL_API_KEY env var)",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        help="OpenAI API key (can also be set via OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--openai-base-url",
        type=str,
        help="Custom base URL for OpenAI-compatible API endpoints",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use for translation (defaults to gpt-4o)",
    )
    parser.add_argument(
        "--api-provider",
        type=str,
        default="openai",
        choices=["openai"],
        help="API provider to use for translation",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=100,
        help="Maximum number of retry attempts for API calls",
    )
    parser.add_argument(
        "--no-bilingual",
        action="store_true",
        help="Output only translated text without original",
    )
    parser.add_argument(
        "--requests-per-minute",
        type=int,
        default=1000,
        help="Maximum API requests per minute",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=20,
        help="Maximum number of concurrent translation requests (default: 10)",
    )
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="Disable SSL certificate verification (use with caution)",
    )

    args = parser.parse_args()

    try:
        # Verify PDF file exists
        if not os.path.exists(args.pdf_path):
            raise FileNotFoundError(f"PDF file not found: {args.pdf_path}")

        # Get Mistral API key (prioritize command line arguments over environment variables)
        mistral_api_key = args.mistral_api_key or os.getenv("MISTRAL_API_KEY")
        if not mistral_api_key:
            raise ValueError(
                "Mistral API key is required. Set it via --mistral-api-key or MISTRAL_API_KEY env var."
            )

        # Process PDF with OCR
        markdown_path = process_pdf(args.pdf_path, mistral_api_key, args.output_dir)

        # Perform translation if requested
        if not args.ocr_only:
            api_key = None
            base_url = args.openai_base_url or os.getenv("OPENAI_BASE_URL")

            api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key is required for translation. Set it via --openai-api-key or OPENAI_API_KEY env var."
                )
            # Set OpenAI API key
            os.environ["OPENAI_API_KEY"] = api_key

            # Translate markdown to target language
            translate_markdown(
                markdown_path,
                args.target_language,
                api_provider=args.api_provider,
                model=args.model,
                api_key=api_key,
                base_url=base_url,
                max_retries=args.max_retries,
                bilingual=not args.no_bilingual,
                requests_per_minute=args.requests_per_minute,
                max_concurrency=args.max_concurrency,
                verify_ssl=not args.no_verify_ssl,
            )

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
