# PDF OCR and Translation Tool

A powerful tool for extracting text from academic PDFs and translating the content while preserving formatting and images.

## Features

- Convert PDF documents to markdown using Mistral OCR API
- Extract and preserve images from PDF documents
- Translate content using large language models
- Maintain original formatting, tables, and technical elements
- Support for multiple translation providers (OpenAI, Mistral)
- Automatic chunking for handling large documents

## Requirements

- Python 3.8 or later
- Mistral AI API key
- OpenAI API key (optional, if using OpenAI for translation)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/pdf-ocr-and-translate.git
cd pdf-ocr-and-translate
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file from the template:
```bash
cp .env.example .env
```

4. Edit the `.env` file to add your API keys:
```
MISTRAL_API_KEY=your_mistral_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### Environment Configuration

You can configure default behavior by setting these environment variables in your `.env` file:

```
# API Keys
MISTRAL_API_KEY=your_mistral_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# API Configuration
OPENAI_BASE_URL=https://api.siliconflow.cn/v1  # Optional custom endpoint

# Default Settings
DEFAULT_TARGET_LANGUAGE=Chinese  # Default target language for translation
DEFAULT_OPENAI_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct  # Default model for OpenAI provider
VERIFY_SSL=false  # Disable SSL certificate verification if you have issues
```

## Usage

### Basic Usage

Process a PDF and translate it to the default language (from environment):

```bash
python pdf_ocr_translate.py your_document.pdf
```

### OCR Only (No Translation)

If you only want to extract text without translation:

```bash
python pdf_ocr_translate.py your_document.pdf --ocr-only
```

### Translation Modes

By default, the tool uses bilingual mode (translation followed by original text):

```bash
# Default: Bilingual mode
python pdf_ocr_translate.py your_document.pdf

# Translation only mode (no original text)
python pdf_ocr_translate.py your_document.pdf --no-bilingual
```

The bilingual mode formats output like this:
```markdown
This is the translated text in the target language.

> This is the original text in the source language.
```

### Translate to a Specific Language

```bash
python pdf_ocr_translate.py your_document.pdf --target-language "Chinese"
```

### Using Different Translation Providers

Use OpenAI (default):
```bash
python pdf_ocr_translate.py your_document.pdf --api-provider openai
```

Use Mistral:
```bash
python pdf_ocr_translate.py your_document.pdf --api-provider mistral
```

### Specifying a Model

For OpenAI:
```bash
python pdf_ocr_translate.py your_document.pdf --api-provider openai --model gpt-4o
```

For Mistral:
```bash
python pdf_ocr_translate.py your_document.pdf --api-provider mistral --model mistral-large-latest
```

### Specifying Output Directory

```bash
python pdf_ocr_translate.py your_document.pdf --output-dir custom_output
```

### Full Options

```
usage: pdf_ocr_translate.py [-h] [--target-language TARGET_LANGUAGE]
                           [--output-dir OUTPUT_DIR]
                           [--mistral-api-key MISTRAL_API_KEY]
                           [--openai-api-key OPENAI_API_KEY]
                           [--openai-base-url OPENAI_BASE_URL]
                           [--model MODEL]
                           [--api-provider {openai,mistral}]
                           [--ocr-only]
                           [--debug]
                           [--max-retries MAX_RETRIES]
                           [--rate-limit RATE_LIMIT]
                           [--no-bilingual]
                           pdf_path

PDF OCR and Translation Tool

positional arguments:
  pdf_path              Path to the PDF file

optional arguments:
  -h, --help            show this help message and exit
  --target-language TARGET_LANGUAGE, -t TARGET_LANGUAGE
                        Target language for translation
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Custom output directory
  --mistral-api-key MISTRAL_API_KEY
                        Mistral API key (can also be set via MISTRAL_API_KEY env var)
  --openai-api-key OPENAI_API_KEY
                        OpenAI API key (can also be set via OPENAI_API_KEY env var)
  --openai-base-url OPENAI_BASE_URL
                        Custom OpenAI API base URL (can also be set via OPENAI_BASE_URL env var)
  --model MODEL         Model to use for translation
  --api-provider {openai,mistral}
                        API provider for translation (default: openai)
  --ocr-only            Only perform OCR without translation
  --debug               Enable debug mode for detailed error information
  --max-retries MAX_RETRIES
                        Maximum number of retry attempts for failed API requests (default: 10)
  --rate-limit RATE_LIMIT
                        Maximum API requests per minute (default: 300, which is 5 per second)
  --no-bilingual        Disable bilingual mode (translation only, without original text)
```

### Using a Custom OpenAI-compatible API Endpoint

You can use a custom OpenAI-compatible API endpoint (like SiliconFlow):

```bash
python pdf_ocr_translate.py your_document.pdf --api-provider openai --openai-base-url "https://api.siliconflow.cn/v1" --model "Qwen/Qwen2.5-Coder-32B-Instruct"
```

Or set it in your `.env` file:
```
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
DEFAULT_OPENAI_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
```

### Handling API Rate Limits and Connection Issues

The tool includes built-in rate limiting and concurrent processing:

```bash
# Set custom rate limits (requests per minute)
python pdf_ocr_translate.py your_document.pdf --rate-limit 1000  # Default is 1000 requests per minute

# Set concurrency level (number of parallel requests)
python pdf_ocr_translate.py your_document.pdf --concurrency 20  # Default is 10 parallel requests

# Set maximum retry attempts for failed API requests (default is 10)
python pdf_ocr_translate.py your_document.pdf --max-retries 15

# Disable SSL certificate verification (use if you have SSL connection errors)
python pdf_ocr_translate.py your_document.pdf --no-verify-ssl
```

For persistent connection issues, detailed error information will be displayed to help with debugging.

### SSL Certificate Issues

If you encounter SSL certificate verification errors with a custom API endpoint, you have several options:

```bash
# Test API connectivity with detailed diagnostics
./test_api_connection.py --show-env

# Test with SSL verification disabled (use only for testing, less secure)
./test_api_connection.py --no-verify-ssl

# When running the main program, disable SSL verification
python pdf_ocr_translate.py your_document.pdf --no-verify-ssl
```

You can also permanently disable SSL verification in your `.env` file:
```
VERIFY_SSL=false
```

Common solutions for SSL certificate issues:
1. Update your system's CA certificates
2. Ensure your system time is correct
3. If using a self-signed certificate, consider adding it to your trusted certificates
4. Use the `--no-verify-ssl` option as a last resort for testing purposes

### Translation Processing

The tool now processes translations using an intelligent concurrent approach:
- Paragraphs are processed in parallel for faster translation
- Built-in rate limiting ensures API limits are not exceeded
- Automatically adjusts processing speed to stay within limits
- Results are properly ordered regardless of completion time
- Progress bars show both task preparation and translation progress

## Output

The tool creates a directory (by default named `results_<pdf_filename>`) containing:

- `original.md`: The original extracted markdown content
- `translated_<language>.md`: The translated content (if translation was requested)
- `images/`: Directory containing all extracted images from the PDF

## Notes

- OCR quality depends on the PDF content and formatting
- Translation quality varies by provider and model used
- For best results with complex academic content, use more advanced models like GPT-4o
- API usage may incur costs according to the respective provider pricing

## License

[Apache License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.