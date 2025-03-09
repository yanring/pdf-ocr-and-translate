#!/usr/bin/env python
"""
Test script to demonstrate how to use the pdf_ocr_translate module programmatically.
"""

import os
from dotenv import load_dotenv
from pdf_ocr_translate import process_pdf, translate_markdown


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Get API keys from environment variables
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL")  # 获取自定义OpenAI API端点
    openai_model = os.getenv(
        "DEFAULT_OPENAI_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct"
    )  # 获取默认模型
    target_language = os.getenv(
        "DEFAULT_TARGET_LANGUAGE", "English"
    )  # 获取默认目标语言

    # 默认使用双语模式
    bilingual_mode = True

    # 并发和速率限制设置
    max_concurrency = int(os.getenv("MAX_CONCURRENCY", "10"))  # 并发请求数
    rate_limit = int(os.getenv("RATE_LIMIT", "1000"))  # 每分钟最大请求数

    # SSL 验证设置
    verify_ssl = os.getenv("VERIFY_SSL", "true").lower() != "false"  # 默认启用 SSL 验证

    if not mistral_api_key:
        print("Error: MISTRAL_API_KEY environment variable is not set.")
        return

    # Example PDF file path
    pdf_path = "example.pdf"

    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found.")
        return

    try:
        # Step 1: Process PDF with OCR
        print(f"Processing PDF: {pdf_path}")
        output_dir = "test_results"
        markdown_path = process_pdf(pdf_path, mistral_api_key, output_dir)

        # Step 2: Translate markdown (optional)
        if openai_api_key:
            import os

            os.environ["OPENAI_API_KEY"] = openai_api_key

            print(
                f"Translating extracted text to {target_language} using model: {openai_model}"
            )
            print(
                f"Translation mode: {'Bilingual' if bilingual_mode else 'Translation only'}"
            )
            print(
                f"Using concurrent translation with {max_concurrency} parallel requests"
            )
            print(f"Rate limit: {rate_limit} requests per minute")
            print(f"SSL verification: {'Disabled' if not verify_ssl else 'Enabled'}")

            translated_path = translate_markdown(
                markdown_path,
                target_language=target_language,
                api_provider="openai",
                model=openai_model,
                base_url=openai_base_url,  # 使用自定义OpenAI API端点
                bilingual=bilingual_mode,  # 使用双语模式
                max_concurrency=max_concurrency,  # 设置并发数
                requests_per_minute=rate_limit,  # 设置速率限制
                verify_ssl=verify_ssl,  # SSL 验证设置
            )
            print(f"Translation complete. Result saved to: {translated_path}")
        else:
            print("OpenAI API key not set. Skipping translation.")

        print("All operations completed successfully!")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
