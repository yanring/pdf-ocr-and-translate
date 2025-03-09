#!/usr/bin/env python
"""
Test script to verify API connectivity to OpenAI or compatible endpoints.
This can be used to diagnose connection issues.
"""

import os
import sys
import time
import random
import requests
import argparse
from dotenv import load_dotenv
from openai import OpenAI


def print_env_info():
    """Print current environment variable configuration."""
    print("\n===== Current Environment Configuration =====")

    # List of environment variables to check
    env_vars = {
        "OPENAI_API_KEY": "OpenAI API Key",
        "OPENAI_BASE_URL": "OpenAI Base URL",
        "DEFAULT_OPENAI_MODEL": "Default OpenAI Model",
        "MISTRAL_API_KEY": "Mistral API Key",
    }

    for var, description in env_vars.items():
        value = os.getenv(var)
        if value:
            # Mask API keys for security
            if "API_KEY" in var:
                masked_value = (
                    value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
                )
                print(f"{description}: {masked_value} (length: {len(value)})")
            else:
                print(f"{description}: {value}")
        else:
            print(f"{description}: Not set")


def test_openai_connectivity(
    api_key, base_url=None, model=None, max_retries=10, verify_ssl=True
):
    """Test connectivity to OpenAI API or compatible endpoint."""
    print("\n===== API Connectivity Test =====")

    # 1. Check API Key
    if not api_key:
        print("❌ Error: No API key provided")
        return False
    else:
        print(f"✅ API key provided (length: {len(api_key)})")

    # 2. Check Basic HTTP Connectivity to endpoint
    if base_url:
        print(f"🔍 Testing connectivity to custom endpoint: {base_url}")
        retry_count = 0
        while retry_count <= max_retries:
            try:
                # Try a simple GET request first
                response = requests.get(base_url, timeout=10, verify=verify_ssl)
                print(f"✅ Basic HTTP connectivity: Status code {response.status_code}")
                break
            except Exception as e:
                retry_count += 1
                print(f"❌ Connection attempt {retry_count} failed: {e}")

                if retry_count > max_retries:
                    print(
                        "   Please check your internet connection and API endpoint URL."
                    )
                    if "SSL" in str(e) or "certificate" in str(e).lower():
                        print(
                            "\n⚠️ SSL Certificate verification failed. You may need to:"
                        )
                        print(
                            "   1. Check your system's certificate authority settings"
                        )
                        print("   2. Update your CA certificates")
                        print(
                            "   3. Try running with --no-verify-ssl option if you want to bypass verification (less secure)"
                        )
                    return False

                # Add backoff before retry
                backoff_time = 2 + random.random() * 3
                print(f"   Retrying in {backoff_time:.1f} seconds...")
                time.sleep(backoff_time)
    else:
        print("🔍 Using default OpenAI API endpoint")

    # 3. Attempt a simple API call
    retry_count = 0
    while retry_count <= max_retries:
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)

            # Use provided model or a default one
            model_to_use = model or "gpt-3.5-turbo"
            print(f"🔍 Testing API call with model: {model_to_use}")

            # Simple completion request
            try:
                print(f"   Making API request to: {base_url or 'api.openai.com'}")
                completion = client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Say hello in one word."},
                    ],
                    max_tokens=10,
                )

                response_text = completion.choices[0].message.content.strip()
                print(f"✅ API call successful. Response: {response_text}")
                return True
            except Exception as api_error:
                retry_count += 1
                print(
                    f"❌ API request failed (attempt {retry_count}/{max_retries+1}): {api_error}"
                )

                # Try to extract useful error information
                error_msg = str(api_error)
                if "invalid_api_key" in error_msg.lower():
                    print("   👉 The API key appears to be invalid.")
                elif "model" in error_msg.lower() and "exist" in error_msg.lower():
                    print(
                        f"   👉 The model '{model_to_use}' might not exist or is not available."
                    )
                elif "exceeded" in error_msg.lower() or "limit" in error_msg.lower():
                    print("   👉 You may have exceeded your API rate limits.")

                # Print detailed error information
                print("\n   --- Detailed Error Information ---")
                print(f"   Error type: {type(api_error).__name__}")
                print(f"   Error message: {str(api_error)}")

                # Print underlying cause if available
                if hasattr(api_error, "__cause__") and api_error.__cause__ is not None:
                    print(
                        f"   Caused by: {type(api_error.__cause__).__name__}: {str(api_error.__cause__)}"
                    )

                # More detailed error information
                if hasattr(api_error, "response"):
                    if hasattr(api_error.response, "status_code"):
                        print(f"   Status code: {api_error.response.status_code}")
                    if hasattr(api_error.response, "text"):
                        print(f"   Response content: {api_error.response.text}")
                print("   -------------------------------\n")

                # If we've reached max retries, stop trying
                if retry_count > max_retries:
                    return False

                # Add backoff before retry
                backoff_time = 2 + random.random() * 3
                print(f"   Retrying in {backoff_time:.1f} seconds...")
                time.sleep(backoff_time)

        except Exception as e:
            print(f"❌ Client initialization failed: {e}")
            return False


def main():
    """Main function for API connectivity testing."""
    # Load environment variables
    load_dotenv()

    parser = argparse.ArgumentParser(description="Test API connectivity")
    parser.add_argument(
        "--api-key", type=str, help="API key to use (default: from environment)"
    )
    parser.add_argument(
        "--base-url", type=str, help="Base URL for the API (default: from environment)"
    )
    parser.add_argument(
        "--model", type=str, help="Model to test (default: from environment)"
    )
    parser.add_argument(
        "--show-env", action="store_true", help="Show current environment variables"
    )
    parser.add_argument(
        "--max-retries", type=int, default=10, help="Maximum number of retry attempts"
    )
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        default=True,
        help="Disable SSL certificate verification (less secure)",
    )

    args = parser.parse_args()

    # Show environment info if requested
    if args.show_env:
        print_env_info()
        return 0

    # Get configuration from environment variables or command line arguments
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    model = args.model or os.getenv("DEFAULT_OPENAI_MODEL")

    # Print environment configuration
    print("Testing with current environment configuration:")
    print(f"API Base URL: {base_url or 'Default OpenAI API'}")
    print(f"Model: {model or 'Not specified (will use default)'}")
    print(f"Max retries: {args.max_retries}")
    print(f"SSL verification: {'Disabled' if args.no_verify_ssl else 'Enabled'}")

    # Check if we have an API key
    if not api_key:
        print(
            "\n❌ Error: No API key provided. Please set OPENAI_API_KEY environment variable or use --api-key."
        )
        return 1

    # Run the test
    success = test_openai_connectivity(
        api_key, base_url, model, args.max_retries, not args.no_verify_ssl
    )

    if success:
        print("\n✅ API Connectivity Test: SUCCESSFUL")
        return 0
    else:
        print("\n❌ API Connectivity Test: FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
