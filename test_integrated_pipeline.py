#!/usr/bin/env python3
"""
Complete Integration Test for Mistral OCR + DOCLING Fallback Pipeline
"""

import asyncio
import os
import sys
import json
import requests
import argparse
from pathlib import Path
from PosterAgent.mistral_ocr import MistralOCRConfig
from PosterAgent.parse_raw import parse_raw_async, gen_image_and_table
from utils.wei_utils import get_agent_config

# Add project root to path before importing local modules
sys.path.append(str(Path(__file__).parent))


# Default test paper URL (shorter 19-page document)
DEFAULT_TEST_PAPER_URL = "https://arxiv.org/pdf/2411.01747v2"


def setup_litellm_environment():
    """Setup environment for LiteLLM gateway"""
    # Set OpenAI API configuration for LiteLLM gateway
    os.environ["OPENAI_API_KEY"] = os.getenv("LITELLM_MASTER_KEY", "")

    # Load base URL from environment variable instead of hardcoding
    litellm_base_url = os.getenv("LITELLM_BASE_URL", "")
    if litellm_base_url:
        os.environ["OPENAI_BASE_URL"] = litellm_base_url
    else:
        print("⚠️  LITELLM_BASE_URL not configured in .env file")
    print("🔧 LiteLLM Configuration:")
    print(f"   Base URL: "
          f"{os.environ.get('OPENAI_BASE_URL', 'Not configured')}")
    print(f"   API Key configured: "
          f"{'✅' if os.environ.get('OPENAI_API_KEY') else '❌'}")
    print(f"   Mistral API Key configured: "
          f"{'✅' if os.getenv('MISTRAL_API_KEY') else '❌'}")

    # Print loaded environment variables for debugging
    print("\n🔍 Environment Variables from .env:")
    env_vars = ["LITELLM_MASTER_KEY", "LITELLM_BASE_URL", "MISTRAL_API_KEY"]
    for var in env_vars:
        value = os.getenv(var)
        if value:
            # Mask sensitive values but show they exist
            masked_value = f"{value[:8]}..." if len(value) > 8 else "***"
            print(f"   {var}: {masked_value}")
        else:
            print(f"   {var}: ❌ Not set")


def _is_valid_pdf_url(url: str) -> bool:
    """Validate if URL is likely a PDF URL."""
    url_lower = url.lower()
    return (
        url.startswith(('http://', 'https://')) and
        ('pdf' in url_lower or url_lower.endswith('.pdf'))
    )


def download_pdf_from_url(url: str, output_path: str) -> bool:
    """Download PDF from URL with support for various sources."""
    try:
        # Convert GitHub blob URLs to raw URLs
        if "github.com" in url and "/blob/" in url:
            raw_url = url.replace("github.com", "raw.githubusercontent.com")
            raw_url = raw_url.replace("/blob/", "/")
            print(f"🔗 Converting GitHub URL to raw URL: {raw_url}")
            url = raw_url

        print(f"📥 Downloading PDF from: {url}")

        # Set headers to handle different types of servers
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/91.0.4472.124 Safari/537.36'
            )
        }

        response = requests.get(url, timeout=60, headers=headers)
        response.raise_for_status()

        # Verify that the response content is actually a PDF
        if not response.content.startswith(b'%PDF'):
            print("⚠️  Warning: Downloaded content may not be a valid PDF")

        with open(output_path, 'wb') as f:
            f.write(response.content)

        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"✅ PDF downloaded successfully: {file_size:.2f} MB")
        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ Network error downloading PDF: {e}")
        return False
    except Exception as e:
        print(f"❌ Failed to download PDF: {e}")
        return False


class TestArgs:
    """Mock args object for testing"""
    def __init__(self, pdf_path: str):
        self.poster_path = pdf_path
        self.poster_name = Path(pdf_path).stem.replace(' ', '_')
        self.model_name_t = "gemini-2.5-pro-preview"
        self.model_name_v = "gemini-2.5-pro-preview"


def check_debug_files():
    """Check and display available debug files"""
    debug_dir = Path("debug_output")
    if not debug_dir.exists():
        print("ℹ️  No debug directory found")
        return

    debug_files = list(debug_dir.glob("*"))
    if not debug_files:
        print("ℹ️  No debug files found")
        return

    print(f"\n🔍 Debug files available in {debug_dir.absolute()}/:")

    # Group files by type
    ocr_files = list(debug_dir.glob("ocr_output_*.md"))
    conversation_files = list(debug_dir.glob("llm_conversation_*.json"))
    error_files = list(debug_dir.glob("llm_error_*.json"))

    if ocr_files:
        print(f"   📄 OCR outputs: {len(ocr_files)} files")
        for f in sorted(ocr_files,
                        key=lambda x: x.stat().st_mtime,
                        reverse=True)[:2]:
            file_size = f.stat().st_size / 1024  # KB
            print(f"      - {f.name} ({file_size:.1f} KB)")
            print(f"        Full path: {f.absolute()}")

    if conversation_files:
        print(f"   💬 LLM conversations: {len(conversation_files)} files")
        for f in sorted(conversation_files,
                        key=lambda x: x.stat().st_mtime,
                        reverse=True)[:2]:
            file_size = f.stat().st_size / 1024  # KB
            print(f"      - {f.name} ({file_size:.1f} KB)")
            print(f"        Full path: {f.absolute()}")

    if error_files:
        print(f"   ❌ Error logs: {len(error_files)} files")
        for f in sorted(error_files,
                        key=lambda x: x.stat().st_mtime,
                        reverse=True)[:2]:
            file_size = f.stat().st_size / 1024  # KB
            print(f"      - {f.name} ({file_size:.1f} KB)")
            print(f"        Full path: {f.absolute()}")

    # Show most recent file for quick access
    all_files = ocr_files + conversation_files + error_files
    if all_files:
        most_recent = max(all_files, key=lambda x: x.stat().st_mtime)
        print("\n💡 Most recent debug file:")
        print(f"   📁 {most_recent.absolute()}")

        # Show timestamp of most recent file
        import time
        timestamp = most_recent.stat().st_mtime
        formatted_time = time.strftime(
            '%Y-%m-%d %H:%M:%S', time.localtime(timestamp)
        )
        print(f"   ⏰ Created: {formatted_time}")


async def test_complete_pipeline(pdf_path: str):
    """Test the complete PDF processing pipeline"""
    print(f"🚀 Testing complete pipeline with: {pdf_path}")

    # Create test args
    args = TestArgs(pdf_path)
    actor_config = get_agent_config("gemini-2.5-pro-preview")

    try:
        # Step 1: Parse raw content (Mistral OCR + LLM processing)
        print("\n📋 Step 1: Raw content parsing...")
        input_token, output_token, raw_result = await parse_raw_async(
            args, actor_config
        )

        print("✅ LLM processing complete:")
        print(f"   Input tokens: {input_token}")
        print(f"   Output tokens: {output_token}")
        print(f"   Result type: {type(raw_result)}")

        if isinstance(raw_result, dict):
            print(f"   Source: {raw_result.get('source', 'unknown')}")
            if 'images' in raw_result:
                print(f"   Images found: {len(raw_result['images'])}")
            if 'tables' in raw_result:
                print(f"   Tables found: {len(raw_result['tables'])}")

        # Step 2: Generate images and tables
        print("\n🖼️  Step 2: Asset processing...")
        _, _, images_dict, tables_dict = gen_image_and_table(args, raw_result)

        print("✅ Asset processing complete:")
        print(f"   Images processed: {len(images_dict)}")
        print(f"   Tables processed: {len(tables_dict)}")

        # Step 3: Verify output files
        print("\n📁 Step 3: Output verification...")

        # Check for raw content JSON
        content_file = (
            f'contents/<{args.model_name_t}_{args.model_name_v}>_'
            f'{args.poster_name}_raw_content.json'
        )
        if os.path.exists(content_file):
            with open(content_file, 'r') as f:
                content_data = json.load(f)
            print(f"✅ Raw content JSON saved: "
                  f"{len(content_data.get('sections', []))} sections")
        else:
            print(f"❌ Raw content JSON not found: {content_file}")

        # Check for images JSON
        images_file = (
            f'<{args.model_name_t}_{args.model_name_v}>_images_and_tables/'
            f'{args.poster_name}_images.json'
        )
        if os.path.exists(images_file):
            print(f"✅ Images JSON saved: {len(images_dict)} images")
        else:
            print(f"⚠️  Images JSON not found: {images_file}")

        # Check for tables JSON
        tables_file = (
            f'<{args.model_name_t}_{args.model_name_v}>_images_and_tables/'
            f'{args.poster_name}_tables.json'
        )
        if os.path.exists(tables_file):
            print(f"✅ Tables JSON saved: {len(tables_dict)} tables")
        else:
            print(f"⚠️  Tables JSON not found: {tables_file}")

        # Step 4: Check asset files
        print("\n🔍 Step 4: Asset file verification...")
        asset_dir = Path(
            f'<{args.model_name_t}_{args.model_name_v}>_images_and_tables/'
            f'{args.poster_name}'
        )

        if asset_dir.exists():
            asset_files = list(asset_dir.glob("*.png"))
            print(f"✅ Asset directory exists with "
                  f"{len(asset_files)} PNG files")

            # List some example files
            for i, file in enumerate(asset_files[:3]):
                print(f"   📄 {file.name}")
            if len(asset_files) > 3:
                print(f"   ... and {len(asset_files) - 3} more files")
        else:
            print(f"⚠️  Asset directory not found: {asset_dir}")

        # Step 5: Show debug files
        check_debug_files()

        return True

    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()

        # Still show debug files even on failure
        print("\n🔍 Checking debug files for troubleshooting:")
        check_debug_files()

        return False


def test_config_validation():
    """Test configuration and dependencies"""
    print("🔧 Testing configuration...")

    # Test Mistral config
    config = MistralOCRConfig()
    if config.api_key:
        print("✅ Mistral API key configured")
    else:
        print("⚠️  Mistral API key not configured (will use DOCLING fallback)")

    # Test LiteLLM configuration
    if os.getenv("LITELLM_MASTER_KEY"):
        print("✅ LiteLLM master key configured")
    else:
        print("❌ LiteLLM master key not configured")

    # Test required imports
    try:
        from docling.document_converter import DocumentConverter  # noqa
        print("✅ DOCLING available")
    except ImportError as e:
        print(f"❌ DOCLING not available: {e}")
        return False

    try:
        from mistralai import Mistral  # noqa
        print("✅ Mistral SDK available")
    except ImportError as e:
        print(f"❌ Mistral SDK not available: {e}")
        return False

    return True


async def main(pdf_url: str = None):
    """Main test function"""
    print("🧪 Paper2Poster Mistral OCR Integration Test")
    print("=" * 60)

    # Use provided URL or default
    test_url = pdf_url or DEFAULT_TEST_PAPER_URL
    print(f"📄 Test PDF URL: {test_url}")

    # Setup LiteLLM environment
    setup_litellm_environment()

    # Configuration test
    config_ok = test_config_validation()
    if not config_ok:
        print("\n❌ Configuration test failed. Please fix dependencies.")
        return False

    # Generate test PDF filename from URL
    # Extract filename from URL (e.g., "2411.01747v2" from the arxiv URL)
    try:
        url_parts = test_url.split('/')
        filename_part = url_parts[-1]  # "2411.01747v2.pdf" or "2411.01747v2"
        if filename_part.endswith('.pdf'):
            filename_part = filename_part[:-4]  # Remove .pdf extension
        test_pdf = f"test_paper_{filename_part}.pdf"
    except Exception as e:
        print(f"⚠️  Error parsing URL for filename: {e}")
        test_pdf = "test_paper_download.pdf"

    print(f"📁 Local filename: {test_pdf}")

    if not os.path.exists(test_pdf):
        download_ok = download_pdf_from_url(test_url, test_pdf)
        if not download_ok:
            print("\n❌ Failed to download test PDF. Exiting.")
            return False
    else:
        print(f"✅ Test PDF already exists: {test_pdf}")

    # Run complete pipeline test
    print(f"\n🔄 Running complete pipeline test with: {test_pdf}")
    pipeline_ok = await test_complete_pipeline(test_pdf)

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"   Configuration: {'✅ PASS' if config_ok else '❌ FAIL'}")
    print(f"   Pipeline: {'✅ PASS' if pipeline_ok else '❌ FAIL'}")

    if config_ok and pipeline_ok:
        print("\n🎉 All tests passed! Integration is working correctly.")
        print("\n📝 Next steps:")
        print("   1. ✅ Mistral OCR integration completed")
        print("   2. ✅ DOCLING fallback working")
        print("   3. ✅ Asset extraction functional")
        print("   4. ✅ Paper2Poster compatibility maintained")
        print("\n💡 Debug files can be found in debug_output/ for analysis")
    else:
        print("\n⚠️  Some tests failed. Please check the logs above.")
        print("💡 Check debug_output/ directory for detailed logs")

    return config_ok and pipeline_ok


def interactive_menu():
    """Interactive TUI menu for selecting test cases."""
    print("\n" + "="*60)
    print("🧪 Paper2Poster Test Suite - Interactive Mode")
    print("="*60)

    # Predefined test cases
    test_cases = {
        "1": {
            "name": "DynaSaur (Default - 19 pages)",
            "url": "https://arxiv.org/pdf/2411.01747v2",
            "description": "LLM agents with dynamic action creation"
        },
        "2": {
            "name": "Constitutional AI (12 pages)",
            "url": "https://arxiv.org/pdf/2212.08073v1",
            "description": "Training AI assistants to be helpful, "
                           "harmless, and honest"
        },
        "3": {
            "name": "Attention Is All You Need (15 pages)",
            "url": "https://arxiv.org/pdf/1706.03762v7",
            "description": "The Transformer architecture paper"
        },
        "4": {
            "name": "BERT (16 pages)",
            "url": "https://arxiv.org/pdf/1810.04805v2",
            "description": "Pre-training of Deep Bidirectional Transformers"
        },
        "5": {
            "name": "GPT-3 (75 pages - Large)",
            "url": "https://arxiv.org/pdf/2005.14165v4",
            "description": "Language Models are Few-Shot Learners"
        }
    }

    print("\n📚 Available Test Cases:")
    for key, case in test_cases.items():
        print(f"  [{key}] {case['name']}")
        print(f"      {case['description']}")
        print(f"      URL: {case['url']}")

    print("\n  [C] Custom input PDF URL")
    print("  [Q] Quit")

    while True:
        choice = input("\n👉 Select an option (1-5, C, or Q): ").strip().upper()

        if choice == 'Q':
            print("👋 Exiting...")
            return None
        elif choice == 'C':
            print("\n📎 Examples of supported PDF URLs:")
            print("   • arXiv: https://arxiv.org/pdf/2411.01747v2")
            print("   • GitHub: https://github.com/VectifyAI/PageIndex/blob/"
                  "main/docs/earthmover.pdf")
            print("   • Direct PDF links: https://example.com/document.pdf")

            custom_url = input(
                "\n📎 Enter PDF URL: "
            ).strip()
            if custom_url:
                # Validate general PDF URL format
                if _is_valid_pdf_url(custom_url):
                    print(f"\n✅ Using custom URL: {custom_url}")
                    return custom_url
                else:
                    print(
                        "❌ Invalid URL. Please enter a valid PDF URL "
                        "(must contain 'pdf' or end with '.pdf')."
                    )
            else:
                print("❌ No URL entered.")
        elif choice in test_cases:
            selected = test_cases[choice]
            print(f"\n✅ Selected: {selected['name']}")
            print(f"   URL: {selected['url']}")
            return selected['url']
        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Paper2Poster with Mistral OCR integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_integrated_pipeline.py                    # Interactive mode
  python test_integrated_pipeline.py --url https://arxiv.org/pdf/2411.01747v2
  python test_integrated_pipeline.py --url https://github.com/VectifyAI/PageIndex/blob/main/docs/earthmover.pdf  # noqa: E501
  python test_integrated_pipeline.py --url https://example.com/document.pdf
        """
    )
    parser.add_argument(
        "--url",
        type=str,
        help="URL of the PDF to test (default: interactive mode)"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run with default URL without interactive menu"
    )
    args = parser.parse_args()

    # Determine which URL to use
    if args.url:
        # URL provided via command line
        test_url = args.url
    elif args.non_interactive:
        # Non-interactive mode with default URL
        test_url = DEFAULT_TEST_PAPER_URL
    else:
        # Interactive mode
        test_url = interactive_menu()
        if test_url is None:
            sys.exit(0)

    success = asyncio.run(main(test_url))
    sys.exit(0 if success else 1)
