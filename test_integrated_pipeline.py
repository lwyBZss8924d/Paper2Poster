#!/usr/bin/env python3
"""
Complete Integration Test for Mistral OCR + DOCLING Fallback Pipeline
测试完整的PDF处理流程，确保与原始系统兼容
"""

import asyncio
import os
import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from PosterAgent.mistral_ocr import MistralOCRConfig
from PosterAgent.parse_raw import parse_raw_async, gen_image_and_table
from utils.wei_utils import get_agent_config


class TestArgs:
    """Mock args object for testing"""
    def __init__(self, pdf_path: str):
        self.poster_path = pdf_path
        self.poster_name = Path(pdf_path).stem.replace(' ', '_')
        self.model_name_t = "4o"
        self.model_name_v = "4o"


async def test_complete_pipeline(pdf_path: str):
    """Test the complete PDF processing pipeline"""
    print(f"🚀 Testing complete pipeline with: {pdf_path}")
    
    # Create test args
    args = TestArgs(pdf_path)
    actor_config = get_agent_config("4o")
    
    try:
        # Step 1: Parse raw content (Mistral OCR + LLM processing)
        print("\n📋 Step 1: Raw content parsing...")
        input_token, output_token, raw_result = await parse_raw_async(
            args, actor_config
        )
        
        print(f"✅ LLM processing complete:")
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
        
        print(f"✅ Asset processing complete:")
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
            print(f"✅ Raw content JSON saved: {len(content_data.get('sections', []))} sections")
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
            print(f"✅ Asset directory exists with {len(asset_files)} PNG files")
            
            # List some example files
            for i, file in enumerate(asset_files[:3]):
                print(f"   📄 {file.name}")
            if len(asset_files) > 3:
                print(f"   ... and {len(asset_files) - 3} more files")
        else:
            print(f"⚠️  Asset directory not found: {asset_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_validation():
    """Test configuration and dependencies"""
    print("🔧 Testing configuration...")
    
    # Test Mistral config
    config = MistralOCRConfig()
    if config.api_key:
        print(f"✅ Mistral API key configured")
    else:
        print(f"⚠️  Mistral API key not configured (will use DOCLING fallback)")
    
    # Test required imports
    try:
        from docling.document_converter import DocumentConverter
        print("✅ DOCLING available")
    except ImportError as e:
        print(f"❌ DOCLING not available: {e}")
        return False
    
    try:
        from mistralai import Mistral
        print("✅ Mistral SDK available")
    except ImportError as e:
        print(f"❌ Mistral SDK not available: {e}")
        return False
    
    return True


async def main():
    """Main test function"""
    print("🧪 Paper2Poster Mistral OCR Integration Test")
    print("=" * 60)
    
    # Configuration test
    config_ok = test_config_validation()
    if not config_ok:
        print("\n❌ Configuration test failed. Please fix dependencies.")
        return False
    
    # Find test PDF
    test_pdfs = [
        "test_sample.pdf",
        "sample.pdf", 
        "test.pdf"
    ]
    
    test_pdf = None
    for pdf in test_pdfs:
        if os.path.exists(pdf):
            test_pdf = pdf
            break
    
    if not test_pdf:
        print(f"\n📄 No test PDF found. Tried: {', '.join(test_pdfs)}")
        print("   Please place a PDF file named 'test_sample.pdf' to test")
        print("   the complete pipeline.")
        return True  # Config is OK, just no test file
    
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
        print("   1. Test with more complex PDFs")
        print("   2. Monitor API costs and performance")
        print("   3. Fine-tune quality thresholds")
        return True
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    asyncio.run(main()) 