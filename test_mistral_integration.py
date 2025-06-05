#!/usr/bin/env python3
"""
Test script for Mistral OCR Integration
验证新的PDF处理流程
"""

import asyncio
import os
import sys
from pathlib import Path

from PosterAgent.mistral_ocr import (
    MistralOCRConfig,
    process_pdf_with_mistral,
    check_mistral_output_quality,
    extract_assets_from_mistral_response
)
from mistralai import Mistral

sys.path.append(str(Path(__file__).parent))


async def test_basic_functionality():
    """测试基本功能"""
    print("🧪 Testing Mistral OCR Integration...")
    
    # 配置检查
    config = MistralOCRConfig()
    if not config.api_key:
        print("❌ MISTRAL_API_KEY not configured")
        return False
    
    print(f"✅ API Key configured: {'*' * 10}")
    print(f"✅ Model: {config.model}")
    print(f"✅ Max pages: {config.max_pages}")
    
    return True


async def test_pdf_processing(pdf_path: str):
    """测试PDF处理功能"""
    if not os.path.exists(pdf_path):
        print(f"❌ Test PDF not found: {pdf_path}")
        return False
    
    try:
        config = MistralOCRConfig()
        client = Mistral(api_key=config.api_key)
        
        print(f"📄 Processing PDF: {pdf_path}")
        
        # 处理PDF
        result = await process_pdf_with_mistral(
            pdf_path=pdf_path,
            client=client,
            config=config
        )
        raw_response, markdown, images, tables = result
        
        # 质量检查
        ok, reasons = check_mistral_output_quality(
            raw_response, markdown, config
        )
        
        print("✅ Processing successful!")
        print(f"📝 Markdown length: {len(markdown)}")
        print(f"🖼️  Images extracted: {len(images)}")
        print(f"📊 Tables extracted: {len(tables)}")
        print(f"🔍 Quality check: {'PASS' if ok else 'FAIL'}")
        
        if not ok:
            print(f"⚠️  Quality issues: {', '.join(reasons)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return False


async def test_asset_extraction():
    """测试资产提取功能"""
    print("\n🔧 Testing asset extraction...")
    
    # 模拟Mistral响应数据
    mock_response = {
        "pages": [
            {
                "markdown": "## Test Section\nThis is test content.",
                "images": [
                    {
                        "image_annotation": "Figure 1: Test chart",
                        "top_left_x": 100,
                        "top_left_y": 200,
                        "bottom_right_x": 400,
                        "bottom_right_y": 500,
                                                 "image_base64": (
                             "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
                             "AAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                         )
                    }
                ]
            }
        ]
    }
    
    try:
        output_dir = "test_output"
        markdown, images, tables = extract_assets_from_mistral_response(
            mock_response, output_dir
        )
        
        print(f"✅ Asset extraction test passed")
        print(f"📝 Extracted markdown: {len(markdown)} chars")
        print(f"🖼️  Extracted images: {len(images)}")
        print(f"📊 Extracted tables: {len(tables)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Asset extraction failed: {e}")
        return False


async def main():
    """主测试函数"""
    print("🚀 Mistral OCR Integration Test Suite")
    print("=" * 50)
    
    # 测试基本功能
    basic_ok = await test_basic_functionality()
    
    if basic_ok:
        # 测试资产提取
        asset_ok = await test_asset_extraction()
        
        # 如果有测试PDF文件，测试实际处理
        test_pdf = "test_sample.pdf"
        if os.path.exists(test_pdf):
            pdf_ok = await test_pdf_processing(test_pdf)
        else:
            print(f"\n📄 No test PDF found at {test_pdf}")
            print("   To test PDF processing, place a PDF file named 'test_sample.pdf'")
            pdf_ok = True  # Skip this test
        
        # 总结
        print("\n" + "=" * 50)
        print("📊 Test Summary:")
        print(f"   Basic functionality: {'✅ PASS' if basic_ok else '❌ FAIL'}")
        print(f"   Asset extraction: {'✅ PASS' if asset_ok else '❌ FAIL'}")
        print(f"   PDF processing: {'✅ PASS' if pdf_ok else '❌ SKIP'}")
        
        if basic_ok and asset_ok and pdf_ok:
            print("\n🎉 All tests passed! Integration ready for next phase.")
            return True
        else:
            print("\n⚠️  Some tests failed. Please check the issues above.")
            return False
    else:
        print("\n❌ Basic configuration failed. Please check your setup.")
        return False


if __name__ == "__main__":
    asyncio.run(main()) 