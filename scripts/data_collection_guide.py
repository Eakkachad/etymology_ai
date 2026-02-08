#!/usr/bin/env python3
"""
Data Collection Guide for Thai Etymology

This script shows how to collect Thai word etymology data from various sources.
Focus: Thai words -> Etymology (Pali/Sanskrit origins)
"""

import requests
import json
from pathlib import Path

def collect_from_kaikki():
    """
    Method 1: Use Kaikki.org pre-parsed Wiktionary data
    
    คำแนะนำ:
    1. ดาวน์โหลด: wget https://kaikki.org/dictionary/Thai/kaikki.org-dictionary-Thai.json
    2. แยกคำที่มี etymology
    3. Filter เฉพาะที่มาจาก บาลี/สันสกฤต
    """
    
    print("=" * 70)
    print("วิธีที่ 1: Kaikki.org (Recommended)")
    print("=" * 70)
    
    print("""
    ขั้นตอน:
    1. ดาวน์โหลดข้อมูล:
       wget https://kaikki.org/dictionary/Thai/kaikki.org-dictionary-Thai.json
    
    2. Filter คำที่มี etymology:
       - มี key "etymology_text" หรือ "etymology_templates"
       - มีคำว่า "Sanskrit", "Pali", "บาลี", "สันสกฤต"
    
    3. Extract ข้อมูล:
       - คำไทย (word)
       - คำอ่าน (IPA/romanization)
       - คำต้นทาง (บาลี/สันสกฤต)
       - ความหมาย
    
    ตัวอย่าง entry:
    {
      "word": "ไตร",
      "pos": "num",
      "senses": [{"glosses": ["three"]}],
      "etymology_text": "จากบาลีและสันสกฤต tri",
      "sounds": [{"ipa": "/traj/"}]
    }
    """)

def collect_from_wiktionary():
    """
    Method 2: Scrape Thai Wiktionary directly
    """
    
    print("\n" + "=" * 70)
    print("วิธีที่ 2: Thai Wiktionary (Scraping)")
    print("=" * 70)
    
    print("""
    ขั้นตอน:
    1. เลือกคำไทยที่ต้องการ (เช่น จากรายการคำยืมภาษาบาลี/สันสกฤต)
    
    2. Scrape แต่ละหน้า:
       URL: https://th.wiktionary.org/wiki/{คำไทย}
    
    3. Extract ส่วน "รากศัพท์" (Etymology section)
    
    ตัวอย่าง Code:
    """)
    
    example_code = '''
import requests
from bs4 import BeautifulSoup

def get_thai_etymology(word):
    url = f"https://th.wiktionary.org/wiki/{word}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # หา section "รากศัพท์"
    etymology_section = soup.find('span', {'id': 'รากศัพท์'})
    if etymology_section:
        parent = etymology_section.parent
        # ดึงข้อความใน section นี้
        etymology_text = parent.find_next_sibling('p').text
        return etymology_text
    return None

# ทดสอบ
word = "ไตร"
etym = get_thai_etymology(word)
print(f"{word}: {etym}")
    '''
    print(example_code)

def manual_curation_guide():
    """
    Method 3: Manual curation from reliable sources
    """
    
    print("\n" + "=" * 70)
    print("วิธีที่ 3: รวบรวมข้อมูลด้วยมือ (Manual Curation)")
    print("=" * 70)
    
    print("""
    แหล่งข้อมูลน่าเชื่อถือ:
    
    1. พจนานุกรมฉบับราชบัณฑิตยสถาน
       https://dictionary.orst.go.th/
       - มีข้อมูล etymology บางคำ
       - ต้องค้นหาแต่ละคำ
    
    2. หนังสือ "คำยืมภาษาบาลีและสันสกฤตในภาษาไทย"
       - รวบรวมโดยนักภาษาศาสตร์
       - มีความแม่นยำสูง
    
    3. เว็บไซต์ thai-language.com
       http://www.thai-language.com/
       - มีข้อมูล etymology บางส่วน
    
    รูปแบบข้อมูลที่ควรเก็บ:
    {
      "thai_word": "ไตร",
      "thai_pronunciation": "ไตร",
      "ipa": "/traj/",
      "meaning": "สาม",
      "etymology": {
        "source_language": "Pali/Sanskrit",
        "source_word": "tri",
        "source_ipa": "/tri/",
        "source_meaning": "three",
        "pie_root": "*tréyes"
      }
    }
    """)

def example_dataset_structure():
    """
    Show recommended dataset structure
    """
    
    print("\n" + "=" * 70)
    print("โครงสร้างข้อมูลที่แนะนำ")
    print("=" * 70)
    
    sample_data = [
        {
            "thai_word": "ไตร",
            "thai_ipa": "traj",
            "meaning_th": "สาม",
            "meaning_en": "three",
            "etymology": {
                "source_lang": "Pali/Sanskrit",
                "source_word": "tri",
                "source_ipa": "tri",
                "pie_root": "*tréyes"
            },
            "cognates": {
                "english": "three",
                "latin": "tres",
                "greek": "treis"
            }
        },
        {
            "thai_word": "มาตร",
            "thai_ipa": "maːt",
            "meaning_th": "แม่",
            "meaning_en": "mother",
            "etymology": {
                "source_lang": "Sanskrit",
                "source_word": "mātr̥",
                "source_ipa": "maːtr̩",
                "pie_root": "*méh₂tēr"
            },
            "cognates": {
                "english": "mother",
                "latin": "mater",
                "greek": "meter"
            }
        }
    ]
    
    output_path = Path(__file__).parent.parent / "data/thai_etymology_template.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ สร้างไฟล์ตัวอย่าง: {output_path}")
    print("\nโครงสร้างข้อมูล:")
    print(json.dumps(sample_data[0], ensure_ascii=False, indent=2))

def main():
    print("\n" + "=" * 70)
    print("คู่มือการรวบรวมข้อมูล Etymology คำไทย")
    print("=" * 70)
    
    collect_from_kaikki()
    collect_from_wiktionary()
    manual_curation_guide()
    example_dataset_structure()
    
    print("\n" + "=" * 70)
    print("สรุปคำแนะนำ")
    print("=" * 70)
    print("""
    📌 แนะนำสำหรับเริ่มต้น:
    
    1. ใช้ Kaikki.org (ง่ายที่สุด)
       - ดาวน์โหลด JSON file
       - Filter คำที่มี etymology
       - สร้าง dataset ขนาด 1,000-10,000 คำ
    
    2. เสริมด้วย Manual Curation
       - เลือกคำสำคัญ 100-200 คำ
       - ตรวจสอบความถูกต้องจาก Royal Institute Dictionary
       - เพิ่ม cognates และ PIE roots
    
    3. Validate ข้อมูล
       - ตรวจสอบ IPA pronunciation
       - ยืนยัน etymology จากหลายแหล่ง
       - ทดสอบกับ demo script
    
    เป้าหมาย: 1,000+ Thai-Sanskrit/Pali pairs สำหรับ training
    """)

if __name__ == "__main__":
    main()
