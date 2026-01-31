# stage1_segmentation.py
"""
Stage 1: PDF Segmentation for Regulatory Documents

What it does:
1. Extracts title page as Section 1 with full document information
2. Enhanced TOC parsing - finds embedded TOC within document content
3. Creates meaningful sections using real regulatory topic names
4. Intelligent page filtering - skips intermediate pages between title and content
5. Multi-page table merging - maintains table consistency across page boundaries
6. Outputs clean, structured JSON optimized for Stage 2 LLM processing

Key Features:
- Title page becomes Section 1 (works with any document)
- Adaptive TOC parsing: finds TOC wherever it appears in the document
- Generic section extraction: uses whatever section names the document has
- Flexible page range detection: follows TOC page numbers automatically
- Batch processing: handles entire directories of mixed document types
- Multi-page table merging: automatically detects and merges split tables
- Fallback handling: creates basic structure if TOC parsing fails

Works with any regulatory PDF that has:
- A title page (becomes Section 1)
- Some form of Table of Contents (embedded or separate page)
- Structured sections with page numbers
- Examples: SASB, TCFD, IFRS, national standards, compliance frameworks

Usage:
  # Process single PDF:
  python stage1_segmentation.py path/to/document.pdf
  
  # Process all PDFs in input directory:
  python stage1_segmentation.py --batch
  python stage1_segmentation.py --batch custom/input/directory

Output Structure:
  outputs/stage1_segments/
  ├── {document_name}_segments.json     # Main segmented content for LLM
  ├── {document_name}_metadata.json    # Document metadata and structure
  └── {document_name}_summary.txt      # Human-readable processing summary

Segment JSON Format (generic structure for ANY regulatory PDF):

  {
    "segment_id": "seg_001",                  # Always starts with seg_001
    "section_title": "Title & Document Information", # Always Section 1 = title page
    "section_number": "1",                   # Always numbered "1"
    "page_start": 0,                         # Always page 0 (title page)
    "page_end": 0,
    "content": "[Whatever text is on the title page]", # Extracted title page content
    "has_tables": true/false,                # True if title page has tables
    "tables": [...]                          # Any tables found on title page
  },
  {
    "segment_id": "seg_002",                  # Sequential segment IDs
    "section_title": "[Whatever the TOC says]", # Extracted from document's TOC
    "section_number": "2",                   # Sequential numbering
    "page_start": X,                         # Determined from TOC page numbers
    "page_end": Y,
    "content": "[Extracted text from pages X-Y]", # All text from this section
    "has_tables": true/false,                # True if section contains tables
    "tables": [...]                          # Any tables found in this section
  }
  // ... more segments based on what the TOC contains

How it works:
1. Section 1 is ALWAYS the title page (regardless of document type)
2. Section 2+ are created from whatever sections the document's TOC contains
3. Section titles come directly from the TOC parsing (not hardcoded)
4. Content is extracted from the page ranges specified in the TOC
5. Tables are detected and merged automatically across pages
6. Works with any PDF that has a title page and some kind of TOC structure

This structured output is optimized for Stage 2 LLM processing and knowledge graph construction.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import pdfplumber
import re
import glob
import time

@dataclass
class DocumentSegment:
    """
    Represents one logical section of a regulatory document.
    
    Each segment corresponds to a section from the document's Table of Contents
    and contains all text and tables from that section's page range.
    This structure is optimized for LLM processing in Stage 2.
    """
    segment_id: str          # Unique identifier: "seg_001", "seg_002", etc.
    section_title: str       # Section name from TOC: "Climate-related Disclosures"
    section_number: str      # Section numbering: "1.0", "A.1", "Appendix B"
    page_start: int          # Starting page number in original PDF
    page_end: int            # Ending page number in original PDF
    content: str             # Clean extracted text content
    has_tables: bool         # Flag indicating presence of tables
    tables: List[Dict]       # Structured table data (merged across pages if needed)
    
    def to_dict(self):
        return asdict(self)


class RegulatoryPDFSegmenter:
    """
    Advanced segmenter for regulatory documents with enhanced TOC parsing.
    
    Features:
    - Title page integration as Section 1
    - Enhanced TOC detection (embedded and standalone)
    - Multi-page table merging
    - Real regulatory section names
    - Batch processing support
    """
    
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.pdf = None
        self.title = ""
        self.title_content = ""  # Full title page content
        self.toc = []  # Table of Contents
        self.segments = []
        
    def process(self) -> Dict:
        """Main process - does everything"""
        
        print(f"\n{'='*60}")
        print(f"Processing: {self.pdf_path.name}")
        print(f"{'='*60}\n")
        
        with pdfplumber.open(self.pdf_path) as pdf:
            self.pdf = pdf
            
            # Step 1: Extract title page
            print("Step 1: Extracting title page...")
            self.title = self._extract_title()
            print(f"  Title: {self.title}")
            
            # Step 2: Find Table of Contents
            print("\nStep 2: Finding Table of Contents...")
            toc_page = self._find_toc_page()
            print(f"  TOC found on page: {toc_page}")
            
            # Step 3: Parse TOC structure
            print("\nStep 3: Parsing TOC structure...")
            self.toc = self._parse_toc(toc_page)
            print(f"  Found {len(self.toc)} sections")
            
            # Step 4: Split document by TOC sections
            print("\nStep 4: Splitting document by sections...")
            self.segments = self._create_segments()
            print(f"  Created {len(self.segments)} segments")
            
            # Step 5: Save outputs
            print("\nStep 5: Saving outputs...")
            output = self._save_outputs()
            
        print(f"\n{'='*60}")
        print("✅ COMPLETE")
        print(f"{'='*60}\n")
        
        return output
    
    def _extract_title(self) -> str:
        """Extract title and full content from first page"""
        first_page = self.pdf.pages[0]
        text = first_page.extract_text()
        
        if not text:
            return "Untitled Document"
        
        # Store full title page content for segmentation
        self.title_content = self._clean_text(text)
        
        # Get first few meaningful lines for title
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        
        # Usually title is in first 3 lines
        title = ' '.join(lines[:3])
        
        # Clean up title
        title = re.sub(r'\s+', ' ', title)
        title = title[:200]  # Limit length
        
        return title
    
    def _find_toc_page(self) -> int:
        """Find which page has Table of Contents"""
        
        # Keywords that indicate TOC
        toc_keywords = [
            'contents', 'table of contents', 
            'introduction', 'overview'
        ]
        
        # Check first 10 pages
        for page_num in range(min(10, len(self.pdf.pages))):
            page = self.pdf.pages[page_num]
            text = page.extract_text().lower()
            
            # Look for TOC keywords
            if any(keyword in text for keyword in toc_keywords):
                # Check if it has section numbers (1., 2., etc.)
                if re.search(r'\d+\.?\s+[A-Z]', page.extract_text()):
                    return page_num
        
        # If not found, assume page 1 or 2
        return 1
    
    def _parse_toc(self, toc_page: int) -> List[Dict]:
        """
        Parse Table of Contents with enhanced detection capabilities.
        
        Features:
        - Traditional TOC page parsing
        - Embedded TOC detection within document content
        - Multiple pattern matching for various TOC formats
        - Fallback to basic structure if no TOC found
        
        Returns:
            List of TOC entries with section_number, title, and page
        """
        
        # Try to find TOC in multiple pages and within content
        toc_entries = []
        
        # First, try the traditional approach on the TOC page
        if toc_page < len(self.pdf.pages):
            page = self.pdf.pages[toc_page]
            text = page.extract_text()
            toc_entries = self._extract_toc_from_text(text)
        
        # If no TOC found, search in the first few pages for embedded TOC
        if not toc_entries:
            print("  Searching for embedded TOC in document content...")
            for page_num in range(min(5, len(self.pdf.pages))):
                page = self.pdf.pages[page_num]
                text = page.extract_text()
                
                # Look for TOC section within the text - try multiple variations
                text_lower = text.lower()
                toc_found = False
                for toc_var in ['table of contents', 'contents', 'table of content', 'content']:
                    if toc_var in text_lower:
                        toc_found = True
                        print(f"  Found '{toc_var}' in page {page_num}")
                        break
                
                if toc_found:
                    toc_entries = self._extract_embedded_toc(text)
                    if toc_entries:
                        print(f"  Successfully extracted embedded TOC from page {page_num}")
                        break
        
        # If still no TOC found, create basic structure
        if not toc_entries:
            print("  ⚠️  No TOC found, creating basic structure...")
            toc_entries = self._create_basic_structure()
        
        return toc_entries
    
    def _extract_toc_from_text(self, text: str) -> List[Dict]:
        """Extract TOC entries from text using various patterns"""
        toc_entries = []
        
        patterns = [
            # "1. Introduction ..... 5" or "1 Introduction ..... 5"
            r'(\d+\.?\d*)\s+([A-Z][^\.\n]{5,50})[\.\s]+(\d+)',
            
            # "Appendix A - Title ..... 45"
            r'(Appendix\s+[A-Z])\s*[\-\:]?\s*([^\.\n]{5,50})[\.\s]+(\d+)',
            
            # "A.1 Title ..... 10"
            r'([A-Z]\.\d+)\s+([A-Z][^\.\n]{5,50})[\.\s]+(\d+)',
        ]
        
        for line in text.split('\n'):
            line = line.strip()
            
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    section_num = match.group(1).strip()
                    section_title = match.group(2).strip()
                    page_num = int(match.group(3))
                    
                    toc_entries.append({
                        'section_number': section_num,
                        'title': section_title,
                        'page': page_num
                    })
                    break
        
        return toc_entries
    
    def _extract_embedded_toc(self, text: str) -> List[Dict]:
        """Extract TOC from embedded table of contents in document content"""
        toc_entries = []
        
        # Find the TOC section - try multiple variations
        text_lower = text.lower()
        toc_variations = [
            'table of contents',
            'contents',
            'table of content', 
            'content',
            'index'
        ]
        
        toc_start = -1
        for variation in toc_variations:
            toc_start = text_lower.find(variation)
            if toc_start != -1:
                print(f"  Found TOC variation: '{variation}' at position {toc_start}")
                break
        
        if toc_start == -1:
            return toc_entries
        
        # Get text after "Table of Contents"
        toc_text = text[toc_start:]
        
        # Enhanced patterns for embedded TOC (often without dots)
        patterns = [
            # "INTRODUCTION......4" or "Data Security .....8"
            r'([A-Z][A-Za-z\s&,\-]+?)[\.\s]*(\d+)$',
            
            # "Overview of SASB Standards.....4"
            r'([A-Z][A-Za-z\s&,\-()]+?)[\.\s]*(\d+)$',
            
            # Handle multi-word titles with various separators
            r'([A-Z][A-Za-z\s&,\-()]+?)\s*[\.\s]*(\d+)$',
            
            # "OBJECTIVE 1" or "SCOPE 3" (section name followed by number)
            r'^([A-Z][A-Z\s&,\-()]+?)\s+(\d+)$',
            
            # "from paragraph" or standalone section names (no page numbers)
            r'^([A-Z][A-Z\s&,\-()]{4,})$'
        ]
        
        lines = toc_text.split('\n')
        section_counter = 1
        
        for line in lines[:20]:  # Look at first 20 lines after TOC
            line = line.strip()
            
            # Skip empty lines and the TOC header lines
            if not line or len(line) < 5:
                continue
            
            # Skip TOC header variations
            line_lower = line.lower()
            if any(toc_var in line_lower for toc_var in ['table of contents', 'contents', 'table of content']):
                continue
            
            # Skip lines that are clearly not TOC entries
            if any(skip in line.lower() for skip in ['page', 'sustainability accounting standard', '©', 'ifrs']):
                continue
            
            for i, pattern in enumerate(patterns):
                match = re.search(pattern, line)
                if match and len(match.group(1).strip()) > 3:
                    section_title = match.group(1).strip()
                    
                    # Handle patterns with and without page numbers
                    if match.lastindex >= 2 and match.group(2).isdigit():
                        page_num = int(match.group(2))
                    else:
                        # For patterns without page numbers, estimate based on position
                        page_num = section_counter + 1
                    
                    # Clean up the title
                    section_title = re.sub(r'\s+', ' ', section_title)
                    
                    toc_entries.append({
                        'section_number': str(section_counter),
                        'title': section_title,
                        'page': page_num
                    })
                    section_counter += 1
                    break
        
        return toc_entries
    
    def _find_first_content_page(self) -> int:
        """Find the first page with actual content (after title and TOC)"""
        
        # Start after title page (page 0)
        for page_num in range(1, min(10, len(self.pdf.pages))):
            page = self.pdf.pages[page_num]
            text = page.extract_text()
            
            if not text:
                continue
                
            text_lower = text.lower()
            
            # Skip TOC pages
            if any(keyword in text_lower for keyword in ['contents', 'table of contents']):
                continue
                
            # Skip pages with mostly page numbers, headers, footers
            meaningful_lines = [line.strip() for line in text.split('\n') 
                              if line.strip() and len(line.strip()) > 10]
            
            if len(meaningful_lines) >= 5:  # Has substantial content
                return page_num
        
        # Default to page after TOC
        toc_page = self._find_toc_page()
        return toc_page + 1
    
    def _create_basic_structure(self) -> List[Dict]:
        """Create basic structure if no TOC found"""
        
        # Start from first content page, not page 0
        first_content = self._find_first_content_page()
        total_pages = len(self.pdf.pages)
        chunk_size = 10
        
        entries = []
        for i in range(first_content, total_pages, chunk_size):
            entries.append({
                'section_number': f'{(i-first_content)//chunk_size + 1}',
                'title': f'Section {(i-first_content)//chunk_size + 1}',
                'page': i
            })
        
        return entries
    
    def _create_segments(self) -> List[DocumentSegment]:
        """Create segments with title page as Section 1, followed by real TOC structure"""
        
        segments = []
        
        # Create title page segment as Section 1 (following your original vision)
        print(f"  Processing: Section 1 - Title & Document Info (page 0)")
        
        # Extract any tables from title page
        title_tables = []
        if len(self.pdf.pages) > 0:
            title_page = self.pdf.pages[0]
            tables = title_page.extract_tables()
            for table_idx, table in enumerate(tables):
                if table and len(table) > 1:
                    table_dict = {
                        'page': 0,
                        'table_id': f'table_p0_t{table_idx}',
                        'headers': table[0],
                        'rows': table[1:]
                    }
                    title_tables.append(table_dict)
        
        # Create title segment as Section 1
        title_segment = DocumentSegment(
            segment_id="seg_001",
            section_title="Title & Document Information",
            section_number="1",
            page_start=0,
            page_end=0,
            content=self.title_content,
            has_tables=len(title_tables) > 0,
            tables=title_tables
        )
        segments.append(title_segment)
        
        # Find first actual content page (skip title and intermediate pages)
        first_content_page = self._find_first_content_page()
        print(f"  First content page: {first_content_page}")
        
        # Process TOC-based segments starting from Section 2
        section_counter = 2  # Start from 2 since title is Section 1
        
        for i, toc_entry in enumerate(self.toc):
            
            # Determine page range
            page_start = max(toc_entry['page'], first_content_page)
            
            # Page end is start of next section (or last page)
            if i < len(self.toc) - 1:
                page_end = self.toc[i + 1]['page'] - 1
            else:
                page_end = len(self.pdf.pages) - 1
            
            # Skip if page range is invalid
            if page_start > page_end:
                print(f"  Skipping: {toc_entry['title']} (invalid page range)")
                continue
                
            print(f"  Processing: Section {section_counter} - {toc_entry['title']} (pages {page_start}-{page_end})")
            
            # Extract content from these pages
            content, tables = self._extract_content(page_start, page_end)
            
            # Skip segments with minimal content
            if len(content.strip()) < 100:
                print(f"    Skipping segment with minimal content")
                continue
            
            # Create segment with sequential numbering
            segment = DocumentSegment(
                segment_id=f"seg_{section_counter:03d}",
                section_title=toc_entry['title'],
                section_number=str(section_counter),  # Sequential: 2, 3, 4, 5...
                page_start=page_start,
                page_end=page_end,
                content=content,
                has_tables=len(tables) > 0,
                tables=tables
            )
            
            segments.append(segment)
            section_counter += 1  # Increment only for successfully created segments
        
        return segments
    
    def _extract_content(self, page_start: int, page_end: int) -> tuple:
        """Extract text and tables from page range"""
        
        all_text = []
        all_tables = []
        page_tables = {}  # Track tables by page for merging
        
        for page_num in range(page_start, page_end + 1):
            if page_num >= len(self.pdf.pages):
                break
                
            page = self.pdf.pages[page_num]
            
            # Extract text
            text = page.extract_text()
            if text and len(text.strip()) > 50:  # Skip nearly empty pages
                all_text.append(text)
            
            # Extract tables
            tables = page.extract_tables()
            page_tables[page_num] = []
            
            for table_idx, table in enumerate(tables):
                if table and len(table) > 1:  # Has headers + data
                    
                    # Convert table to readable format
                    table_dict = {
                        'page': page_num,
                        'table_id': f'table_p{page_num}_t{table_idx}',
                        'headers': table[0],
                        'rows': table[1:],
                        'raw_table': table
                    }
                    
                    page_tables[page_num].append(table_dict)
        
        # Merge multi-page tables
        merged_tables = self._merge_multipage_tables(page_tables)
        all_tables.extend(merged_tables)
        
        # Combine all text
        combined_text = '\n\n'.join(all_text)
        
        # Clean text
        combined_text = self._clean_text(combined_text)
        
        return combined_text, all_tables
    
    def _merge_multipage_tables(self, page_tables: Dict) -> List[Dict]:
        """Merge tables that span multiple pages"""
        
        merged_tables = []
        processed_pages = set()
        
        for page_num, tables in page_tables.items():
            if page_num in processed_pages:
                continue
                
            for table in tables:
                # Check if this table continues on next pages
                merged_table = self._try_merge_table(table, page_tables, page_num)
                
                # Mark pages as processed
                for p in range(page_num, merged_table.get('last_page', page_num) + 1):
                    processed_pages.add(p)
                
                merged_tables.append(merged_table)
        
        return merged_tables
    
    def _try_merge_table(self, base_table: Dict, page_tables: Dict, start_page: int) -> Dict:
        """Try to merge a table with continuation on next pages"""
        
        merged_table = base_table.copy()
        merged_table['pages'] = [start_page]
        merged_table['last_page'] = start_page
        
        # Look for continuation on next pages
        current_page = start_page + 1
        base_headers = base_table['headers']
        
        while current_page in page_tables:
            page_tables_list = page_tables[current_page]
            
            # Look for table with similar headers
            continuation_found = False
            for next_table in page_tables_list:
                if self._tables_match(base_headers, next_table['headers']):
                    # Merge the rows
                    merged_table['rows'].extend(next_table['rows'])
                    merged_table['pages'].append(current_page)
                    merged_table['last_page'] = current_page
                    merged_table['table_id'] = f"table_p{start_page}-{current_page}_merged"
                    continuation_found = True
                    break
            
            if not continuation_found:
                break
                
            current_page += 1
        
        # Clean up the merged table
        if 'raw_table' in merged_table:
            del merged_table['raw_table']
            
        return merged_table
    
    def _tables_match(self, headers1: List, headers2: List) -> bool:
        """Check if two tables have matching headers (likely same table)"""
        
        if not headers1 or not headers2:
            return False
            
        # Clean headers for comparison
        clean_h1 = [str(h).strip().lower() if h else '' for h in headers1]
        clean_h2 = [str(h).strip().lower() if h else '' for h in headers2]
        
        # Check if headers are similar (at least 70% match)
        if len(clean_h1) != len(clean_h2):
            return False
            
        matches = sum(1 for h1, h2 in zip(clean_h1, clean_h2) if h1 == h2)
        similarity = matches / len(clean_h1) if clean_h1 else 0
        
        return similarity >= 0.7
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        
        # Remove excessive whitespace
        text = re.sub(r'\n\n+', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove page numbers (usually at bottom)
        text = re.sub(r'\n\d+\n', '\n', text)
        text = re.sub(r'Page \d+', '', text)
        
        return text.strip()
    
    def _save_outputs(self) -> Dict:
        """Save all outputs in structured format for Stage 2 processing"""
        
        output_dir = Path('outputs/stage1_segments')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        doc_name = self.pdf_path.stem
        
        # 1. Save segments - Main content for LLM processing
        # Contains full text and table data for each document section
        segments_file = output_dir / f'{doc_name}_segments.json'
        with open(segments_file, 'w', encoding='utf-8') as f:
            json.dump(
                [seg.to_dict() for seg in self.segments],
                f,
                indent=2,
                ensure_ascii=False
            )
        print(f"  ✓ Segments: {segments_file}")
        
        # 2. Save metadata - Document structure and overview
        # Used for context and navigation in knowledge graph construction
        metadata = {
            'document_name': doc_name,
            'title': self.title,
            'total_pages': len(self.pdf.pages),
            'total_segments': len(self.segments),
            'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'sections': [
                {
                    'number': seg.section_number,
                    'title': seg.section_title,
                    'pages': f'{seg.page_start}-{seg.page_end}',
                    'word_count': len(seg.content.split()),
                    'table_count': len(seg.tables)
                }
                for seg in self.segments
            ]
        }
        
        metadata_file = output_dir / f'{doc_name}_metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Metadata: {metadata_file}")
        
        # 3. Save human-readable summary - For quick review and debugging
        # Provides overview of processing results and segment statistics
        summary_file = output_dir / f'{doc_name}_summary.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Document: {self.title}\n")
            f.write(f"Total Pages: {len(self.pdf.pages)}\n")
            f.write(f"Total Segments: {len(self.segments)}\n")
            f.write(f"Processed: {metadata['processing_timestamp']}\n\n")
            f.write("="*60 + "\n\n")
            
            for seg in self.segments:
                f.write(f"Section {seg.section_number}: {seg.section_title}\n")
                f.write(f"  Pages: {seg.page_start}-{seg.page_end}\n")
                f.write(f"  Words: {len(seg.content.split())}\n")
                f.write(f"  Tables: {len(seg.tables)}\n")
                f.write("\n")
        print(f"  ✓ Summary: {summary_file}")
        
        return {
            'segments': self.segments,
            'metadata': metadata,
            'output_dir': str(output_dir),
            'success': True,
            'error': None
        }


def process_single_pdf(pdf_path: str) -> Dict:
    """Process a single PDF file with error handling"""
    try:
        segmenter = RegulatoryPDFSegmenter(pdf_path)
        result = segmenter.process()
        return result
    except Exception as e:
        print(f"\n❌ Error processing {Path(pdf_path).name}: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'pdf_path': pdf_path
        }


def process_batch_pdfs(input_dir: str) -> Dict:
    """Process all PDF files in the input directory"""
    
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"\n❌ Error: Input directory not found: {input_dir}")
        return {'success': False, 'error': 'Directory not found'}
    
    # Find all PDF files
    pdf_files = list(input_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"\n❌ No PDF files found in: {input_dir}")
        return {'success': False, 'error': 'No PDF files found'}
    
    print(f"\n🔍 Found {len(pdf_files)} PDF files to process")
    print(f"📁 Input directory: {input_dir}")
    print(f"{'='*80}")
    
    results = []
    successful = 0
    failed = 0
    start_time = time.time()
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_file.name}")
        print("-" * 60)
        
        result = process_single_pdf(str(pdf_file))
        results.append(result)
        
        if result.get('success', False):
            successful += 1
            print(f"✅ Success: {pdf_file.name}")
        else:
            failed += 1
            print(f"❌ Failed: {pdf_file.name}")
    
    # Summary
    elapsed_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"📊 BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"📁 Total files: {len(pdf_files)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total time: {elapsed_time:.1f} seconds")
    print(f"📁 Outputs saved to: outputs/stage1_segments/")
    
    if failed > 0:
        print(f"\n⚠️  Failed files:")
        for result in results:
            if not result.get('success', False):
                pdf_name = Path(result.get('pdf_path', 'unknown')).name
                error = result.get('error', 'Unknown error')
                print(f"   - {pdf_name}: {error}")
    
    return {
        'success': True,
        'total_files': len(pdf_files),
        'successful': successful,
        'failed': failed,
        'results': results,
        'elapsed_time': elapsed_time
    }


# Command-line interface
if __name__ == '__main__':
    import sys
    
    # Check arguments
    if len(sys.argv) < 2:
        print("\n❌ Error: Please provide PDF file path or --batch flag")
        print("\nUsage:")
        print("  # Process single PDF:")
        print("  python stage1_segmentation.py path/to/document.pdf")
        print("\n  # Process all PDFs in input directory:")
        print("  python stage1_segmentation.py --batch")
        print("  python stage1_segmentation.py --batch data/input_pdfs")
        print("\nExamples:")
        print("  python stage1_segmentation.py data/input_pdfs/sasb-banks.pdf")
        print("  python stage1_segmentation.py --batch")
        print()
        sys.exit(1)
    
    arg = sys.argv[1]
    
    # Batch processing mode
    if arg == '--batch':
        # Use default input directory or specified directory
        if len(sys.argv) > 2:
            input_dir = sys.argv[2]
        else:
            input_dir = "data/input_pdfs"
        
        print(f"\n🚀 Starting batch processing...")
        result = process_batch_pdfs(input_dir)
        
        if not result.get('success', False):
            sys.exit(1)
    
    # Single file processing mode
    else:
        pdf_path = arg
        
        # Check file exists
        if not Path(pdf_path).exists():
            print(f"\n❌ Error: File not found: {pdf_path}\n")
            sys.exit(1)
        
        # Process single file
        print(f"\n🚀 Processing single PDF: {Path(pdf_path).name}")
        result = process_single_pdf(pdf_path)
        
        if result.get('success', False):
            print(f"\n📁 Output saved to: {result['output_dir']}\n")
        else:
            print(f"\n❌ Processing failed: {result.get('error', 'Unknown error')}\n")
            sys.exit(1)


