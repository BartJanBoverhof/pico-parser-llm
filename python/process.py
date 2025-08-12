import os
import re
import json
import statistics
import pdfplumber
from collections import defaultdict
import numpy as np
from typing import Dict, Any, Optional, List


class TableDetector:
    """
    Enhanced table detection that works across languages and document types
    without hardcoded language-specific patterns
    """
    
    def __init__(self, pdf_processor):
        """Initialize the detector with a PDFProcessor instance."""
        self.pdf_processor = pdf_processor
        
        # Universal patterns that indicate structured tabular data
        self.strong_table_patterns = [
            # Multiple aligned numeric values
            r'^\s*\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+',
            r'^\s*\d+,\d+\s+\d+,\d+\s+\d+,\d+',  # European decimal format
            
            # Multiple percentages or statistical measures
            r'^\s*\d+%\s+\d+%\s+\d+%',
            r'^\s*[<>=≤≥]\s*\d+[\.,]\d+\s+[<>=≤≥]\s*\d+[\.,]\d+',
            
            # Multiple fractions or ratios
            r'^\s*\d+/\d+\s+\d+/\d+\s+\d+/\d+',
            r'^\s*\d+:\d+\s+\d+:\d+\s+\d+:\d+',
            
            # Clear tabular separators
            r'\|\s*[^|]+\s*\|\s*[^|]+\s*\|',  # Pipe separators
            r'^\s*[^\t]+\t[^\t]+\t[^\t]+',     # Tab separators
            
            # Generated table content (from PDF extraction)
            r'Row\s+\d+.*:.*Row\s+\d+.*:',
            r'Column\s+\d+.*:.*Column\s+\d+.*:',
        ]
        
        # Pharmaceutical domain patterns
        self.pharmaceutical_patterns = [
            # Currency patterns (Euro symbols)
            r'€\s*\d+[\.,]\d+',
            r'\d+[\.,]\d+\s*€',
            
            # Medical dosage patterns
            r'\d+\s*mg(?:/m²)?(?:\s|$)',
            r'\d+\s*μg(?:\s|$)',
            r'\d+\s*ng(?:\s|$)',
            r'\d+\s*pg(?:\s|$)',
            r'\d+\s*IU(?:\s|$)',
            
            # Pharmaceutical abbreviations
            r'\b(?:FCT|CIS|TAB|AMP|SC|PIS)\b',
            r'\b(?:mg|ml|kg|mmol|μg|ng|pg|IU)\b',
            
            # Treatment cycle patterns
            r'\d+\s*x\s*(?:daily|per\s+day)',
            r'per\s+\d+-day\s+cycle',
            r'every\s+\d+\s+days',
            r'continuously',
            
            # Treatment designations
            r'designation\s+of\s+(?:the\s+)?therapy',
            r'treatment\s+mode',
            r'treatment\s+costs',
            r'appropriate\s+comparator\s+therapy',
        ]
        
        # Table title patterns
        self.table_title_patterns = [
            r'^\s*Table\s+\d+',
            r'^\s*Designation\s+of\s+the\s+therapy',
            r'^\s*Treatment\s+(?:costs|mode|schedule)',
            r'^\s*Medicinal\s+product\s+to\s+be\s+assessed',
            r'^\s*Appropriate\s+comparator\s+therapy',
            r'^\s*Consumption:?',
            r'^\s*Costs:?',
        ]
        
        # Patterns that strongly suggest prose (universal across languages)
        self.prose_patterns = [
            # Sentence-like structures with conjunctions/connectors
            r'\b\w{2,}\s+(?:and|or|but|however|therefore|moreover|furthermore|nevertheless|additionally)\s+\w{2,}',
            r'\b\w{2,}\s+(?:et|ou|mais|cependant|donc|de plus|néanmoins|également)\s+\w{2,}',  # French
            r'\b\w{2,}\s+(?:und|oder|aber|jedoch|daher|außerdem|dennoch|zusätzlich)\s+\w{2,}',  # German
            r'\b\w{2,}\s+(?:y|o|pero|sin embargo|por lo tanto|además|no obstante|también)\s+\w{2,}',  # Spanish
            r'\b\w{2,}\s+(?:e|o|ma|tuttavia|pertanto|inoltre|tuttavia|anche)\s+\w{2,}',  # Italian
            r'\b\w{2,}\s+(?:i|lub|ale|jednak|dlatego|ponadto|niemniej|również)\s+\w{2,}',  # Polish
            
            # Long sentences with punctuation
            r'[.!?]\s+[A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞŸ][a-zàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ]{2,}',
            
            # Phrases with articles and prepositions (common in prose)
            r'\b(?:the|a|an|in|on|at|by|for|with|from|to|of|as)\s+\w{2,}\s+\w{2,}',  # English
            r'\b(?:le|la|les|un|une|des|dans|sur|avec|pour|par|de|du|des)\s+\w{2,}\s+\w{2,}',  # French
            r'\b(?:der|die|das|ein|eine|in|auf|mit|für|durch|von|zu|bei)\s+\w{2,}\s+\w{2,}',  # German
            r'\b(?:el|la|los|las|un|una|en|con|por|para|de|del|desde)\s+\w{2,}\s+\w{2,}',  # Spanish
            r'\b(?:il|la|lo|gli|le|un|una|in|con|per|da|di|del|sulla)\s+\w{2,}\s+\w{2,}',  # Italian
        ]


    def enhanced_table_validation(self, table_data, page, page_num, sensitivity_level=1) -> bool:
        """
        Enhanced table validation with graduated sensitivity levels.
        Level 1: Standard validation (threshold 0.6)
        Level 2: Relaxed validation (threshold 0.5) 
        Level 3: Medical-specific validation (threshold 0.4 with domain patterns)
        """
        if not table_data or len(table_data) < 3:
            return False
        
        cleaned_table = self.pdf_processor.clean_table_data(table_data)
        if len(cleaned_table) < 3:
            return False
        
        # Core validation checks
        structure_score = self._check_table_structure(cleaned_table)
        content_score = self._check_table_content(cleaned_table)
        visual_score = 1.0 if self.has_explicit_table_structure(page) else 0.0
        
        # Apply sensitivity-based adjustments
        if sensitivity_level == 3:
            # Medical domain boost for level 3
            medical_score = self._check_medical_domain_patterns(cleaned_table)
            overall_score = (structure_score * 0.3) + (content_score * 0.3) + (visual_score * 0.1) + (medical_score * 0.3)
            threshold = 0.4
        elif sensitivity_level == 2:
            # Relaxed threshold for level 2
            overall_score = (structure_score * 0.4) + (content_score * 0.4) + (visual_score * 0.2)
            threshold = 0.5
        else:
            # Standard validation for level 1
            overall_score = (structure_score * 0.4) + (content_score * 0.4) + (visual_score * 0.2)
            threshold = 0.6
        
        return overall_score > threshold

    def _check_medical_domain_patterns(self, cleaned_table) -> float:
        """Check for medical domain-specific table patterns focusing on dosages and pricing."""
        if not cleaned_table:
            return 0.0
        
        score = 0.0
        all_cells = []
        
        # Collect all cell content
        for row in cleaned_table:
            for cell in row:
                if cell and cell.strip():
                    all_cells.append(cell.strip().lower())
        
        if not all_cells:
            return 0.0
        
        cell_text = ' '.join(all_cells)
        
        # Dosage patterns (high priority)
        dosage_patterns = [
            r'\d+\s*mg(?:/m²)?(?:\s|$)',
            r'\d+\s*μg(?:\s|$)',
            r'\d+\s*ml(?:\s|$)',
            r'\d+\s*(?:mg|μg|ml|g)\s*(?:daily|per\s+day|twice\s+daily)',
            r'cycle\s+\d+',
            r'\d+\s*x\s*daily',
            r'every\s+\d+\s+(?:days|weeks)',
        ]
        
        # Pricing patterns (high priority)
        pricing_patterns = [
            r'€\s*\d+(?:[.,]\d+)?',
            r'\d+(?:[.,]\d+)?\s*€',
            r'\$\s*\d+(?:[.,]\d+)?',
            r'cost(?:s)?',
            r'price(?:s)?',
            r'treatment\s+cost',
        ]
        
        # Medical table indicators (medium priority)
        medical_indicators = [
            r'designation\s+of\s+therapy',
            r'medicinal\s+product',
            r'appropriate\s+comparator',
            r'consumption',
            r'treatment\s+(?:mode|schedule)',
            r'therapeutic\s+indication',
        ]
        
        # Count pattern matches
        dosage_matches = sum(1 for pattern in dosage_patterns if re.search(pattern, cell_text, re.IGNORECASE))
        pricing_matches = sum(1 for pattern in pricing_patterns if re.search(pattern, cell_text, re.IGNORECASE))
        medical_matches = sum(1 for pattern in medical_indicators if re.search(pattern, cell_text, re.IGNORECASE))
        
        # Weight scoring based on priority
        if dosage_matches > 0:
            score += 0.4
        if pricing_matches > 0:
            score += 0.4
        if medical_matches > 0:
            score += 0.2
        
        # Bonus for combination of patterns
        if dosage_matches > 0 and pricing_matches > 0:
            score += 0.2
        
        return min(score, 1.0)

    def _check_table_structure(self, cleaned_table) -> float:
        """Check basic structural characteristics of potential table."""
        if not cleaned_table:
            return 0.0
        
        # 1. Reasonable dimensions
        rows = len(cleaned_table)
        cols = len(cleaned_table[0]) if cleaned_table[0] else 0
        
        if not (3 <= rows <= 50 and 2 <= cols <= 10):
            return 0.0
        
        # 2. Column consistency (most rows should have similar column count)
        col_counts = [len(row) for row in cleaned_table]
        most_common_cols = max(set(col_counts), key=col_counts.count)
        consistency = col_counts.count(most_common_cols) / len(col_counts)
        
        if consistency < 0.7:
            return 0.0
        
        # 3. Content density (not too sparse, not too dense)
        total_cells = sum(len(row) for row in cleaned_table)
        filled_cells = sum(1 for row in cleaned_table for cell in row if cell and cell.strip())
        density = filled_cells / max(total_cells, 1)
        
        if not (0.3 <= density <= 0.95):
            return 0.0
        
        return 1.0

    def _check_table_content(self, cleaned_table) -> float:
        """Check if content looks like typical table data."""
        if not cleaned_table:
            return 0.0
        
        score = 0.0
        total_cells = 0
        
        # Collect all non-empty cells
        all_cells = []
        for row in cleaned_table:
            for cell in row:
                if cell and cell.strip():
                    all_cells.append(cell.strip())
                    total_cells += 1
        
        if total_cells < 6:  # Too few cells to assess
            return 0.0
        
        # 1. Check for numeric content (common in tables)
        numeric_cells = sum(1 for cell in all_cells if self._is_numeric_like(cell))
        numeric_ratio = numeric_cells / total_cells
        
        if numeric_ratio > 0.3:  # Good amount of numeric data
            score += 0.4
        elif numeric_ratio > 0.1:  # Some numeric data
            score += 0.2
        
        # 2. Check cell length consistency (tables have concise cells)
        short_cells = sum(1 for cell in all_cells if len(cell.split()) <= 5)
        short_ratio = short_cells / total_cells
        
        if short_ratio > 0.7:  # Most cells are short
            score += 0.3
        elif short_ratio > 0.5:  # Many cells are short
            score += 0.15
        
        # 3. Check for table-like patterns
        table_patterns = 0
        text_sample = ' '.join(all_cells[:20])  # Sample for pattern checking
        
        # Currency, percentages, measurements
        if re.search(r'[€$£¥]\s*\d+|(\d+[.,]\d*\s*[%€$£¥])', text_sample):
            table_patterns += 1
        
        # Statistical patterns
        if re.search(r'[<>=≤≥]\s*\d+|p\s*[<>=]\s*\d', text_sample):
            table_patterns += 1
        
        # Medical/scientific units
        if re.search(r'\d+\s*(mg|ml|kg|%|mm|cm|years?|days?)\b', text_sample, re.IGNORECASE):
            table_patterns += 1
        
        if table_patterns > 0:
            score += 0.3
        
        return min(score, 1.0)


    def _group_words_into_lines(self, words, y_tolerance=3):
        """Group extracted words into lines based on their vertical position."""
        if not words:
            return []

        sorted_words = sorted(words, key=lambda w: w.get("top", 0))

        lines = []
        current_line = []
        current_top = None

        for word in sorted_words:
            top = word.get("top", 0)
            if current_top is None or abs(top - current_top) <= y_tolerance:
                current_line.append(word)
                if current_top is None:
                    current_top = top
            else:
                lines.append(sorted(current_line, key=lambda w: w.get("x0", 0)))
                current_line = [word]
                current_top = top

        if current_line:
            lines.append(sorted(current_line, key=lambda w: w.get("x0", 0)))

        return lines


    def find_table_title(self, page, table_region=None):
        """Locate a nearby heading that likely serves as the table title."""
        try:
            # Extract words with positioning
            words = page.extract_words(x_tolerance=3, y_tolerance=3)
            if not words:
                return None
            
            # Get lines of text
            lines = self._group_words_into_lines(words)
            if not lines:
                return None
            
            # If we have table region info, look above it
            search_lines = lines
            if table_region:
                # Look for title in lines above the table region
                table_top = min(word.get('top', 0) for line in table_region for word in line if isinstance(word, dict))
                search_lines = [line for line in lines if any(word.get('top', 0) < table_top - 10 for word in line)]
            
            # Look for table title patterns in reverse order (closest to table first)
            for line in reversed(search_lines):
                line_text = ' '.join([w.get('text', '') for w in line if w.get('text')])
                
                # Check for table title patterns
                for pattern in self.table_title_patterns:
                    if re.search(pattern, line_text, re.IGNORECASE):
                        return line_text.strip()
                
                # Check for pharmaceutical table indicators
                pharma_indicators = ['treatment', 'therapy', 'medicinal', 'costs', 'dosage', 'consumption']
                if any(indicator in line_text.lower() for indicator in pharma_indicators):
                    # Make sure it's not too long (likely not a title if > 100 chars)
                    if len(line_text) <= 100:
                        return line_text.strip()
            
            # Fallback: look for any short line with key pharmaceutical terms
            for line in reversed(search_lines[-10:]):  # Last 10 lines before table
                line_text = ' '.join([w.get('text', '') for w in line if w.get('text')])
                if (len(line_text.split()) <= 10 and 
                    any(term in line_text.lower() for term in ['table', 'designation', 'treatment', 'costs', 'therapy'])):
                    return line_text.strip()
            
            return None
            
        except Exception as e:
            print(f"      Error finding table title: {e}")
            return None

    def _estimate_table_region(self, table_data):
        """Estimate the region occupied by a table for proximity detection."""
        # Simple region estimation - in practice this could be more sophisticated
        return {
            "estimated_rows": len(table_data) if table_data else 0,
            "estimated_cols": len(table_data[0]) if table_data and table_data[0] else 0
        }

    def _create_table_metadata(self, table_data, table_title, extraction_method, page_num):
        """Create comprehensive metadata for detected tables."""
        metadata = {
            "original_rows": len(table_data) if table_data else 0,
            "extraction_method": extraction_method,
            "has_title": bool(table_title),
            "page": page_num
        }
        
        # Add medical-specific metadata
        if table_data:
            cleaned_table = self.pdf_processor.clean_table_data(table_data)
            all_text = ' '.join([' '.join(row) for row in cleaned_table if row])
            
            metadata.update({
                "contains_dosage": bool(re.search(r'\d+\s*(?:mg|μg|ml)', all_text, re.IGNORECASE)),
                "contains_pricing": bool(re.search(r'[€$]\s*\d+|cost|price', all_text, re.IGNORECASE)),
                "contains_medication": bool(re.search(r'sorafenib|lenvatinib|treatment|therapy', all_text, re.IGNORECASE)),
                "table_title": table_title if table_title else None,
                "narrative_length": 0  # Will be updated after narrative creation
            })
        
        return metadata

    def convert_table_to_narrative(self, table_data, table_title=None):
        """Convert table to hybrid format with narrative and structured metadata."""
        if not table_data or len(table_data) == 0:
            return ""
        
        cleaned_table = self.pdf_processor.clean_table_data(table_data)
        if len(cleaned_table) == 0:
            return ""
        
        # Generate narrative (existing logic)
        narrative_parts = []
        
        if table_title:
            narrative_parts.append(f"Table Title: {table_title}")
            narrative_parts.append("")
        
        headers = self.identify_table_headers(cleaned_table)
        
        if headers:
            narrative_parts.append(f"Table contains the following columns: {', '.join(headers)}")
            data_rows = cleaned_table[1:] if len(cleaned_table) > 1 else []
            
            for row_idx, row in enumerate(data_rows, 1):
                row_description = self.create_row_description(headers, row, row_idx)
                if row_description:
                    narrative_parts.append(row_description)
        else:
            title_part = f"Table Title: {table_title}\n\n" if table_title else ""
            narrative_content = title_part + self.convert_headerless_table(cleaned_table)
            return narrative_content
        
        narrative_text = "\n".join(narrative_parts)
        
        return narrative_text


    def _classify_table_content(self, text):
        """Classify the primary content type of the table."""
        text_lower = text.lower()
        
        if re.search(r'cost|price|€|\$', text_lower):
            return "pricing"
        elif re.search(r'\d+\s*(?:mg|μg|ml)', text_lower):
            return "dosage"
        elif re.search(r'patient|study|trial', text_lower):
            return "clinical_data"
        elif re.search(r'treatment|therapy|medicinal', text_lower):
            return "treatment_info"
        else:
            return "general"
    
    def identify_table_headers(self, cleaned_table):
        """
        Try to identify table headers from the first row(s).
        """
        if not cleaned_table:
            return None
        
        first_row = cleaned_table[0]
        
        # Check if first row looks like headers
        header_indicators = [
            # Check for typical header words
            any(word.lower() in cell.lower() for word in ['name', 'type', 'value', 'result', 'outcome', 'dose', 'drug', 'treatment', 'group', 'arm', 'study', 'n=', 'patient', 'endpoint', 'efficacy', 'safety', 'adverse', 'event'] for cell in first_row if cell),
            
            # Check if cells are short (typical of headers)
            all(len(cell.split()) <= 4 for cell in first_row if cell),
            
            # Check if subsequent rows have different content patterns
            len(cleaned_table) > 1 and any(
                self.is_numeric(cell) for cell in cleaned_table[1] if cell
            )
        ]
        
        if any(header_indicators):
            return [cell if cell else f"Column_{i+1}" for i, cell in enumerate(first_row)]
        
        return None

    def create_row_description(self, headers, row, row_idx):
        """
        Create a descriptive sentence for a table row.
        """
        if not headers or not row:
            return ""
        
        # Pair headers with values
        pairs = []
        for i, (header, value) in enumerate(zip(headers, row)):
            if value and value.strip():
                # Clean header and value
                clean_header = header.strip().rstrip(':')
                clean_value = value.strip()
                
                pairs.append(f"{clean_header}: {clean_value}")
        
        if not pairs:
            return ""
        
        # Create a descriptive sentence
        if len(pairs) == 1:
            return f"Row {row_idx}: {pairs[0]}"
        elif len(pairs) == 2:
            return f"Row {row_idx}: {pairs[0]} and {pairs[1]}"
        else:
            # Multiple pairs - create a structured description
            return f"Row {row_idx}: {', '.join(pairs[:-1])}, and {pairs[-1]}"

    def convert_headerless_table(self, cleaned_table):
        """
        Convert a table without clear headers into narrative form.
        """
        narrative_parts = []
        
        for row_idx, row in enumerate(cleaned_table, 1):
            # Filter out empty cells
            non_empty_cells = [cell for cell in row if cell and cell.strip()]
            
            if non_empty_cells:
                if len(non_empty_cells) == 1:
                    narrative_parts.append(f"Row {row_idx}: {non_empty_cells[0]}")
                else:
                    # Join multiple cells with descriptive text
                    cell_desc = ", ".join(f"'{cell}'" for cell in non_empty_cells)
                    narrative_parts.append(f"Row {row_idx} contains: {cell_desc}")
        
        return "\n".join(narrative_parts)

    def is_numeric(self, text: str) -> bool:
        """Check whether a text string represents a numeric value."""
        if not text or not isinstance(text, str):
            return False
        
        text = text.strip()
        if not text:
            return False
        
        # Remove common non-numeric characters that might appear in numbers
        cleaned = text.replace(',', '').replace(' ', '')
        
        try:
            # Try to convert to float
            float(cleaned)
            return True
        except (ValueError, TypeError):
            pass
        
        # Check for percentage
        if text.endswith('%'):
            try:
                float(text[:-1].replace(',', '').replace(' ', ''))
                return True
            except (ValueError, TypeError):
                pass
        
        # Check for simple patterns like "< 0.001" or "> 100"
        try:
            if re.match(r'^[<>=≤≥]\s*[\d.,]+$', text):
                return True
        except (TypeError, re.error):
            pass
        
        # Check for ranges like "1.2-3.4"
        try:
            if re.match(r'^\d+\.?\d*[-–]\d+\.?\d*$', text):
                return True
        except (TypeError, re.error):
            pass
        
        return False
    
    def _is_numeric_like(self, text: str) -> bool:
        """Simple check if text represents numeric data."""
        if not text:
            return False
        
        # Pure numbers
        if re.match(r'^\d+([.,]\d+)?$', text):
            return True
        
        # Numbers with units/symbols
        if re.match(r'^\d+([.,]\d+)?\s*[%€$£¥]$', text):
            return True
        
        # Ranges
        if re.match(r'^\d+([.,]\d+)?\s*[-–]\s*\d+([.,]\d+)?$', text):
            return True
        
        # Comparison operators
        if re.match(r'^[<>=≤≥]\s*\d+([.,]\d+)?$', text):
            return True

        return False

    def has_explicit_table_structure(self, page) -> bool:
        """Check for a clear table grid on the page."""
        try:
            lines = page.lines if hasattr(page, 'lines') else []
            if len(lines) < 6:
                return False

            horizontal_lines = []
            vertical_lines = []

            for line in lines:
                x1, y1, x2, y2 = line.get('x0', 0), line.get('y0', 0), line.get('x1', 0), line.get('y1', 0)
                width = abs(x2 - x1)
                height = abs(y2 - y1)

                if width > 100 and height < 3:
                    horizontal_lines.append(line)
                elif height > 30 and width < 3:
                    vertical_lines.append(line)

            if len(horizontal_lines) >= 3 and len(vertical_lines) >= 2:
                return self._check_line_intersections(horizontal_lines, vertical_lines)

            return False
        except Exception:
            return False

    def _check_line_intersections(self, h_lines, v_lines) -> bool:
        """Simple intersection check for grid formation."""
        try:
            intersections = 0

            for h_line in h_lines[:5]:
                h_y = h_line.get('y0', 0)
                h_x1, h_x2 = h_line.get('x0', 0), h_line.get('x1', 0)

                for v_line in v_lines[:4]:
                    v_x = v_line.get('x0', 0)
                    v_y1, v_y2 = v_line.get('y0', 0), v_line.get('y1', 0)

                    if (min(h_x1, h_x2) <= v_x <= max(h_x1, h_x2) and
                            min(v_y1, v_y2) <= h_y <= max(v_y1, v_y2)):
                        intersections += 1

            expected_min = min(len(h_lines), 5) * min(len(v_lines), 4) * 0.3
            return intersections >= expected_min
        except Exception:
            return False


    def extract_tables_ultra_strict(self, page):
        """Extract tables with strict visual settings."""
        try:
            explicit_v_lines = self.detect_vertical_table_lines_strict(page)
            explicit_h_lines = self.detect_horizontal_table_lines_strict(page)

            if explicit_v_lines and explicit_h_lines:
                tables = page.extract_tables(
                    table_settings={
                        "vertical_strategy": "explicit",
                        "horizontal_strategy": "explicit",
                        "explicit_vertical_lines": explicit_v_lines,
                        "explicit_horizontal_lines": explicit_h_lines,
                        "min_words_vertical": 3,
                        "min_words_horizontal": 2,
                        "keep_blank_chars": False,
                        "text_tolerance": 2,
                        "intersection_tolerance": 2,
                    }
                )
                if tables:
                    return tables

            tables = page.extract_tables(
                table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "min_words_vertical": 4,
                    "min_words_horizontal": 3,
                    "keep_blank_chars": False,
                    "text_tolerance": 1,
                    "intersection_tolerance": 1,
                }
            )

            return tables if tables else []
        except Exception:
            return []

    def detect_vertical_table_lines_strict(self, page) -> List[float]:
        """Detect vertical lines with relaxed criteria."""
        try:
            lines = page.lines if hasattr(page, 'lines') else []
            vertical_lines = []

            for line in lines:
                x1, y1, x2, y2 = line.get('x0', 0), line.get('y0', 0), line.get('x1', 0), line.get('y1', 0)
                line_width = abs(x2 - x1)
                line_height = abs(y2 - y1)

                if line_height > 25 and line_width < 6 and line_height > line_width * 8:
                    vertical_lines.append(x1)

            unique_lines = []
            for x in sorted(vertical_lines):
                if not unique_lines or abs(x - unique_lines[-1]) > 5:
                    unique_lines.append(x)

            return unique_lines
        except Exception:
            return []

    def detect_horizontal_table_lines_strict(self, page) -> List[float]:
        """Detect horizontal lines with relaxed criteria."""
        try:
            lines = page.lines if hasattr(page, 'lines') else []
            horizontal_lines = []

            for line in lines:
                x1, y1, x2, y2 = line.get('x0', 0), line.get('y0', 0), line.get('x1', 0), line.get('y1', 0)
                line_width = abs(x2 - x1)
                line_height = abs(y2 - y1)

                if line_width > 40 and line_height < 6 and line_width > line_height * 12:
                    horizontal_lines.append(y1)

            unique_lines = []
            for y in sorted(horizontal_lines):
                if not unique_lines or abs(y - unique_lines[-1]) > 3:
                    unique_lines.append(y)

            return unique_lines
        except Exception:
            return []


    def detect_complete_tables(self, pdf):
        """Enhanced multi-pass table detection with graduated sensitivity."""
        tables_info = []
        detected_regions = []  # Track regions where tables were found

        print("  🔍 Multi-pass table detection with graduated sensitivity...")

        for page_num, page in enumerate(pdf.pages, start=1):
            page_tables = []
            
            # Pass 1: Standard detection (sensitivity level 1)
            if self.has_explicit_table_structure(page):
                visual_tables = self.extract_tables_ultra_strict(page)
                if visual_tables:
                    page_tables.extend(visual_tables)

            if not page_tables:
                try:
                    standard_tables = page.extract_tables()
                    if standard_tables:
                        page_tables.extend(standard_tables)
                except Exception as e:
                    print(f"      Page {page_num}: Standard extraction failed: {e}")

            # Validate Pass 1 results
            pass_1_tables = []
            for table_idx, table_data in enumerate(page_tables):
                if self.enhanced_table_validation(table_data, page, page_num, sensitivity_level=1):
                    table_title = self.find_table_title(page)
                    narrative_text = self.convert_table_to_narrative(table_data, table_title)
                    
                    if narrative_text.strip():
                        heading = table_title if table_title else f"Table {table_idx + 1} on page {page_num}"
                        
                        table_info = {
                            "page": page_num,
                            "heading": heading,
                            "text": narrative_text,
                            "table_type": "multi_pass_validated_table",
                            "table_metadata": self._create_table_metadata(table_data, table_title, "pass_1_standard", page_num)
                        }
                        
                        pass_1_tables.append(table_info)
                        # Track detected region for pass 2
                        detected_regions.append((page_num, self._estimate_table_region(table_data)))

            # Pass 2: Relaxed detection near existing tables (sensitivity level 2)
            if detected_regions:
                additional_tables = []
                for table_idx, table_data in enumerate(page_tables):
                    # Skip if already validated in pass 1
                    already_found = any(t["page"] == page_num for t in pass_1_tables)
                    if not already_found and self.enhanced_table_validation(table_data, page, page_num, sensitivity_level=2):
                        table_title = self.find_table_title(page)
                        narrative_text = self.convert_table_to_narrative(table_data, table_title)
                        
                        if narrative_text.strip():
                            heading = table_title if table_title else f"Table {table_idx + 1} on page {page_num}"
                            
                            table_info = {
                                "page": page_num,
                                "heading": heading,
                                "text": narrative_text,
                                "table_type": "multi_pass_validated_table",
                                "table_metadata": self._create_table_metadata(table_data, table_title, "pass_2_relaxed", page_num)
                            }
                            
                            additional_tables.append(table_info)

                pass_1_tables.extend(additional_tables)

            # Pass 3: Medical domain-specific detection (sensitivity level 3)
            medical_tables = []
            for table_idx, table_data in enumerate(page_tables):
                # Skip if already found in previous passes
                already_found = any(t["page"] == page_num for t in pass_1_tables)
                if not already_found and self.enhanced_table_validation(table_data, page, page_num, sensitivity_level=3):
                    table_title = self.find_table_title(page)
                    narrative_text = self.convert_table_to_narrative(table_data, table_title)
                    
                    if narrative_text.strip():
                        heading = table_title if table_title else f"Medical Table {table_idx + 1} on page {page_num}"
                        
                        table_info = {
                            "page": page_num,
                            "heading": heading,
                            "text": narrative_text,
                            "table_type": "multi_pass_validated_table", 
                            "table_metadata": self._create_table_metadata(table_data, table_title, "pass_3_medical", page_num)
                        }
                        
                        medical_tables.append(table_info)

            pass_1_tables.extend(medical_tables)
            
            if pass_1_tables:
                print(f"      Page {page_num}: Found {len(pass_1_tables)} tables via multi-pass detection")
            
            tables_info.extend(pass_1_tables)

        print(f"  ✅ Multi-pass detection complete: {len(tables_info)} validated tables")
        return tables_info


class PDFProcessor:
    def __init__(
        self,
        pdf_path,
        boilerplate_threshold=0.3,
        doc_id=None,
        language="unknown",
        region="unknown",
        title="Unknown Title",
        created_date="unknown_date",
        keywords=None
    ):
        """Initialize processor and gather basic metadata."""
        self.pdf_path = pdf_path
        self.boilerplate_threshold = boilerplate_threshold
        self.doc_id = doc_id or os.path.splitext(os.path.basename(pdf_path))[0]
        self.language = language
        self.region = region
        self.title = title
        self.created_date = created_date
        self.keywords = keywords or []
        self.global_headings_list = []
        self.page_headings_map = defaultdict(list)

        # Initialize table detection class
        self.table_detector = TableDetector(self)

        # Immediately extract and store submission year upon initialization
        self.created_date = self.find_submission_year()

        # Extract country from folder structure
        self.country = self.extract_country_from_path()

        # Extract source type 
        self.source_type = self.extract_source_type_from_path()


        print("--------------------------------------------")
        print(f"Source type for '{self.pdf_path}': {self.source_type}")
        print(f"Submission year for '{self.pdf_path}': {self.created_date}")
        print(f"Country for '{self.pdf_path}': {self.country}")

    def extract_source_type_from_path(self):
        """Identifies whether this is an HTA submission or clinical guideline."""
        if "hta submission" in self.pdf_path.lower() or "hta submissions" in self.pdf_path.lower():
            return "hta_submission"
        elif "clinical guideline" in self.pdf_path.lower() or "clinical guidelines" in self.pdf_path.lower():
            return "clinical_guideline"
        else:
            return "unknown"

    def extract_country_from_path(self):
        """Extracts the country code from the parent directory name."""
        parent_dir = os.path.basename(os.path.dirname(self.pdf_path))
        
        # All EU country codes plus additional European countries and special codes
        country_codes = {
            # EU member states
            "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", 
            "DE", "EL", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", 
            "NL", "PL", "PO", "PT", "RO", "SK", "SI", "ES", "SE",
            
            # Non-EU European countries
            "CH", "NO", "IS", "UK", "GB", "UA", "RS", "ME", "MK", "AL", 
            "BA", "MD", "XK", "LI",
            
            # Special codes and regions
            "EU", "EN", "INT", # INT for international
            
            # Other codes found in your folder structure
            "AE", "TR"
        }
        
        return parent_dir if parent_dir in country_codes else "unknown"

    def find_submission_year(self):
        """
        Finds the submission/report year from the first page of the PDF.
        Stores the year in self.created_date, or 'unknown_year' if none found.
        """
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                if not pdf.pages:
                    return "unknown_year"
                first_page_text = pdf.pages[0].extract_text() or ""
        except Exception:
            return "unknown_year"

        year_pattern = r"\b(19|20)\d{2}\b"

        triggers = [
            "submission date", "submitted on", "report date", "date of submission", "date of issue",
            "soumission", "data di presentazione", "fecha de presentación",
            "datum der einreichung", "fecha de remisión", "submitted:", "issued on", "rapport",
            "published:", "published", "publication date", "date of publication"
        ]

        lines = first_page_text.splitlines()

        for line in lines:
            line_lower = line.lower()
            if any(trigger in line_lower for trigger in triggers):
                years_in_line = re.findall(year_pattern, line)
                if years_in_line:
                    return years_in_line[0]

        all_years = re.findall(r"\b(?:19|20)\d{2}\b", first_page_text)
        if all_years:
            return all_years[0]

        return "unknown_year"
    
    def print_boilerplate_lines(self, boilerplate_normed, normed_line_pages):
        """Logs boilerplate lines that were removed from the PDF."""
        print(f"Boilerplate lines removed from {self.pdf_path}:")
        for normed in boilerplate_normed:
            pages = sorted(normed_line_pages[normed])
            print(f"'{normed}' found on pages: {pages}")

    @staticmethod
    def remove_links(line: str) -> str:
        """Removes URLs and empty parentheses from a given text line."""
        line_no_links = re.sub(r'\(?(?:https?://|www\.)\S+\)?', '', line)
        line_no_links = re.sub(r'\(\s*\)', '', line_no_links)
        return line_no_links.strip()

    @staticmethod
    def advanced_normalize(line: str) -> str:
        """Normalizes text by removing digits and non-alpha characters."""
        norm = line.lower()
        norm = re.sub(r'\d+', '', norm)
        norm = re.sub(r'[^a-z]+', '', norm)
        return norm.strip()

    @staticmethod
    def contains_of_contents(line: str) -> bool:
        """Checks if the line contains 'of contents', indicating a table of contents entry."""
        return bool(re.search(r'\bof\s+contents\b', line, re.IGNORECASE))

    @staticmethod
    def is_toc_dotline(line: str) -> bool:
        """Checks if a line is a dotted table of contents entry."""
        return bool(re.search(r'\.{5,}\s*\d+$', line))

    @staticmethod
    def is_footnote_source(line: str) -> bool:
        """Determines if the line is a reference or footnote source."""
        stripped = line.strip()
        if not stripped:
            return False

        if not re.match(r'^(\[\d+\]|[\d\.]+)\b', stripped):
            return False

        reference_patterns = [r'et al\.', r'Disponible en ligne', r'consulté le', r'NEJM', r'PubMed', r'doi']
        combined = '(' + '|'.join(reference_patterns) + ')'
        return bool(re.search(combined, stripped, flags=re.IGNORECASE))

    def merge_hyphenated_words(self, lines: List[str]) -> List[str]:
        """
        Merge hyphenated words that are split across lines.
        This handles cases like 'signifi-' on one line and 'cant' on the next.
        """
        if not lines:
            return lines
            
        cleaned_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i].rstrip()
            
            # Check if current line ends with hyphen and there's a next line
            if i < len(lines) - 1 and line.endswith('-'):
                next_line = lines[i + 1].lstrip()
                
                # Only merge if the next line starts with a lowercase letter (continuation)
                # or if it's clearly a continuation (no punctuation at start)
                if (next_line and 
                    (next_line[0].islower() or 
                     not re.match(r'^[A-Z\d\(\[\{]', next_line))):
                    
                    # Remove hyphen and merge with next line
                    merged = line[:-1] + next_line
                    cleaned_lines.append(merged)
                    i += 2  # Skip the next line as it's been merged
                else:
                    # Keep the hyphen if it doesn't look like a word break
                    cleaned_lines.append(line)
                    i += 1
            else:
                cleaned_lines.append(line)
                i += 1
                
        return cleaned_lines

    def improve_paragraph_cohesion(self, lines: List[str]) -> List[str]:
        """
        Improve paragraph cohesion by joining lines that appear to be 
        continuation of the same sentence or paragraph.
        """
        if not lines:
            return lines
            
        cohesive_lines = []
        i = 0
        
        while i < len(lines):
            current_line = lines[i].strip()
            
            if not current_line:
                cohesive_lines.append(current_line)
                i += 1
                continue
            
            # Look ahead to see if we should merge with next line
            if i < len(lines) - 1:
                next_line = lines[i + 1].strip()
                
                # Conditions for merging:
                # 1. Current line doesn't end with sentence-ending punctuation
                # 2. Next line exists and isn't empty
                # 3. Next line starts with lowercase (likely continuation)
                # 4. Current line doesn't look like a heading or list item
                should_merge = (
                    current_line and next_line and
                    not re.search(r'[.!?:;]', current_line) and
                    next_line[0].islower() and
                    not re.match(r'^\d+[\.\)]\s', current_line) and  # Not a numbered list
                    not re.match(r'^[•\-\*]\s', current_line) and   # Not a bullet list
                    len(current_line.split()) > 1  # Not a single word (likely heading)
                )
                
                if should_merge:
                    # Merge current line with next line
                    merged_line = current_line + ' ' + next_line
                    cohesive_lines.append(merged_line)
                    i += 2  # Skip the next line
                else:
                    cohesive_lines.append(current_line)
                    i += 1
            else:
                cohesive_lines.append(current_line)
                i += 1
                
        return cohesive_lines

    def detect_headings_by_font(self, page):
        """
        Detect headings based on font size, boldness, numbering, uppercase, length,
        and vertical spacing (space above and below).
        """
        words = page.extract_words(
            x_tolerance=2,
            y_tolerance=2,
            extra_attrs=["fontname", "size"]
        )
        if not words:
            return []

        # Sort words top->down, left->right
        words_sorted = sorted(words, key=lambda w: (round(w.get('top', 0)), w.get('x0', 0)))

        # 1) Group words into raw lines
        raw_lines = []
        current_line = []
        line_tolerance = 5

        for w in words_sorted:
            if w.get('size') is None or w.get('top') is None:
                continue

            if not current_line:
                current_line = [w]
                continue

            last_word = current_line[-1]
            if abs(w['top'] - last_word['top']) <= line_tolerance:
                current_line.append(w)
            else:
                raw_lines.append(current_line)
                current_line = [w]
        if current_line:
            raw_lines.append(current_line)

        # 2) Merge adjacent lines if same font style, etc. (multi-line headings)
        merged_lines = []
        if raw_lines:
            current_group = raw_lines[0]

            for next_line in raw_lines[1:]:
                if not current_group:
                    current_group = next_line
                    continue
                    
                current_avg_size = statistics.mean([w['size'] for w in current_group if 'size' in w])
                next_avg_size = statistics.mean([w['size'] for w in next_line if 'size' in w])
                current_bold = all('bold' in w.get('fontname', '').lower() for w in current_group)
                next_bold = all('bold' in w.get('fontname', '').lower() for w in next_line)
                
                # Calculate vertical gap safely
                current_bottom = max(w.get('bottom', 0) for w in current_group)
                next_top = min(w.get('top', 0) for w in next_line)
                vertical_gap = next_top - current_bottom

                if (
                    abs(current_avg_size - next_avg_size) < 1.0 and
                    current_bold == next_bold and
                    vertical_gap <= 10
                ):
                    current_group.extend(next_line)
                else:
                    merged_lines.append(current_group)
                    current_group = next_line
            if current_group:
                merged_lines.append(current_group)

        # 3) Build line_objects
        line_objects = []
        font_sizes = []

        for line_words in merged_lines:
            text_parts = [lw["text"] for lw in line_words if lw.get("text")]
            if not text_parts:
                continue

            text = " ".join(text_parts).strip()
            sizes = [lw['size'] for lw in line_words if lw.get('size') is not None]
            avg_size = statistics.mean(sizes) if sizes else 10.0
            top_pos = min(lw.get('top', 0) for lw in line_words)
            bottom_pos = max(lw.get('bottom', 0) for lw in line_words)

            line_objects.append({
                "text": text,
                "avg_size": avg_size,
                "top": top_pos,
                "bottom": bottom_pos,
                "words": line_words
            })
            font_sizes.append(avg_size)

        if not line_objects:
            return []

        median_size = statistics.median(font_sizes)

        # 4) Score each line
        for idx, obj in enumerate(line_objects):
            text = obj["text"]
            avg_size = obj["avg_size"]
            lw = obj["words"]
            word_count = len(text.split())

            # Font size ratio
            ratio_to_median = avg_size / median_size if median_size else 1.0
            font_score = 2 if ratio_to_median >= 1.3 else (1 if ratio_to_median >= 1.1 else 0)

            # Uppercase ratio
            letters = [c for c in text if c.isalpha()]
            uppercase_ratio = sum(c.isupper() for c in letters) / len(letters) if letters else 0
            uppercase_score = 2 if uppercase_ratio > 0.8 else (1 if uppercase_ratio > 0.6 else 0)

            # Length
            length_score = 2 if word_count < 6 else (1 if word_count < 12 else 0)

            # Boldface
            bold_words = [w for w in lw if w.get('fontname') and 'bold' in w['fontname'].lower()]
            bold_ratio = len(bold_words) / len(lw) if lw else 0
            bold_score = 2 if bold_ratio > 0.7 else (1 if bold_ratio > 0.3 else 0)

            # Numbering
            numbering_score = 0
            numbering_pattern = r'^(\d+(\.\d+)*[\.\,\)]?\s+)'
            if re.match(numbering_pattern, text):
                numbering_score = 2

            # Vertical spacing
            vertical_space_score = 0
            space_above = obj["top"] - line_objects[idx-1]["bottom"] if idx > 0 else 100
            space_below = line_objects[idx+1]["top"] - obj["bottom"] if idx < len(line_objects)-1 else 100

            if space_above > 15 and space_below > 10:
                vertical_space_score = 2
            elif space_above > 10 or space_below > 8:
                vertical_space_score = 1

            total_score = (
                font_score +
                uppercase_score +
                length_score +
                bold_score +
                numbering_score +
                vertical_space_score
            )
            obj["likely_heading"] = (total_score >= 5)

        return line_objects


    def clean_table_data(self, table_data):
        """
        Clean and standardize table data.
        """
        cleaned = []
        
        for row in table_data:
            if not row:
                continue
            
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cell_text = ""
                else:
                    # Clean cell content
                    cell_text = re.sub(r'\s+', ' ', str(cell)).strip()
                    
                cleaned_row.append(cell_text)
            
            # Only include rows that have some content
            if any(cell.strip() for cell in cleaned_row):
                cleaned.append(cleaned_row)
        
        return cleaned


    def extract_text_by_columns(self, page):
        """
        Extract text from a page considering column layout.
        Returns text organized by columns and preserving reading order.
        Enhanced with hyphenation and paragraph cohesion improvements.
        """
        num_columns, column_boundaries = self.detect_columns(page)
        
        if num_columns == 1:
            # For single column, extract text and apply improvements
            column_text = page.extract_text()
            if column_text:
                lines = column_text.split('\n')
                # Apply hyphenation merging
                lines = self.merge_hyphenated_words(lines)
                # Apply paragraph cohesion
                lines = self.improve_paragraph_cohesion(lines)
                return ['\n'.join(lines)]
            return [""]
        
        # For multi-column layout, extract text for each column separately
        column_texts = []
        
        for i in range(num_columns):
            left_bound = column_boundaries[i]
            right_bound = column_boundaries[i+1]
            
            # Extract text only within this column's boundaries
            column_area = (left_bound, 0, right_bound, page.height)
            column_text = page.crop(column_area).extract_text()
            
            if column_text and column_text.strip():
                lines = column_text.split('\n')
                # Apply hyphenation merging
                lines = self.merge_hyphenated_words(lines)
                # Apply paragraph cohesion
                lines = self.improve_paragraph_cohesion(lines)
                column_texts.append('\n'.join(lines))
        
        return column_texts

    def detect_columns(self, page):
        """
        Detect if a page has multiple columns by analyzing text positions.
        Returns the number of columns and their x-boundaries.
        """
        words = page.extract_words(x_tolerance=2, y_tolerance=2)
        if not words:
            return 1, []  # Default to single column if no words
            
        # Collect x-positions (horizontal position) of words
        x_positions = [word['x0'] for word in words]
        
        # Identify potential column gaps using histogram analysis
        hist, bin_edges = np.histogram(x_positions, bins=20)
        
        # Find significant gaps in word positions
        significant_gaps = []
        for i in range(len(hist)):
            if hist[i] < max(hist) * 0.1:  # Threshold for considering a gap
                left_edge = bin_edges[i]
                right_edge = bin_edges[i+1]
                middle = (left_edge + right_edge) / 2
                significant_gaps.append(middle)
        
        # Determine number of columns based on gaps
        if len(significant_gaps) == 0:
            return 1, []  # Single column
        elif len(significant_gaps) == 1:
            # Likely a two-column layout
            # Determine boundaries between columns
            column_boundaries = [0, significant_gaps[0], page.width]
            return 2, column_boundaries
        else:
            # Multi-column layout (more than two)
            # Sort gaps and create column boundaries
            significant_gaps.sort()
            column_boundaries = [0] + significant_gaps + [page.width]
            return len(column_boundaries) - 1, column_boundaries

    def extract_preliminary_chunks(self):
        """
        Main function that:
        1. Extracts text and identifies headings from each page (skipping footnotes, boilerplate, etc.).
        2. Stores headings per page in self.page_headings_map.
        3. Extracts tables using enhanced multi-pass detection and hybrid metadata conversion.
        4. Returns a dictionary with enhanced text processing including hyphenation and paragraph cohesion.
        Enhanced with better error handling for problematic PDFs and improved table handling.
        """
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                # We'll first gather all lines/heading info (without handling tables)
                num_pages = len(pdf.pages)
                normed_line_pages = {}
                pages_with_lines = []
                problematic_pages = 0
                total_pages_processed = 0

                # Pass 1: gather potential headings and normal lines, handling columns
                for i, page in enumerate(pdf.pages, start=1):
                    try:
                        # First detect headings using font information
                        line_objs = self.detect_headings_by_font(page)
                        
                        # Create a mapping of heading text to heading status
                        likely_headings = {}
                        for lo in line_objs:
                            likely_headings[lo["text"]] = lo.get("likely_heading", False)
                        
                        # Extract text column by column with enhanced processing
                        column_texts = self.extract_text_by_columns(page)
                        
                        # Track all lines with their page numbers for boilerplate detection
                        all_lines = []
                        
                        # Process each column
                        for column_text in column_texts:
                            # Split column text into lines
                            col_lines = column_text.split('\n')
                            
                            # Process each line for headings and content
                            for line in col_lines:
                                if not line.strip():
                                    continue
                                    
                                # Create a simple line object
                                line_obj = {
                                    "text": line.strip(),
                                    "likely_heading": False
                                }
                                
                                # Check if this line matches any of our detected headings
                                for heading_text, is_heading in likely_headings.items():
                                    if heading_text in line and is_heading:
                                        line_obj["likely_heading"] = True
                                        break
                                
                                all_lines.append(line_obj)
                            
                        pages_with_lines.append((i, all_lines))
                        total_pages_processed += 1
                        
                        # Identify potential boilerplate lines
                        unique_normed = set()
                        for lo in all_lines:
                            norm = self.advanced_normalize(lo["text"])
                            if norm:
                                unique_normed.add(norm)
                        for n in unique_normed:
                            normed_line_pages.setdefault(n, set()).add(i)
                    
                    except ValueError as page_err:
                        # Check specifically for negative width/height error
                        if "negative width or height" in str(page_err):
                            print(f"Warning: Skipping page {i} due to layout issues")
                            problematic_pages += 1
                            continue
                        else:
                            raise page_err
                    except Exception as page_err:
                        print(f"Warning: Error processing page {i}: {page_err}")
                        problematic_pages += 1
                        continue

                # If too many pages were problematic, switch to fallback method
                if problematic_pages > 0 and (total_pages_processed == 0 or problematic_pages / num_pages > 0.2):
                    print(f"Too many problematic pages ({problematic_pages}/{num_pages}). Switching to fallback method.")
                    return self.extract_using_fallback()

                # If we couldn't extract any content, try the fallback method
                if not pages_with_lines:
                    print(f"No content extracted from {self.pdf_path}. Switching to fallback method.")
                    return self.extract_using_fallback()

                # Determine boilerplate lines
                boilerplate_normed = set()
                for normed_line, pset in normed_line_pages.items():
                    if len(pset) / num_pages >= self.boilerplate_threshold:
                        boilerplate_normed.add(normed_line)
                self.print_boilerplate_lines(boilerplate_normed, normed_line_pages)

                filtered_lines = []
                in_footnote = False

                # Filter out footnotes and boilerplate
                for page_num, line_objs in pages_with_lines:
                    for lo in line_objs:
                        line_text = lo["text"]
                        if not line_text.strip():
                            continue

                        if in_footnote:
                            # If we hit a new heading or an empty line, end footnote consumption
                            if not line_text.strip() or lo["likely_heading"]:
                                in_footnote = False
                                continue
                            else:
                                continue

                        # If line looks like a footnote reference, skip
                        if self.is_footnote_source(line_text):
                            in_footnote = True
                            continue

                        line_text = self.remove_links(line_text)
                        if not line_text.strip():
                            continue

                        norm = self.advanced_normalize(line_text)
                        if not norm or norm in boilerplate_normed:
                            continue
                        if self.contains_of_contents(line_text):
                            continue
                        if self.is_toc_dotline(line_text):
                            continue

                        heading_flag = lo["likely_heading"]
                        filtered_lines.append((page_num, line_text.strip(), heading_flag))

                # Build text chunks, track headings
                chunks = []
                buffer = []
                current_heading = ""

                def flush_buffer():
                    """Dump the buffered lines into a chunk if any."""
                    nonlocal current_heading
                    if buffer:
                        min_page = min(x[0] for x in buffer)
                        max_page = max(x[0] for x in buffer)
                        chunk_text = "\n".join(x[1] for x in buffer)
                        chunks.append({
                            "heading": current_heading,
                            "text": chunk_text,
                            "start_page": min_page,
                            "end_page": max_page
                        })
                        buffer.clear()

                for (pnum, text_line, is_heading) in filtered_lines:
                    if is_heading:
                        # Before we switch heading, flush existing buffer
                        flush_buffer()
                        current_heading = text_line
                        self.global_headings_list.append((pnum, text_line))
                        self.page_headings_map[pnum].append(text_line)
                    else:
                        buffer.append((pnum, text_line))

                flush_buffer()

                # Now extract tables with enhanced multi-pass detection and hybrid metadata conversion
                tables_info = self.table_detector.detect_complete_tables(pdf)

                # Insert each table as its own chunk with hybrid metadata
                for tinfo in tables_info:
                    pg = tinfo["page"]
                    heading_for_table = tinfo["heading"]
                    table_text = tinfo["text"]
                    
                    # Get enhanced metadata from table detection
                    table_metadata = tinfo.get("table_metadata", {})
                    
                    # Update narrative length in metadata now that we have the text
                    if "narrative_length" in table_metadata:
                        table_metadata["narrative_length"] = len(table_text)
                    
                    # Create enhanced table chunk with hybrid metadata support
                    table_chunk = {
                        "heading": heading_for_table,
                        "text": table_text,
                        "start_page": pg,
                        "end_page": pg,
                        "table_type": tinfo["table_type"],
                        "table_metadata": table_metadata
                    }
                    
                    # Add structured metadata for hybrid approach
                    if table_metadata.get("contains_dosage") or table_metadata.get("contains_pricing"):
                        # This table contains medical data, add extraction hints
                        extraction_hints = {
                            "key_numbers": self._extract_key_numbers(table_text),
                            "key_terms": self._extract_key_medical_terms(table_text),
                            "relationships": self._identify_data_relationships(table_metadata)
                        }
                        table_metadata["extraction_hints"] = extraction_hints
                    
                    chunks.append(table_chunk)

                # Create final document structure with enhanced table summary
                final_structure = {
                    "doc_id": self.doc_id,
                    "created_date": self.created_date,
                    "country": self.country,
                    "source_type": self.source_type,
                    "chunks": chunks
                }
                
                # Add enhanced table summary with hybrid metadata insights
                if tables_info:
                    table_summary = {
                        "total_tables_found": len(tables_info),
                        "tables_by_page": {},
                        "table_storage_info": "Tables stored as individual chunks with hybrid metadata (narrative + structured)",
                        "medical_table_insights": {
                            "pricing_tables": sum(1 for t in tables_info if t.get("table_metadata", {}).get("contains_pricing", False)),
                            "dosage_tables": sum(1 for t in tables_info if t.get("table_metadata", {}).get("contains_dosage", False)),
                            "medication_tables": sum(1 for t in tables_info if t.get("table_metadata", {}).get("contains_medication", False)),
                            "multi_pass_detection_summary": {
                                "pass_1_standard": sum(1 for t in tables_info if t.get("table_metadata", {}).get("extraction_method") == "pass_1_standard"),
                                "pass_2_relaxed": sum(1 for t in tables_info if t.get("table_metadata", {}).get("extraction_method") == "pass_2_relaxed"),
                                "pass_3_medical": sum(1 for t in tables_info if t.get("table_metadata", {}).get("extraction_method") == "pass_3_medical")
                            }
                        }
                    }
                    
                    for tinfo in tables_info:
                        page_num = tinfo["page"]
                        if page_num not in table_summary["tables_by_page"]:
                            table_summary["tables_by_page"][page_num] = []
                        
                        page_table_info = {
                            "heading": tinfo["heading"],
                            "narrative_length": len(tinfo["text"]),
                            "extraction_method": tinfo.get("table_metadata", {}).get("extraction_method", "unknown"),
                            "original_rows": tinfo.get("table_metadata", {}).get("original_rows", "unknown"),
                            "primary_content_type": tinfo.get("table_metadata", {}).get("primary_content_type", "general"),
                            "contains_medical_data": bool(
                                tinfo.get("table_metadata", {}).get("contains_dosage") or 
                                tinfo.get("table_metadata", {}).get("contains_pricing") or 
                                tinfo.get("table_metadata", {}).get("contains_medication")
                            )
                        }
                        
                        table_summary["tables_by_page"][page_num].append(page_table_info)
                    
                    final_structure["_table_detection_summary"] = table_summary
                    
                    # Report enhanced table storage and detection insights
                    medical_count = table_summary["medical_table_insights"]["pricing_tables"] + table_summary["medical_table_insights"]["dosage_tables"]
                    print(f"  📍 Table storage: {len(tables_info)} tables stored as chunks with hybrid metadata")
                    print(f"  🏥 Medical table insights: {medical_count} tables contain dosage/pricing data")
                    print(f"  📊 Multi-pass detection: P1={table_summary['medical_table_insights']['multi_pass_detection_summary']['pass_1_standard']}, "
                            f"P2={table_summary['medical_table_insights']['multi_pass_detection_summary']['pass_2_relaxed']}, "
                            f"P3={table_summary['medical_table_insights']['multi_pass_detection_summary']['pass_3_medical']}")
                    print(f"  📋 Enhanced metadata added to '_table_detection_summary' field")

                return final_structure

        except Exception as e:
            print(f"Error reading {self.pdf_path}: {e}")
            # Try fallback method if primary extraction fails
            return self.extract_using_fallback()

    def _extract_key_numbers(self, text):
        """Extract key numerical values for extraction hints."""
        numbers = re.findall(r'\d+(?:[.,]\d+)?(?:\s*[€$%])?', text)
        return numbers[:10]  # Limit to first 10 to avoid bloat

    def _extract_key_medical_terms(self, text):
        """Extract key medical terms for extraction hints."""
        medical_terms = []
        common_terms = ['sorafenib', 'lenvatinib', 'treatment', 'therapy', 'dose', 'cost', 'patient', 'study']
        text_lower = text.lower()
        
        for term in common_terms:
            if term in text_lower:
                medical_terms.append(term)
        
        return medical_terms

    def _identify_data_relationships(self, metadata):
        """Identify potential data relationships based on metadata."""
        relationships = []
        
        if metadata.get("contains_dosage") and metadata.get("contains_medication"):
            relationships.append("drug-dose")
        
        if metadata.get("contains_pricing") and metadata.get("contains_medication"):
            relationships.append("drug-cost")
        
        if metadata.get("contains_dosage"):
            relationships.append("dose-frequency")
        
        if metadata.get("contains_comparisons"):
            relationships.append("comparative-analysis")
        
        return relationships


    def extract_using_fallback(self):
        """
        Fallback method for problematic PDFs that can't be processed normally.
        Uses PyPDF2 for more reliable text extraction when pdfplumber fails.
        Enhanced with hyphenation and paragraph cohesion improvements.
        """
        try:
            import PyPDF2
            
            print(f"Using fallback extraction for {self.pdf_path}")
            
            # Create a basic document structure
            doc_structure = {
                "doc_id": self.doc_id,
                "created_date": self.created_date,
                "country": self.country,
                "source_type": self.source_type,
                "chunks": []
            }
            
            # Open with PyPDF2 as a more robust alternative
            with open(self.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                print(f"PDF has {len(reader.pages)} pages")
                
                # Extract text page by page
                current_chunk_text = []
                current_heading = "Introduction"
                start_page = 1
                
                # Process each page
                for page_num, page in enumerate(reader.pages, 1):
                    try:
                        text = page.extract_text()
                        
                        # Skip empty pages
                        if not text or not text.strip():
                            continue
                        
                        # Apply text improvements to fallback method too
                        lines = text.split('\n')
                        lines = self.merge_hyphenated_words(lines)
                        lines = self.improve_paragraph_cohesion(lines)
                        improved_text = '\n'.join(lines)
                        
                        # Check for potential section headings
                        lines = improved_text.split('\n')
                        potential_heading = None
                        
                        for line in lines[:5]:  # Look at first few lines for potential headings
                            clean_line = line.strip()
                            if not clean_line:
                                continue
                                
                            # Simple heuristic for headings: short lines that are capitalized or numbered
                            if (len(clean_line.split()) <= 7 and 
                                (clean_line.isupper() or 
                                (any(char.isdigit() for char in clean_line[:3]) and 
                                not clean_line.lower().startswith("page")))):
                                potential_heading = clean_line
                                break
                        
                        # If we found a potential heading, start a new chunk
                        if potential_heading and len(current_chunk_text) > 0:
                            # Save the previous chunk
                            combined_text = "\n\n".join(current_chunk_text)
                            doc_structure["chunks"].append({
                                "heading": current_heading,
                                "text": combined_text,
                                "start_page": start_page,
                                "end_page": page_num - 1
                            })
                            
                            # Start a new chunk
                            current_chunk_text = [improved_text]
                            current_heading = potential_heading
                            start_page = page_num
                        else:
                            # Continue with current chunk
                            current_chunk_text.append(improved_text)
                        
                        # Create a new chunk every few pages if no natural breaks found
                        if page_num - start_page >= 4 and not potential_heading:
                            combined_text = "\n\n".join(current_chunk_text)
                            doc_structure["chunks"].append({
                                "heading": current_heading,
                                "text": combined_text,
                                "start_page": start_page,
                                "end_page": page_num
                            })
                            
                            # Reset for next chunk
                            current_chunk_text = []
                            current_heading = f"Section starting on page {page_num + 1}"
                            start_page = page_num + 1
                            
                    except Exception as page_error:
                        print(f"Warning: Error processing page {page_num} in fallback method: {page_error}")
                        continue
                
                # Add any remaining text
                if current_chunk_text:
                    combined_text = "\n\n".join(current_chunk_text)
                    doc_structure["chunks"].append({
                        "heading": current_heading,
                        "text": combined_text,
                        "start_page": start_page,
                        "end_page": len(reader.pages)
                    })
            
            print(f"Fallback extraction completed: created {len(doc_structure['chunks'])} chunks")
            return doc_structure
            
        except Exception as fallback_error:
            print(f"Fallback extraction also failed: {fallback_error}")
            # Return empty structure if all methods fail
            return {
                "doc_id": self.doc_id,
                "created_date": self.created_date,
                "country": self.country,
                "source_type": self.source_type,
                "chunks": []
            }

    @staticmethod
    def process_pdfs(
        input_dir,
        output_dir,
        boilerplate_threshold=0.3,
        doc_id=None,
        language="unknown",
        region="unknown",
        title="Unknown Title",
        created_date="unknown_date",
        keywords=None
    ):
        """Process all PDFs in a folder and save extracted JSON data."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        processed_files = 0
        errors = 0
        
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_path = os.path.join(root, file)
                    try:
                        # Create processor and extract content
                        processor = PDFProcessor(
                            pdf_path,
                            boilerplate_threshold=boilerplate_threshold,
                            doc_id=os.path.splitext(file)[0],
                            language=language,
                            region=region,
                            title=title,
                            created_date=created_date,
                            keywords=keywords or []
                        )

                        result = processor.extract_preliminary_chunks()
                        if result and result["chunks"]:
                            # Get exact relative path from input directory
                            rel_path = os.path.relpath(root, input_dir)
                            
                            # Create identical folder structure in output directory
                            output_subdir = os.path.join(output_dir, rel_path)
                            os.makedirs(output_subdir, exist_ok=True)

                            # Output file path with same name as input (plus _cleaned)
                            output_file = os.path.join(
                                output_subdir,
                                f"{os.path.splitext(file)[0]}_cleaned.json"
                            )
                            
                            with open(output_file, "w", encoding="utf-8") as f:
                                json.dump(result, f, indent=2, ensure_ascii=False)
                                
                            processed_files += 1
                            print(f"Successfully processed: {pdf_path} -> {output_file}")
                        else:
                            errors += 1
                            print(f"Warning: No chunks extracted from {pdf_path}")
                    except Exception as e:
                        errors += 1
                        print(f"Error processing {pdf_path}: {e}")
        
        print(f"Processing complete. Successfully processed {processed_files} files with {errors} errors.")


import os
import re
import json
import statistics
import pdfplumber
from collections import defaultdict
import numpy as np  # Add this to the existing imports
import glob


class PostCleaner:
    """
    Advanced class to clean up translation artifacts in processed JSON documents.
    
    This cleaner handles complex cases including:
    1. Numerical pattern repetition (0.09.09.09...)
    2. Dollar sign and other symbol repetition ($$$$$)
    3. Quoted row markers and duplicate rows
    4. Nonsensical word repetitions (agglomeration, agitation...)
    5. Technical phrase repetition (material injury, material...)
    6. Mixed numerical and textual artifacts
    """
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        maintain_folder_structure: bool = True
    ):
        """
        Initialize the translation cleaner.
        
        Args:
            input_dir: Directory containing translated JSON files
            output_dir: Directory to save cleaned files
            maintain_folder_structure: Whether to maintain folder structure when saving
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.maintain_folder_structure = maintain_folder_structure
        
        # Counter for statistics
        self.stats = {
            "files_processed": 0,
            "text_chunks_cleaned": 0,
            "table_chunks_cleaned": 0,
            "artifacts_removed": 0,
            "chinese_chars_removed": 0,
            "excessive_punctuation_fixed": 0,
            "table_rows_fixed": 0,
            "repeated_phrases_removed": 0,
            "repeated_words_fixed": 0,
            "numerical_patterns_fixed": 0,
            "quoted_rows_fixed": 0,
            "symbol_repetition_fixed": 0,
            "special_patterns_fixed": 0
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def clean_all_documents(self):
        """Process all JSON files in the input directory recursively."""
        # Find all JSON files
        json_files = glob.glob(os.path.join(self.input_dir, "**/*.json"), recursive=True)
        print(f"Found {len(json_files)} JSON files to clean.")
        
        for file_path in json_files:
            self.clean_document(file_path)
        
        # Print statistics
        print(f"\nCleaning Complete:")
        print(f"  Files processed: {self.stats['files_processed']}")
        print(f"  Text chunks cleaned: {self.stats['text_chunks_cleaned']}")
        print(f"  Table chunks cleaned: {self.stats['table_chunks_cleaned']}")
        print(f"  Artifacts removed: {self.stats['artifacts_removed']}")
        print(f"  Chinese characters removed: {self.stats['chinese_chars_removed']}")
        print(f"  Excessive punctuation fixed: {self.stats['excessive_punctuation_fixed']}")
        print(f"  Table rows fixed: {self.stats['table_rows_fixed']}")
        print(f"  Repeated phrases removed: {self.stats['repeated_phrases_removed']}")
        print(f"  Repeated words fixed: {self.stats['repeated_words_fixed']}")
        print(f"  Numerical patterns fixed: {self.stats['numerical_patterns_fixed']}")
        print(f"  Quoted rows fixed: {self.stats['quoted_rows_fixed']}")
        print(f"  Symbol repetition fixed: {self.stats['symbol_repetition_fixed']}")
        print(f"  Special patterns fixed: {self.stats['special_patterns_fixed']}")
    
    def clean_document(self, file_path: str):
        """Clean a single JSON document."""
        rel_path = os.path.relpath(file_path, self.input_dir)
        print(f"Cleaning: {rel_path}")
        
        try:
            # Load the document
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Clean the document
            cleaned_data = self._process_document(data)
            
            # Determine output path
            if self.maintain_folder_structure:
                # Create subdirectories if needed
                output_path = os.path.join(self.output_dir, rel_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            else:
                # Flat structure - just use filename
                output_path = os.path.join(self.output_dir, os.path.basename(file_path))
            
            # Save the cleaned document
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
            
            self.stats["files_processed"] += 1
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    def _process_document(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a document by cleaning all its chunks."""
        if not isinstance(data, dict) or "chunks" not in data:
            return data
        
        # Clean each chunk
        for chunk in data["chunks"]:
            # Clean the heading
            if "heading" in chunk:
                chunk["heading"] = self._clean_text(chunk["heading"], is_heading=True)
            
            # Clean the text content
            if "text" in chunk:
                original_text = chunk["text"]
                is_table = self._is_table_chunk(chunk)
                
                if is_table:
                    chunk["text"] = self._clean_table_text(original_text)
                    self.stats["table_chunks_cleaned"] += 1
                else:
                    chunk["text"] = self._clean_text(original_text)
                    self.stats["text_chunks_cleaned"] += 1
                
                # Apply advanced pattern cleaning regardless of chunk type
                chunk["text"] = self._clean_advanced_patterns(chunk["text"])
        
        return data
    
    def _is_table_chunk(self, chunk: Dict[str, Any]) -> bool:
        """Determine if a chunk contains a table."""
        # Check if the heading mentions "table"
        if "heading" in chunk and re.search(r'table', chunk["heading"], re.IGNORECASE):
            return True
        
        # Check if the text contains table-like rows
        if "text" in chunk and re.search(r'Row \d+:', chunk["text"]):
            return True
        
        return False
    
    def _clean_quoted_rows(self, text: str) -> str:
        """
        Clean quoted row markers like "Row 5" "Row 6" "Row 6".
        """
        if not text:
            return text
        
        # Count occurrences before cleaning
        quoted_row_pattern = r'"Row \d+"'
        count_before = len(re.findall(quoted_row_pattern, text))
        
        # Fix consecutive quoted rows (e.g., "Row 6" "Row 6" "Row 7" "Row 7")
        text = re.sub(r'("Row \d+"\s*)(\1)+', r'\1', text)
        
        # Fix rows with too many quotes
        text = re.sub(r'"Row (\d+)"\s+"Row \1"', r'Row \1:', text)
        
        # Count occurrences after cleaning
        count_after = len(re.findall(quoted_row_pattern, text))
        self.stats["quoted_rows_fixed"] += (count_before - count_after)
        
        return text
    
    def _clean_numerical_patterns(self, text: str) -> str:
        """
        Clean numerical pattern repetition like 0.09.09.09.09.09...
        """
        # Find patterns of repeating digits with dots or other separators
        patterns_found = 0
        
        # Find decimal number patterns that repeat (like 0.09.09.09...)
        decimal_repetition = r'(\d+\.\d{1,3})(\.\d{1,3}){3,}'
        matches = re.findall(decimal_repetition, text)
        for match in matches:
            if match and match[0]:
                # Get the first part of the pattern
                base_pattern = match[0]
                # Find the full repeating pattern in the text
                full_pattern = re.escape(base_pattern) + r'(\.\d{1,3}){3,}'
                replacement = base_pattern  # Replace with just the first occurrence
                text = re.sub(full_pattern, replacement, text)
                patterns_found += 1
        
        # Another pattern: repeating decimals like 0.090.090.09...
        decimal_repetition2 = r'(\d+\.\d{2,3})(\d+\.\d{2,3})(\d+\.\d{2,3})+'
        text = re.sub(decimal_repetition2, r'\1', text)
        
        # Yet another pattern: isolated repeating numbers
        repeated_numbers = re.compile(r'(\d{1,3})(\1){3,}')
        text = re.sub(repeated_numbers, r'\1\1', text)
        
        self.stats["numerical_patterns_fixed"] += patterns_found
        return text
    
    def _clean_symbol_repetition(self, text: str) -> str:
        """
        Clean repetitive symbols like $$$$$$$$ or ######## that go beyond normal formatting.
        """
        symbol_patterns = {
            # Repeated $ signs
            r'\${5,}': '$$$',
            # Repeated # signs
            r'#{5,}': '###',
            # Repeated @ signs
            r'@{5,}': '@@@',
            # Repeated + signs
            r'\+{5,}': '+++',
            # Repeated * signs
            r'\*{5,}': '***',
            # Repeated = signs
            r'={5,}': '===',
        }
        
        count = 0
        for pattern, replacement in symbol_patterns.items():
            # Count matches before replacement
            matches = re.findall(pattern, text)
            count += len(matches)
            
            # Replace the repetitions
            text = re.sub(pattern, replacement, text)
        
        self.stats["symbol_repetition_fixed"] += count
        return text
    
    def _clean_special_phrase_repetition(self, text: str) -> str:
        """
        Clean specific phrase repetitions found in examples.
        """
        special_patterns = [
            # Abortion of information/commission repeating
            (r'(Abortion of (?:this information|the Commission)(?:\s+|,)){3,}', r'\1\1'),
            
            # Agglomeration/agitation repeating
            (r'(agglomeration|agitation)(?:\s+\1){3,}', r'\1 \1'),
            
            # Repeated "ag ag ag" sequences
            (r'(ag\s+){3,}', r'ag ag '),
            
            # material injury repetition
            (r'((?:material |)injury,?\s+){5,}', r'material injury, '),
            
            # "material, material, material" repetition
            (r'(material,?\s+){3,}', r'material, material '),
            
            # "etc, etc, etc" repetition
            (r'(etc(?:,|\.)?\s*){3,}', r'etc., etc.'),
            
            # "of the of the of the" repetition
            (r'(of the\s+){3,}', r'of the '),
            
            # "Row: Row: Row:" repetition
            (r'(Row:\s*){3,}', r'Row: '),
            
            # progressively/progressionlessly/progressiveness repeating
            (r'(progress(?:ion|ively|iveness)(?:\s+|,)){3,}', r'\1\1'),
        ]
        
        count = 0
        for pattern, replacement in special_patterns:
            # Count matches before replacement
            matches = len(re.findall(pattern, text))
            count += matches
            
            # Replace the repetitions
            text = re.sub(pattern, replacement, text)
        
        self.stats["special_patterns_fixed"] += count
        return text
    
    def _clean_repeating_row_markers(self, text: str) -> str:
        """
        Clean repetitive row markers, especially in tables.
        """
        # Fix sequences of repeating "Row X:" or "Row: Row: Row:"
        row_fixes = 0
        
        # Fix "Row X: Row X: Row X:" patterns
        row_pattern = r'(Row \d+:)\s*\1+'
        row_fixes += len(re.findall(row_pattern, text))
        text = re.sub(row_pattern, r'\1', text)
        
        # Fix "Row: Row: Row:" patterns
        row_colon_pattern = r'(Row:)\s*\1+'
        row_fixes += len(re.findall(row_colon_pattern, text))
        text = re.sub(row_colon_pattern, r'\1', text)
        
        # Fix "Row: Row: Row: Row: Row:" patterns without numbers
        row_pattern2 = r'(Row:\s+){3,}'
        row_fixes += len(re.findall(row_pattern2, text))
        text = re.sub(row_pattern2, r'Row: ', text)
        
        # Fix sequences with numbers like "Row: 27: Row:"
        row_pattern3 = r'Row:\s*\d+:\s*Row:'
        row_fixes += len(re.findall(row_pattern3, text))
        text = re.sub(row_pattern3, r'Row:', text)
        
        self.stats["table_rows_fixed"] += row_fixes
        return text
    
    def _clean_advanced_patterns(self, text: str) -> str:
        """
        Apply advanced pattern cleaning that works on all document types.
        """
        # Save original length for artifact counting
        original_length = len(text)
        
        # Apply all advanced cleaning methods
        text = self._clean_numerical_patterns(text)
        text = self._clean_symbol_repetition(text)
        text = self._clean_quoted_rows(text)
        text = self._clean_special_phrase_repetition(text)
        text = self._clean_repeating_row_markers(text)
        
        # Specialized pattern for the examples you provided:
        # Pattern with "0.09.09.09..." repeating (from the Discussion section)
        text = re.sub(r'0\.09\.09\.09\.09\.09\.09\.09\.09\.09\.09\.09\.09\.09\.09\.09\.09\.09\.09[\.09]*', 
                      r'0.09', text)
        
        # Count artifacts removed
        artifacts_removed = original_length - len(text)
        if artifacts_removed > 0:
            self.stats["artifacts_removed"] += artifacts_removed
        
        return text
    
    def _find_repeated_phrases(self, text: str, min_length: int = 3, max_length: int = 30) -> List[tuple]:
        """Find repeated phrases in text."""
        words = text.split()
        if len(words) < min_length * 2:  # Need at least 2 occurrences to find repetition
            return []
        
        # Try different phrase lengths
        repeated_phrases = []
        
        for phrase_len in range(min_length, min(max_length, len(words) // 2 + 1)):
            # Check each possible phrase of this length
            for i in range(len(words) - phrase_len + 1):
                phrase = ' '.join(words[i:i+phrase_len])
                
                # Count occurrences
                count = 0
                for j in range(i + phrase_len, len(words) - phrase_len + 1, phrase_len):
                    if ' '.join(words[j:j+phrase_len]) == phrase:
                        count += 1
                    else:
                        break
                
                # If phrase repeats, add it to our list
                if count > 0:
                    repeated_phrases.append((phrase, count + 1))
                    # Skip ahead to avoid finding sub-phrases of this repetition
                    i += (count + 1) * phrase_len - 1
        
        return repeated_phrases
    
    def _remove_repeated_phrases(self, text: str) -> str:
        """Remove repeated phrases, keeping just one instance."""
        if not text:
            return text
        
        # Find repeated phrases
        repeated_phrases = self._find_repeated_phrases(text)
        count = 0
        
        for phrase, occurrences in repeated_phrases:
            # Create pattern that matches exactly this phrase repeated multiple times
            pattern = re.escape(phrase) + r'(?:\s+' + re.escape(phrase) + r')+'
            
            # Replace with single instance
            new_text = re.sub(pattern, phrase, text)
            
            # Update count if replacement occurred
            if new_text != text:
                count += 1
                text = new_text
        
        self.stats["repeated_phrases_removed"] += count
        return text
    
    def _remove_repeated_single_words(self, text: str) -> str:
        """Remove long runs of the same word (e.g., 'no, no, no, no...')."""
        if not text:
            return text
        
        # Pattern for repeated words with optional punctuation
        pattern = r'\b(\w+(?:[,.;:]? |, |\. ))\1{2,}'
        
        # Find all matches
        matches = re.findall(pattern, text)
        
        # Replace each match with just two instances (e.g., "no, no")
        for match in matches:
            repeat_pattern = re.escape(match) + r'{3,}'  # 3+ occurrences
            text = re.sub(repeat_pattern, match + match, text)
            self.stats["repeated_words_fixed"] += 1
        
        return text
    
    def _clean_text(self, text: str, is_heading: bool = False) -> str:
        """
        Clean general text content.
        This enhanced version handles complex patterns.
        """
        if not text:
            return text
        
        original_length = len(text)
        
        # Step 1: Handle basic patterns
        
        # Remove Chinese/Japanese/Korean characters
        chinese_char_count = len(re.findall(r'[一-龥的]', text))
        text = re.sub(r'[一-龥的]', '', text)
        self.stats["chinese_chars_removed"] += chinese_char_count
        
        # Handle repeated ellipses and dots
        text = re.sub(r'\.{5,}', '...', text)  # Replace long runs of dots with ellipsis
        
        # Fix excessive repetitions of common words
        text = re.sub(r'(?:no,? ){3,}', 'no, no ', text)
        text = re.sub(r'(?:yes,? ){3,}', 'yes, yes ', text)
        text = re.sub(r'(?:not ){3,}', 'not not ', text)
        
        # Remove excessive punctuation
        punct_matches = len(re.findall(r'([!?.:;,\-_=\*\+#&\|\[\]\{\}\(\)<>])\1{3,}', text))
        text = re.sub(r'([!?.:;,\-_=\*\+#&\|\[\]\{\}\(\)<>])\1{3,}', r'\1', text)
        self.stats["excessive_punctuation_fixed"] += punct_matches
        
        # Remove excessive letter repetitions
        text = re.sub(r'([a-zA-Z])\1{3,}', r'\1', text)
        
        # Step 2: Handle repeated phrases
        if len(text.split()) > 5:  # Only for longer texts
            text = self._remove_repeated_phrases(text)
            text = self._remove_repeated_single_words(text)
        
        # Step 3: Final formatting adjustments
        
        # Fix spacing issues
        text = re.sub(r'\s+', ' ', text)
        
        # For headings only, apply specific rules
        if is_heading:
            # Remove table markers in headings
            text = re.sub(r'table underheading', 'table heading', text)
            
            # Clean up angle brackets in headings
            text = re.sub(r'<([^<>]*)>', r'\1', text)
            
            # Remove row markers in headings
            text = re.sub(r'Row \d+:\s*', '', text)
        
        # Update artifact counter
        artifacts_removed = original_length - len(text)
        if artifacts_removed > 0:
            self.stats["artifacts_removed"] += artifacts_removed
        
        return text.strip()
    
    def _clean_table_text(self, text: str) -> str:
        """
        Clean text specifically for table content.
        Enhanced to handle complex patterns in tables.
        """
        if not text:
            return text
        
        original_length = len(text)
        
        # Step 1: First run special patterns for tables
        
        # Fix duplicate row labels (Row 1:Row 1:Row 1:)
        row_fixes = 0
        row_fixes += len(re.findall(r'(Row \d+:)\s*\1+', text))
        text = re.sub(r'(Row \d+:)\s*\1+', r'\1', text)
        
        # Fix "Low N" to "Row N"
        row_fixes += len(re.findall(r'Low (\d+)', text))
        text = re.sub(r'Low (\d+)', r'Row \1:', text)
        
        # Fix row label format
        row_fixes += len(re.findall(r'Row (\d+)!!+', text))
        text = re.sub(r'Row (\d+)!!+', r'Row \1:', text)
        
        row_fixes += len(re.findall(r'Row (\d+):的', text))
        text = re.sub(r'Row (\d+):的', r'Row \1:', text)
        
        # Fix consecutive empty row numbers
        text = re.sub(r'(Row \d+:\s*\n\s*){3,}', r'Row 1:\n', text)
        
        self.stats["table_rows_fixed"] += row_fixes
        
        # Apply advanced pattern cleaning
        text = self._clean_advanced_patterns(text)
        
        # Clean repeated content within rows
        parts = re.split(r'(Row \d+:)', text)
        result_parts = []
        
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Even parts are content between "Row N:" markers
                if part.strip():
                    # Clean the content of this row
                    cleaned_part = self._clean_text(part)  # Use standard text cleaning
                    if len(part.split()) > 5:  # Only for longer content
                        # Try to fix repeated phrases in this row
                        cleaned_part = self._remove_repeated_phrases(cleaned_part)
                    result_parts.append(cleaned_part)
                else:
                    result_parts.append(part)
            else:  # Odd parts are "Row N:" markers
                result_parts.append(part)
        
        text = ''.join(result_parts)
        
        # Special handling for common table artifacts
        
        # Repeated "Indication of AMM & " pattern
        text = re.sub(r'(Indication of AMM &\s+)+', r'Indication of AMM & ', text)
        
        # Remove consecutive duplicate items in comma-separated lists
        text = re.sub(r'([^,]+, )(\1)+', r'\1', text)
        
        # Fix row formatting
        
        # Remove empty rows
        text = re.sub(r'Row \d+:\s*(\n|$)', '', text)
        
        # Fix spacing issues
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Make sure rows are on new lines
        text = re.sub(r'(Row \d+:)(?!\n)', r'\1\n', text)
        
        # Update artifact counter
        artifacts_removed = original_length - len(text)
        if artifacts_removed > 0:
            self.stats["artifacts_removed"] += artifacts_removed
        
        return text.strip()