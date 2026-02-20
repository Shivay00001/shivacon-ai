"""
SHIVACON AI - Document Generation System
========================================

Generate various document types:
- Reports
- Summaries
- Articles
- Technical Docs
- Business Letters
- Presentations
- JSON/CSV/HTML/Markdown
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch


class DocumentType(Enum):
    """Types of documents that can be generated."""
    REPORT = "report"
    SUMMARY = "summary"
    ARTICLE = "article"
    TECHNICAL_DOC = "technical_doc"
    BUSINESS_LETTER = "business_letter"
    PRESENTATION = "presentation"
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    MARKDOWN = "markdown"
    EMAIL = "email"
    NOTE = "note"


class DocumentStyle(Enum):
    """Writing styles."""
    FORMAL = "formal"
    INFORMAL = "informal"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    ACADEMIC = "academic"
    BUSINESS = "business"
    CONCISE = "concise"
    PROFESSIONAL = "professional"


@dataclass
class DocumentConfig:
    """Configuration for document generation."""
    doc_type: DocumentType = DocumentType.REPORT
    style: DocumentStyle = DocumentStyle.FORMAL
    title: str = ""
    author: str = "Shivacon AI"
    date: Optional[str] = None
    length: str = "medium"  # short, medium, long
    include_toc: bool = True
    include_metadata: bool = True
    language: str = "en"


@dataclass
class DocumentSection:
    """A section of a document."""
    title: str
    content: str
    level: int = 1


@dataclass
class GeneratedDocument:
    """Generated document with metadata."""
    title: str
    content: str
    doc_type: DocumentType
    style: DocumentStyle
    sections: List[DocumentSection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    word_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    def save(self, path: Union[str, Path], format: str = "txt"):
        """Save document to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "txt":
            path.write_text(self.content, encoding="utf-8")
        elif format == "md":
            path.write_text(self.content, encoding="utf-8")
        elif format == "html":
            path.write_text(self.to_html(), encoding="utf-8")
        elif format == "json":
            path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")
        
        return path
    
    def to_html(self) -> str:
        """Convert to HTML."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{self.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; border-bottom: 2px solid #333; }}
        h2 {{ color: #666; }}
        .metadata {{ color: #999; font-size: 0.9em; }}
        .content {{ line-height: 1.6; }}
    </style>
</head>
<body>
    <h1>{self.title}</h1>
    <div class="metadata">
        <p>Author: {self.metadata.get('author', 'Unknown')}</p>
        <p>Date: {self.created_at.strftime('%Y-%m-%d')}</p>
        <p>Words: {self.word_count}</p>
    </div>
    <div class="content">
        {self.content}
    </div>
</body>
</html>"""
        return html
    
    def to_markdown(self) -> str:
        """Convert to Markdown."""
        md = f"# {self.title}\n\n"
        md += f"**Author:** {self.metadata.get('author', 'Unknown')}\n"
        md += f"**Date:** {self.created_at.strftime('%Y-%m-%d')}\n"
        md += f"**Words:** {self.word_count}\n\n---\n\n"
        md += self.content
        return md


class DocumentGenerator:
    """
    Generate various types of documents.
    
    Features:
    - Multiple document types
    - Customizable styles
    - Template-based generation
    - Metadata support
    - Export to multiple formats
    """
    
    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Dict]:
        """Load document templates."""
        return {
            DocumentType.REPORT: {
                "sections": ["Executive Summary", "Introduction", "Findings", "Analysis", "Recommendations", "Conclusion"],
                "format": "formal"
            },
            DocumentType.SUMMARY: {
                "sections": ["Overview", "Key Points", "Details", "Conclusion"],
                "format": "concise"
            },
            DocumentType.ARTICLE: {
                "sections": ["Title", "Introduction", "Body", "Conclusion", "References"],
                "format": "narrative"
            },
            DocumentType.TECHNICAL_DOC: {
                "sections": ["Overview", "Requirements", "Implementation", "API Reference", "Examples", "Troubleshooting"],
                "format": "technical"
            },
            DocumentType.BUSINESS_LETTER: {
                "sections": ["Header", "Address", "Salutation", "Body", "Closing", "Signature"],
                "format": "formal"
            },
            DocumentType.EMAIL: {
                "sections": ["Subject", "Greeting", "Message", "Closing"],
                "format": "professional"
            },
            DocumentType.NOTE: {
                "sections": ["Title", "Content"],
                "format": "casual"
            },
            DocumentType.PRESENTATION: {
                "sections": ["Slide Title", "Bullet Points", "Summary"],
                "format": "bullet"
            },
        }
    
    def generate(
        self,
        topic: str,
        config: Optional[DocumentConfig] = None,
        content_data: Optional[Dict] = None,
    ) -> GeneratedDocument:
        """Generate a document based on topic and config."""
        
        config = config or DocumentConfig()
        config.title = config.title or self._generate_title(topic, config.doc_type)
        
        # Get template
        template = self.templates.get(config.doc_type, {})
        sections_def = template.get("sections", ["Introduction", "Main Content", "Conclusion"])
        
        # Generate sections
        sections = []
        
        if config.doc_type == DocumentType.JSON:
            content = self._generate_json(topic, content_data or {})
        elif config.doc_type == DocumentType.CSV:
            content = self._generate_csv(topic, content_data or {})
        elif config.doc_type == DocumentType.MARKDOWN:
            content = self._generate_markdown(topic, sections_def, config)
        elif config.doc_type == DocumentType.HTML:
            content = self._generate_html(topic, config)
        else:
            content = self._generate_text_document(topic, sections_def, config)
        
        # Calculate word count
        word_count = len(content.split())
        
        # Create document
        doc = GeneratedDocument(
            title=config.title,
            content=content,
            doc_type=config.doc_type,
            style=config.style,
            sections=sections,
            metadata={
                "author": config.author,
                "topic": topic,
                "length": config.length,
                "language": config.language,
            },
            word_count=word_count,
        )
        
        return doc
    
    def _generate_title(self, topic: str, doc_type: DocumentType) -> str:
        """Generate document title."""
        prefixes = {
            DocumentType.REPORT: "Report on",
            DocumentType.SUMMARY: "Summary of",
            DocumentType.ARTICLE: "Article about",
            DocumentType.TECHNICAL_DOC: "Technical Documentation:",
            DocumentType.BUSINESS_LETTER: "Business Letter regarding",
            DocumentType.EMAIL: "Email:",
            DocumentType.NOTE: "Notes on",
            DocumentType.PRESENTATION: "Presentation:",
        }
        prefix = prefixes.get(doc_type, "")
        return f"{prefix} {topic}".strip()
    
    def _generate_text_document(
        self,
        topic: str,
        sections_def: List[str],
        config: DocumentConfig,
    ) -> str:
        """Generate a text-based document."""
        
        content_parts = []
        
        # Title
        content_parts.append(f"# {config.title}\n")
        
        # Metadata
        if config.include_metadata:
            content_parts.append(f"**Author:** {config.author}")
            content_parts.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
            content_parts.append("")
        
        # Length multiplier
        length_map = {"short": 1, "medium": 2, "long": 3}
        multiplier = length_map.get(config.length, 2)
        
        # Generate sections
        for i, section_title in enumerate(sections_def):
            content_parts.append(f"\n## {section_title}\n")
            
            # Generate section content based on topic
            section_content = self._generate_section_content(
                topic=topic,
                section=section_title,
                style=config.style,
                length=multiplier,
            )
            content_parts.append(section_content)
        
        return "\n".join(content_parts)
    
    def _generate_section_content(
        self,
        topic: str,
        section: str,
        style: DocumentStyle,
        length: int,
    ) -> str:
        """Generate content for a section."""
        
        # Template content based on section name
        section_lower = section.lower()
        
        if "summary" in section_lower or "overview" in section_lower:
            templates = [
                f"This document provides a comprehensive overview of {topic}. "
                f"It covers the key aspects and main points related to the subject matter.",
                f"Below is a summary of the main findings and conclusions regarding {topic}.",
            ]
        elif "introduction" in section_lower:
            templates = [
                f"{topic} is an important subject that requires careful analysis. "
                f"This document examines the various aspects and implications.",
                f"In this document, we explore the fundamentals and details of {topic}.",
            ]
        elif "conclusion" in section_lower:
            templates = [
                f"In conclusion, {topic} represents a significant area of study. "
                f"Further research and exploration are recommended.",
                f"To summarize, the key findings point to the importance of {topic} "
                f"in the broader context.",
            ]
        elif "recommendation" in section_lower:
            templates = [
                f"Based on the analysis, we recommend further investigation into {topic}. "
                f"Additional resources should be allocated for continued study.",
                f"It is recommended that stakeholders consider the implications of {topic} "
                f"when making decisions.",
            ]
        elif "analysis" in section_lower or "finding" in section_lower:
            templates = [
                f"The analysis reveals several important aspects of {topic}. "
                f"Key observations include multiple dimensions of the subject.",
                f"Our analysis of {topic} has identified several critical factors "
                f"that warrant attention.",
            ]
        else:
            templates = [
                f"This section covers {topic} in detail. "
                f"The information presented is based on comprehensive research.",
                f"Regarding {topic}, we present the following information "
                f"for consideration.",
            ]
        
        # Select template and repeat for length
        template = templates[hash(topic) % len(templates)]
        
        # Repeat content for longer documents
        if length > 1:
            template += " " + " ".join([f"Additional details related to {topic}."] * (length - 1))
        
        # Adjust based on style
        if style == DocumentStyle.INFORMAL:
            template = template.replace("comprehensive", "thorough").replace("investigation", "look")
        elif style == DocumentStyle.TECHNICAL:
            template += " Technical specifications and parameters apply."
        
        return template
    
    def _generate_json(self, topic: str, data: Dict) -> str:
        """Generate JSON document."""
        
        json_data = {
            "title": topic,
            "generated_by": "Shivacon AI",
            "timestamp": datetime.now().isoformat(),
            "data": data or {
                "category": "general",
                "items": [
                    {"id": 1, "name": f"{topic} Item 1", "value": 100},
                    {"id": 2, "name": f"{topic} Item 2", "value": 200},
                    {"id": 3, "name": f"{topic} Item 3", "value": 300},
                ]
            },
            "metadata": {
                "version": "1.0",
                "format": "json"
            }
        }
        
        return json.dumps(json_data, indent=2)
    
    def _generate_csv(self, topic: str, data: Dict) -> str:
        """Generate CSV document."""
        
        csv_lines = ["ID,Name,Value,Description"]
        
        items = data.get("items", [
            {"id": 1, "name": f"{topic} - Item 1", "value": 100, "desc": "First item"},
            {"id": 2, "name": f"{topic} - Item 2", "value": 200, "desc": "Second item"},
            {"id": 3, "name": f"{topic} - Item 3", "value": 300, "desc": "Third item"},
        ])
        
        for item in items:
            csv_lines.append(f'{item.get("id", 0)},{item.get("name", "")},{item.get("value", 0)},{item.get("desc", "")}')
        
        return "\n".join(csv_lines)
    
    def _generate_markdown(self, topic: str, sections: List[str], config: DocumentConfig) -> str:
        """Generate Markdown document."""
        
        md = f"# {config.title}\n\n"
        md += f"*Generated by Shivacon AI on {datetime.now().strftime('%Y-%m-%d')}*\n\n"
        
        if config.include_toc:
            md += "## Table of Contents\n\n"
            for i, section in enumerate(sections):
                md += f"{i+1}. [{section}](#{section.lower().replace(' ', '-')})\n"
            md += "\n---\n\n"
        
        for section in sections:
            md += f"## {section}\n\n"
            md += f"Content related to {topic} goes here.\n\n"
        
        return md
    
    def _generate_html(self, topic: str, config: DocumentConfig) -> str:
        """Generate HTML document."""
        
        html = f"""<!DOCTYPE html>
<html lang="{config.language}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{config.title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .meta {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 20px;
        }}
        .content {{
            text-align: justify;
        }}
    </style>
</head>
<body>
    <h1>{config.title}</h1>
    <div class="meta">
        <span>Author: {config.author}</span> |
        <span>Date: {datetime.now().strftime('%Y-%m-%d')}</span> |
        <span>Generated by: Shivacon AI</span>
    </div>
    <div class="content">
        <h2>Overview</h2>
        <p>This document provides information about {topic}.</p>
        
        <h2>Details</h2>
        <p>The following sections contain detailed information regarding {topic}.</p>
        
        <h2>Conclusion</h2>
        <p>This concludes the document on {topic}.</p>
    </div>
</body>
</html>"""
        return html
    
    def generate_report(
        self,
        title: str,
        data: Dict[str, Any],
        sections: Optional[List[str]] = None,
    ) -> GeneratedDocument:
        """Generate a formal report."""
        
        sections = sections or ["Executive Summary", "Introduction", "Findings", "Recommendations", "Conclusion"]
        
        config = DocumentConfig(
            doc_type=DocumentType.REPORT,
            title=title,
            style=DocumentStyle.FORMAL,
        )
        
        content_parts = [f"# {title}\n"]
        content_parts.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n")
        content_parts.append(f"**Generated by:** Shivacon AI\n\n---\n")
        
        for section in sections:
            content_parts.append(f"\n## {section}\n")
            
            if section == "Executive Summary":
                content_parts.append(f"This report presents an analysis of the provided data. ")
                content_parts.append(f"Key findings are summarized below.\n")
            elif section == "Findings":
                for key, value in data.items():
                    content_parts.append(f"- **{key}:** {value}\n")
            elif section == "Recommendations":
                content_parts.append(f"Based on the analysis, the following recommendations are made:\n")
                content_parts.append(f"1. Review the findings carefully\n")
                content_parts.append(f"2. Consider implementation of suggested improvements\n")
                content_parts.append(f"3. Schedule follow-up review\n")
        
        return GeneratedDocument(
            title=title,
            content="\n".join(content_parts),
            doc_type=DocumentType.REPORT,
            style=DocumentStyle.FORMAL,
            metadata={"data_keys": list(data.keys())},
        )
    
    def generate_summary(
        self,
        text: str,
        max_length: int = 200,
    ) -> GeneratedDocument:
        """Generate a summary of given text."""
        
        # Simple extraction-based summarization
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Take first few sentences for summary
        summary_sentences = sentences[:3]
        summary = ". ".join(summary_sentences) + "."
        
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        
        return GeneratedDocument(
            title="Summary",
            content=summary,
            doc_type=DocumentType.SUMMARY,
            style=DocumentStyle.CONCISE,
            metadata={"source_length": len(text), "summary_length": len(summary)},
        )


# Factory function
def create_document_generator(model=None, tokenizer=None) -> DocumentGenerator:
    """Create a document generator."""
    return DocumentGenerator(model=model, tokenizer=tokenizer)


# Example usage
if __name__ == "__main__":
    print("""
    SHIVACON AI - Document Generation System
    ========================================
    
    Usage:
    
    from utils.document_generator import DocumentGenerator, DocumentConfig, DocumentType
    
    # Create generator
    gen = DocumentGenerator()
    
    # Generate a report
    doc = gen.generate(
        topic="Q4 Sales Analysis",
        config=DocumentConfig(
            doc_type=DocumentType.REPORT,
            style=DocumentStyle.BUSINESS,
            length="long"
        )
    )
    
    # Save to file
    doc.save("report.md", format="md")
    
    # Generate JSON
    doc = gen.generate(
        topic="Product Data",
        config=DocumentConfig(doc_type=DocumentType.JSON)
    )
    doc.save("data.json")
    
    # Generate summary
    summary = gen.generate_summary(long_text)
    print(summary.content)
    """)
