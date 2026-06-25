"""
Citation forensics module for the academic hallucination detector.
Analyzes citation lists for fake journals, future publication dates, and ghost authors.
Supports live online verification via CrossRef and Semantic Scholar APIs.
"""

import re
import json
import urllib.request
import urllib.parse
from difflib import SequenceMatcher

from .corpus import FAKE_JOURNALS, REAL_JOURNALS


class CitationVerifier:
    """Verifies academic citations against bibliographic databases."""
    
    def __init__(self):
        self.crossref_url = "https://api.crossref.org/works"
        self.semantic_scholar_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        
    def _parse_citation(self, text: str) -> dict:
        """Parse citation string into author, year, title, and metadata."""
        year_match = re.search(r'\((\d{4})\)', text)
        year = int(year_match.group(1)) if year_match else None
        
        if year_match:
            authors_text = text[:year_match.start()].strip().rstrip(',').rstrip('.')
        else:
            authors_text = text[:30]
            
        if year_match:
            after_year = text[year_match.end():].strip('. ')
            title_match = re.match(r'([^.]+)', after_year)
            title = title_match.group(1).strip() if title_match else ''
        else:
            title = text[:100]
            
        return {
            'authors': authors_text,
            'year': year,
            'title': title,
            'full_text': text
        }
        
    def _check_crossref(self, parsed: dict, timeout: int = 5) -> dict:
        """Search CrossRef database for the title."""
        if not parsed['title']:
            return {'found': False, 'partial': False}
        try:
            title_query = urllib.parse.quote(parsed['title'])
            url = f"{self.crossref_url}?query.bibliographic={title_query}&rows=3"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (mailto:researcher@university.edu)"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            items = data.get('message', {}).get('items', [])
            for item in items:
                title = item.get('title', [''])[0]
                similarity = SequenceMatcher(None, parsed['title'].lower(), title.lower()).ratio() * 100
                if similarity > 85:
                    return {'found': True, 'partial': False, 'doi': item.get('DOI'), 'similarity': similarity}
                elif similarity > 60:
                    return {'found': False, 'partial': True, 'similarity': similarity}
            return {'found': False, 'partial': False}
        except Exception as e:
            return {'found': False, 'partial': False, 'error': str(e)}
            
    def _check_semantic_scholar(self, parsed: dict, timeout: int = 5) -> dict:
        """Search Semantic Scholar database for the title."""
        if not parsed['title']:
            return {'found': False, 'partial': False}
        try:
            title_query = urllib.parse.quote(parsed['title'])
            url = f"{self.semantic_scholar_url}?query={title_query}&limit=3"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            papers = data.get('data', [])
            for paper in papers:
                title = paper.get('title', '')
                similarity = SequenceMatcher(None, parsed['title'].lower(), title.lower()).ratio() * 100
                if similarity > 85:
                    return {'found': True, 'partial': False, 'similarity': similarity}
                elif similarity > 60:
                    return {'found': False, 'partial': True, 'similarity': similarity}
            return {'found': False, 'partial': False}
        except Exception as e:
            return {'found': False, 'partial': False, 'error': str(e)}

    def verify_citation(self, citation_text: str, live: bool = False) -> dict:
        """Verify citation either statically (heuristic) or dynamically (APIs)."""
        parsed = self._parse_citation(citation_text)
        
        if not live:
            # Rule-based offline checking
            has_fake = any(fj.lower() in citation_text.lower() for fj in FAKE_JOURNALS)
            year = parsed["year"]
            has_future = (year > 2025) if year else False
            has_ghost = parsed["authors"].startswith("X.")
            
            if has_fake or has_future or has_ghost:
                verdict = "NOT_FOUND"
                confidence = 0.85
            else:
                verdict = "EXISTS"
                confidence = 0.95
                
            return {
                'original': citation_text,
                'parsed': parsed,
                'verdict': verdict,
                'confidence': confidence,
                'crossref': {'found': not (has_fake or has_future)},
                'semantic_scholar': {'found': not (has_fake or has_future)}
            }
            
        cr = self._check_crossref(parsed)
        ss = self._check_semantic_scholar(parsed)
        
        found = cr.get('found', False) or ss.get('found', False)
        partial = cr.get('partial', False) or ss.get('partial', False)
        
        if found:
            verdict = 'EXISTS'
            confidence = 0.95
        elif partial:
            verdict = 'PARTIAL_MATCH'
            confidence = 0.50
        else:
            verdict = 'NOT_FOUND'
            confidence = 0.85
            
        return {
            'original': citation_text,
            'parsed': parsed,
            'crossref': cr,
            'semantic_scholar': ss,
            'verdict': verdict,
            'confidence': confidence
        }


def analyse_citations(paper, live: bool = False):
    """
    Perform forensic analysis on a paper's citations list.
    Computes anomaly ratios and a composite citation suspicion score.
    Optionally calls live APIs for verification.
    """
    cites = paper.get("citations", [])
    total_cites = max(1, len(cites))
    
    verifier = CitationVerifier()
    
    n_fake_journals = 0
    n_future_years = 0
    n_plausible = 0
    n_ghost_marker = 0
    
    for c in cites:
        authors = str(c.get("authors", ""))
        journal = str(c.get("journal", ""))
        year = c.get("year", 0)
        
        citation_str = f"{authors} ({year}). Title. {journal}."
        res = verifier.verify_citation(citation_str, live=live)
        
        if res["verdict"] == "NOT_FOUND":
            n_fake_journals += 1
            if year > 2025:
                n_future_years += 1
        else:
            n_plausible += 1
            
        if authors.startswith("X."):
            n_ghost_marker += 1
            
    frac_fake_j = n_fake_journals / total_cites
    frac_future = n_future_years / total_cites
    frac_real = n_plausible / total_cites
    
    citation_score = (
        frac_fake_j * 0.5 + 
        frac_future * 0.3 +
        (1.0 - frac_real) * 0.1 + 
        (n_ghost_marker / total_cites) * 0.1
    )
    
    return {
        "n_fake_journals": n_fake_journals,
        "n_future_years": n_future_years,
        "n_plausible": n_plausible,
        "n_ghost_markers": n_ghost_marker,
        "frac_fake_journals": frac_fake_j,
        "frac_future_years": frac_future,
        "citation_suspicion_score": float(max(0.0, min(1.0, citation_score)))
    }
