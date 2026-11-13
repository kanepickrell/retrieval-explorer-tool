"""
Query-intent analysis, kept from the ProtoGraph unified search router (v3); graph_search uses analyze_query_intent.
Intelligently routes queries to the best search strategy (keyword vs neural vs hybrid)

Fixes in v3:
- Short/specific queries (1-2 words like names, IDs) forced to keyword search
- Expanded keyword field search (owner, author, created_by, metadata, tags, etc.)
- Better handling of proper nouns and identifiers
- Improved relevance scoring for keyword matches
"""

import re

from typing import List, Dict, Any
from pydantic import BaseModel
from enum import Enum


class SearchStrategy(str, Enum):
    KEYWORD = "keyword"
    NEURAL = "neural"
    HYBRID = "hybrid"
    VECTOR = "vector"  # dense retrieval; served by graph_search.py (GraphDB backend)


class QueryIntent(BaseModel):
    """Analyzed intent from user query"""
    strategy: SearchStrategy
    confidence: float
    signals: List[str]
    extracted_filters: Dict[str, Any]
    search_terms: List[str]
    is_low_quality: bool = False


# =============================================================================
# QUERY QUALITY & INTENT ANALYSIS
# =============================================================================

# Garbage/low-quality query patterns
GARBAGE_PATTERNS = [
    r'^[^a-zA-Z0-9]*$',  # Only special characters
    r'^.{1,2}$',  # 1-2 characters only
    r'^(the|a|an|is|are|was|were|be|been|being)$',  # Single stop word
    r'^\?+$',  # Only question marks
    r'^\.+$',  # Only periods
]

# Stop words to filter from search terms
STOP_WORDS = {
    'show', 'me', 'all', 'the', 'a', 'an', 'find', 'get', 'list',
    'how', 'do', 'does', 'can', 'to', 'what', 'is', 'are', 'why',
    'which', 'that', 'this', 'for', 'with', 'and', 'or', 'in', 'on',
    'my', 'our', 'we', 'i', 'you', 'about', 'from', 'related', 'of',
    'it', 'its', 'they', 'them', 'their', 'be', 'have', 'has', 'had',
    'will', 'would', 'could', 'should', 'may', 'might', 'must',
    'not', 'no', 'yes', 'if', 'then', 'else', 'when', 'where',
    'there', 'here', 'some', 'any', 'each', 'every', 'both',
    'few', 'more', 'most', 'other', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'under', 'again',
    'further', 'once', 'just', 'also', 'only', 'own', 'same', 'than',
    'too', 'very', 'now', 'did', 'use', 'used', 'using'
}

# Patterns that suggest keyword/exact search
KEYWORD_PATTERNS = [
    r'^"[^"]+"$',  # Quoted exact phrase
    r'\b(lib_|script_|vm_|ttp_)\w+',  # Artifact key patterns
    r'\b[A-Z]{2,}\d+[-_]\d+',  # IDs like OBAP-2024, TTP-001
    r'\.(py|ps1|sh|yml|yaml|json|robot)$',  # File extensions
    r'^show\s+(me\s+)?(all\s+)?',  # "show me all X"
    r'^list\s+',  # "list X"
    r'^find\s+(the\s+)?',  # "find the X"
    r'^get\s+',  # "get X"
]

# Patterns that suggest neural/semantic search
NEURAL_PATTERNS = [
    r'^how\s+(do|does|can|to)\s+',  # "how do we..."
    r'^what\s+(is|are|does|do)\s+',  # "what is..."
    r'^why\s+(do|does|is|are)\s+',  # "why do we..."
    r'^which\s+',  # "which techniques..."
    r'\b(techniques?|methods?|approaches?|ways?|alternatives?)\b',  # Conceptual words
    r'\b(similar|related|like|associated)\b',  # Similarity queries
    r'\b(best|recommended|common|typical)\b',  # Qualitative queries
    r'\?$',  # Questions ending with ?
]

# Cluster/filter keywords
CLUSTER_KEYWORDS = {
    'opfor': ['opfor', 'red team', 'redteam', 'adversary', 'attack', 'attacker'],
    'automation': ['automation', 'automate', 'script', 'orchestration', 'ansible'],
    'range': ['range', 'infrastructure', 'vm', 'network', 'deployment', 'deploy'],
    'content': ['content', 'contentdev', 'handbook', 'documentation', 'intel', 'doc'],
}

# Type filter keywords
TYPE_KEYWORDS = {
    'script': ['script', 'scripts', 'python', 'powershell', 'bash', '.py', '.ps1', '.sh'],
    'playbook': ['playbook', 'playbooks', 'ansible', '.yml', '.yaml'],
    'handbook': ['handbook', 'handbooks', 'documentation', 'docs', 'guide'],
    'ttp': ['ttp', 'technique', 'tactic', 'mitre', 'attack'],
    'vm': ['vm', 'virtual machine', 'server', 'host', 'machine'],
}


def is_garbage_query(query: str) -> bool:
    """Check if query is too low-quality to search"""
    query_stripped = query.strip()
    
    # Check against garbage patterns
    for pattern in GARBAGE_PATTERNS:
        if re.match(pattern, query_stripped, re.IGNORECASE):
            return True
    
    # Extract meaningful words
    words = re.findall(r'\b[a-zA-Z0-9_]+\b', query_stripped.lower())
    meaningful_words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    
    # If no meaningful words, it's garbage
    if len(meaningful_words) == 0:
        return True
    
    return False


def looks_like_identifier_or_name(query: str) -> bool:
    """
    Check if query looks like a specific identifier, name, or proper noun.
    These should use keyword search, not neural/semantic search.
    """
    query_stripped = query.strip()
    words = query_stripped.split()
    
    # Single word that starts with capital letter (proper noun/name)
    if len(words) == 1 and query_stripped[0].isupper():
        return True
    
    # Contains underscores (likely an identifier like lib_cs_mimikatz)
    if '_' in query_stripped:
        return True
    
    # Contains mixed case within a word (CamelCase)
    for word in words:
        if any(c.isupper() for c in word[1:]):  # Capital letter not at start
            return True
    
    # Looks like an ID pattern (letters + numbers)
    if re.match(r'^[A-Za-z]+[-_]?\d+', query_stripped):
        return True
    
    # All caps (acronym or ID)
    if query_stripped.isupper() and len(query_stripped) >= 2:
        return True
    
    return False


def analyze_query_intent(query: str) -> QueryIntent:
    """Analyze query to determine optimal search strategy."""
    query_lower = query.lower().strip()
    query_stripped = query.strip()
    signals = []
    extracted_filters = {}
    
    # Check for garbage query first
    if is_garbage_query(query):
        return QueryIntent(
            strategy=SearchStrategy.KEYWORD,
            confidence=0.5,
            signals=["low_quality_query"],
            extracted_filters={},
            search_terms=[],
            is_low_quality=True
        )
    
    # Extract search terms early (needed for multiple checks)
    words = re.findall(r'\b[a-zA-Z0-9_]+\b', query_lower)
    search_terms = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    
    # Also preserve original case terms for proper nouns
    original_words = re.findall(r'\b[a-zA-Z0-9_]+\b', query_stripped)
    original_search_terms = [w for w in original_words if w.lower() not in STOP_WORDS and len(w) > 2]
    
    # =========================================================================
    # OPTION 1 FIX: Force keyword search for short/specific queries
    # =========================================================================
    word_count = len(query_stripped.split())
    
    # Short queries (1-2 meaningful words) that aren't questions -> keyword search
    if word_count <= 2:
        is_question = any(re.search(p, query, re.IGNORECASE) for p in NEURAL_PATTERNS)
        
        if not is_question:
            # Check if it looks like a name/identifier
            if looks_like_identifier_or_name(query_stripped):
                return QueryIntent(
                    strategy=SearchStrategy.KEYWORD,
                    confidence=0.90,
                    signals=["short_query_identifier_or_name", f"word_count={word_count}"],
                    extracted_filters=extracted_filters,
                    search_terms=original_search_terms if original_search_terms else search_terms,
                    is_low_quality=False
                )
            
            # Even if not obviously a name, short non-question queries should prefer keyword
            return QueryIntent(
                strategy=SearchStrategy.KEYWORD,
                confidence=0.85,
                signals=["short_specific_query_forced_keyword", f"word_count={word_count}"],
                extracted_filters=extracted_filters,
                search_terms=original_search_terms if original_search_terms else search_terms,
                is_low_quality=False
            )
    
    # =========================================================================
    # Standard intent analysis for longer queries
    # =========================================================================
    
    keyword_score = 0
    neural_score = 0
    
    # Check keyword patterns
    for pattern in KEYWORD_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            keyword_score += 1
            signals.append(f"keyword_pattern: {pattern[:30]}")
    
    # Check neural patterns
    for pattern in NEURAL_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            neural_score += 1
            signals.append(f"neural_pattern: {pattern[:30]}")
    
    # Check for cluster filters (don't add to keyword score - just extract filter)
    for cluster, keywords in CLUSTER_KEYWORDS.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', query_lower):
                extracted_filters['cluster'] = cluster
                signals.append(f"cluster_filter: {cluster}")
                break
        if 'cluster' in extracted_filters:
            break
    
    # Check for type filters
    for doc_type, keywords in TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                extracted_filters['type'] = doc_type
                signals.append(f"type_filter: {doc_type}")
                break
        if 'type' in extracted_filters:
            break
    
    # Identifier/name detection for longer queries too
    if looks_like_identifier_or_name(query_stripped):
        keyword_score += 2
        signals.append("contains_identifier_or_name")
    
    # Short queries with specific terms lean keyword
    if len(query.split()) <= 3 and not any(re.search(p, query, re.IGNORECASE) for p in NEURAL_PATTERNS):
        keyword_score += 1
        signals.append("short_specific_query")
    
    # Long queries lean neural
    if len(query.split()) >= 6:
        neural_score += 1
        signals.append("long_natural_language")
    
    # Determine strategy
    if keyword_score > neural_score + 1:
        strategy = SearchStrategy.KEYWORD
        confidence = min(0.95, 0.6 + (keyword_score - neural_score) * 0.1)
    elif neural_score > keyword_score + 1:
        strategy = SearchStrategy.NEURAL
        confidence = min(0.95, 0.6 + (neural_score - keyword_score) * 0.1)
    else:
        strategy = SearchStrategy.HYBRID
        confidence = 0.7
        signals.append("scores_close_using_hybrid")
    
    # Use original case terms if available (better for names like "Kane")
    final_terms = original_search_terms if original_search_terms else search_terms
    
    return QueryIntent(
        strategy=strategy,
        confidence=confidence,
        signals=signals,
        extracted_filters=extracted_filters,
        search_terms=final_terms,
        is_low_quality=False
    )

