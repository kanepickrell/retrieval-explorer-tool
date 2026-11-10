"""
Unified Search Router for ProtoGraph - v3 IMPROVED
Intelligently routes queries to the best search strategy (keyword vs neural vs hybrid)

Fixes in v3:
- Short/specific queries (1-2 words like names, IDs) forced to keyword search
- Expanded keyword field search (owner, author, created_by, metadata, tags, etc.)
- Better handling of proper nouns and identifiers
- Improved relevance scoring for keyword matches
"""

import re
import time
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, HTTPException
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


# =============================================================================
# UNIFIED SEARCH ROUTER CREATION
# =============================================================================

def create_unified_search_router(db, neural_router, ollama_client=None, ollama_model: str = "gpt-oss:120b"):
    """Create the unified search router with access to both search backends."""
    router = APIRouter(prefix="/api/search", tags=["Unified Search"])
    
    EDGE_COLLECTIONS = {'CONTAINS', 'PRODUCES', 'REFERENCES', 'LEADS_TO', 'STARTS_WITH', 'ontology_edges'}
    SCHEMA_COLLECTIONS = {
        'ontology_concepts', 'concepts', 'ontology_taxonomies',
        'taxonomy_schemes', 'taxonomy_terms', 'taxonomies',
        'ontology_relationships', 'ontology_edges', 'relationship_types',
        'ontology_properties', 'schema_definitions',
    }
    
    def get_cluster_from_doc(doc: Dict) -> str:
        """Determine cluster from document attributes"""
        collaboration = doc.get("collaboration_with", [])
        
        if isinstance(collaboration, list):
            if "Automation" in collaboration:
                return "automation"
            elif "Range" in collaboration:
                return "range"
            elif "ContentDev" in collaboration:
                return "content"
            elif "OPFOR" in collaboration:
                return "opfor"
        
        # Fallback: check document type or collection
        doc_type = doc.get("type", "").lower()
        if "script" in doc_type or "automation" in doc_type:
            return "automation"
        elif "vm" in doc_type or "range" in doc_type:
            return "range"
        
        return "planning"
    
    async def run_keyword_search(
        query: str,
        search_terms: List[str],
        filters: Dict[str, Any],
        limit: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Execute keyword-based search against ArangoDB.
        
        OPTION 2 FIX: Expanded field search to include:
        - name, _key, description, label (original)
        - owner, author, created_by (new)
        - tags array (new)
        - metadata object (new - converted to string)
        - scenario_id (new)
        - status (new)
        """
        
        if not db:
            return []
        
        # Don't search if no meaningful terms
        if not search_terms:
            print("   ⚠️ Keyword search: No search terms, returning empty")
            return []
        
        try:
            all_collections = [c['name'] for c in db.collections()
                             if not c['name'].startswith('_')
                             and c['name'] not in EDGE_COLLECTIONS
                             and c['name'] not in SCHEMA_COLLECTIONS]
            
            results = []
            
            for coll_name in all_collections:
                try:
                    # Build text search conditions - EXPANDED FIELD LIST
                    term_conditions = []
                    for term in search_terms:
                        # Escape single quotes in term
                        safe_term = term.replace("'", "\\'").lower()
                        
                        # OPTION 2 FIX: Search many more fields
                        term_conditions.append(f"""(
                            LIKE(LOWER(doc.name || ''), '%{safe_term}%', true) OR
                            LIKE(LOWER(doc._key || ''), '%{safe_term}%', true) OR
                            LIKE(LOWER(doc.description || ''), '%{safe_term}%', true) OR
                            LIKE(LOWER(doc.label || ''), '%{safe_term}%', true) OR
                            LIKE(LOWER(doc.owner || ''), '%{safe_term}%', true) OR
                            LIKE(LOWER(doc.author || ''), '%{safe_term}%', true) OR
                            LIKE(LOWER(doc.created_by || ''), '%{safe_term}%', true) OR
                            LIKE(LOWER(doc.assigned_to || ''), '%{safe_term}%', true) OR
                            LIKE(LOWER(doc.scenario_id || ''), '%{safe_term}%', true) OR
                            LIKE(LOWER(doc.status || ''), '%{safe_term}%', true) OR
                            LIKE(LOWER(doc.category || ''), '%{safe_term}%', true) OR
                            LIKE(LOWER(doc.subcategory || ''), '%{safe_term}%', true) OR
                            LIKE(LOWER(doc.tactic || ''), '%{safe_term}%', true) OR
                            LIKE(LOWER(doc.technique_id || ''), '%{safe_term}%', true) OR
                            '{safe_term}' IN (FOR t IN (doc.tags || []) RETURN LOWER(t)) OR
                            LIKE(LOWER(TO_STRING(doc.metadata || {{}})), '%{safe_term}%', true) OR
                            LIKE(LOWER(TO_STRING(doc.collaboration_with || [])), '%{safe_term}%', true)
                        )""")
                    
                    # Require at least one term to match
                    text_filter = " OR ".join(term_conditions) if term_conditions else "true"
                    
                    # Build match scoring - weight different field matches
                    score_expressions = []
                    for term in search_terms:
                        safe_term = term.replace("'", "\\'").lower()
                        score_expressions.append(f"""(
                            (LIKE(LOWER(doc.name || ''), '%{safe_term}%', true) ? 3 : 0) +
                            (LIKE(LOWER(doc._key || ''), '%{safe_term}%', true) ? 3 : 0) +
                            (LIKE(LOWER(doc.label || ''), '%{safe_term}%', true) ? 3 : 0) +
                            (LIKE(LOWER(doc.owner || ''), '%{safe_term}%', true) ? 2 : 0) +
                            (LIKE(LOWER(doc.author || ''), '%{safe_term}%', true) ? 2 : 0) +
                            (LIKE(LOWER(doc.created_by || ''), '%{safe_term}%', true) ? 2 : 0) +
                            (LIKE(LOWER(doc.assigned_to || ''), '%{safe_term}%', true) ? 2 : 0) +
                            (LIKE(LOWER(doc.description || ''), '%{safe_term}%', true) ? 1 : 0) +
                            (LIKE(LOWER(doc.scenario_id || ''), '%{safe_term}%', true) ? 1 : 0) +
                            (LIKE(LOWER(doc.category || ''), '%{safe_term}%', true) ? 1 : 0) +
                            ('{safe_term}' IN (FOR t IN (doc.tags || []) RETURN LOWER(t)) ? 2 : 0) +
                            (LIKE(LOWER(TO_STRING(doc.metadata || {{}})), '%{safe_term}%', true) ? 1 : 0)
                        )""")
                    
                    score_calc = " + ".join(score_expressions) if score_expressions else "0"
                    
                    query_aql = f"""
                        FOR doc IN `{coll_name}`
                            FILTER {text_filter}
                            LET match_score = ({score_calc})
                            FILTER match_score > 0
                            SORT match_score DESC
                            LIMIT {limit}
                            RETURN {{
                                doc: doc,
                                match_score: match_score
                            }}
                    """
                    
                    docs = list(db.aql.execute(query_aql))
                    
                    for item in docs:
                        doc = item['doc']
                        match_score = item.get('match_score', 1)
                        
                        # Determine which field matched (for debugging/display)
                        matched_fields = []
                        for term in search_terms:
                            term_lower = term.lower()
                            if term_lower in str(doc.get('name', '')).lower():
                                matched_fields.append('name')
                            if term_lower in str(doc.get('_key', '')).lower():
                                matched_fields.append('_key')
                            if term_lower in str(doc.get('owner', '')).lower():
                                matched_fields.append('owner')
                            if term_lower in str(doc.get('author', '')).lower():
                                matched_fields.append('author')
                            if term_lower in str(doc.get('created_by', '')).lower():
                                matched_fields.append('created_by')
                            if term_lower in str(doc.get('assigned_to', '')).lower():
                                matched_fields.append('assigned_to')
                            if term_lower in str(doc.get('description', '')).lower():
                                matched_fields.append('description')
                            tags = doc.get('tags', [])
                            if isinstance(tags, list) and any(term_lower in str(t).lower() for t in tags):
                                matched_fields.append('tags')
                        
                        results.append({
                            "id": doc["_id"],
                            "label": doc.get("name", doc.get("label", doc.get("_key", "Unnamed"))),
                            "type": coll_name,
                            "cluster": get_cluster_from_doc(doc),
                            "description": doc.get("description", ""),
                            "relevance_score": min(1.0, match_score / 10.0),  # Normalize
                            "source": "keyword",
                            "scenario_id": doc.get("scenario_id", "unknown"),
                            "matched_fields": list(set(matched_fields)),  # Deduplicated
                            "metadata": {
                                "collaboration_with": doc.get("collaboration_with", []),
                                "status": doc.get("status", ""),
                                "tags": doc.get("tags", []),
                                "owner": doc.get("owner", ""),
                                "author": doc.get("author", ""),
                                "created_by": doc.get("created_by", ""),
                            }
                        })
                        
                except Exception as coll_error:
                    print(f"   ⚠️ Error searching {coll_name}: {coll_error}")
                    continue
            
            # Apply cluster filter in Python (softer filtering - boost, don't exclude)
            if filters.get('cluster'):
                target_cluster = filters['cluster']
                for r in results:
                    if r['cluster'] == target_cluster:
                        r['relevance_score'] = min(1.0, r['relevance_score'] + 0.3)
            
            # Sort by relevance
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            print(f"   ✅ Keyword search found {len(results)} results")
            return results[:limit]
            
        except Exception as e:
            print(f"❌ Keyword search error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def run_neural_search(query: str, limit: int = 20) -> List[Dict]:
        """Run neural search and convert results to unified format"""
        try:
            # Initialize neural router if needed
            if not getattr(neural_router, '_initialized', False):
                neural_router.initialize()
            
            # Run neural search
            neural_results = None
            
            if hasattr(neural_router, 'search'):
                search_method = neural_router.search
                import asyncio
                if asyncio.iscoroutinefunction(search_method):
                    neural_results = await search_method(query)
                else:
                    neural_results = search_method(query)
            
            if neural_results is None:
                print("⚠️ Neural search returned None")
                return []
            
            unified_results = []
            
            # Get contributing_agents
            contributing_agents = getattr(neural_results, 'contributing_agents', []) or []
            contexts = getattr(neural_results, 'contexts', {}) or {}
            
            # Get the agents dictionary from the neural router
            agents_dict = getattr(neural_router, '_agents', None)
            if agents_dict is None:
                agents_dict = getattr(neural_router, 'agents', {})
            
            # Collect all node IDs from contributing agents
            all_node_ids = []
            agent_node_map = {}
            agent_scores = {}  # Track which agent contributed each node
            
            # Handle both list and dict formats for contributing_agents
            agent_ids_to_process = []
            if isinstance(contributing_agents, list):
                agent_ids_to_process = contributing_agents
            elif isinstance(contributing_agents, dict):
                agent_ids_to_process = list(contributing_agents.keys())
            
            # Score agents by their position (first = most relevant)
            for idx, agent_id in enumerate(agent_ids_to_process):
                agent_obj = agents_dict.get(agent_id)
                
                if agent_obj is None:
                    continue
                
                # Get node IDs
                node_ids = []
                for attr_name in ['node_ids', 'nodes', 'member_nodes', 'members']:
                    if hasattr(agent_obj, attr_name):
                        val = getattr(agent_obj, attr_name)
                        if val:
                            node_ids = list(val) if not isinstance(val, list) else val
                            break
                
                # Score based on agent position (earlier = higher score)
                agent_score = 1.0 - (idx * 0.1)  # First agent: 1.0, second: 0.9, etc.
                
                for nid in node_ids:
                    if nid not in agent_node_map:
                        all_node_ids.append(nid)
                        agent_node_map[nid] = agent_id
                        agent_scores[nid] = agent_score
                    else:
                        # Node found in multiple agents - boost score
                        agent_scores[nid] = min(1.0, agent_scores[nid] + 0.1)
            
            print(f"   📦 Total unique nodes from agents: {len(all_node_ids)}")
            
            # Get node data from neural router's cache
            nodes_data = getattr(neural_router, '_nodes', None)
            if nodes_data is None:
                nodes_data = getattr(neural_router, 'nodes', {})
            
            # Convert list to dict if needed
            if isinstance(nodes_data, list):
                nodes_dict = {}
                for n in nodes_data:
                    if isinstance(n, dict):
                        nid = n.get('id', n.get('_id', ''))
                        if nid:
                            nodes_dict[nid] = n
                nodes_data = nodes_dict
            
            print(f"   🔍 Neural router has {len(nodes_data)} nodes in cache")
            
            for node_id in all_node_ids[:limit]:
                node = nodes_data.get(node_id)
                
                if node is None:
                    short_id = node_id.split('/')[-1] if '/' in node_id else node_id
                    node = nodes_data.get(short_id)
                
                agent_id = agent_node_map.get(node_id, '')
                neural_context = None
                if isinstance(contexts, dict):
                    neural_context = contexts.get(agent_id, '')
                
                # Get score for this node
                relevance_score = agent_scores.get(node_id, 0.5)
                
                if node is not None:
                    if isinstance(node, dict):
                        unified_results.append({
                            "id": node.get("id", node.get("_id", node_id)),
                            "label": node.get("label", node.get("name", "Unknown")),
                            "type": node.get("type", node.get("collection", "Unknown")),
                            "cluster": node.get("cluster", "unknown"),
                            "description": node.get("description", ""),
                            "relevance_score": relevance_score,
                            "source": "neural",
                            "neural_context": neural_context if neural_context else None,
                            "scenario_id": node.get("scenario_id", "unknown"),
                            "metadata": node.get("metadata", {})
                        })
                    else:
                        unified_results.append({
                            "id": getattr(node, 'id', getattr(node, '_id', node_id)),
                            "label": getattr(node, 'label', getattr(node, 'name', 'Unknown')),
                            "type": getattr(node, 'type', getattr(node, 'collection', 'Unknown')),
                            "cluster": getattr(node, 'cluster', 'unknown'),
                            "description": getattr(node, 'description', ''),
                            "relevance_score": relevance_score,
                            "source": "neural",
                            "neural_context": neural_context if neural_context else None,
                            "scenario_id": getattr(node, 'scenario_id', 'unknown'),
                            "metadata": getattr(node, 'metadata', {})
                        })
                else:
                    unified_results.append({
                        "id": node_id,
                        "label": node_id.split('/')[-1] if '/' in node_id else node_id,
                        "type": node_id.split('/')[0] if '/' in node_id else "Unknown",
                        "cluster": "unknown",
                        "description": "",
                        "relevance_score": relevance_score,
                        "source": "neural",
                        "neural_context": neural_context if neural_context else None,
                        "scenario_id": "unknown",
                        "metadata": {}
                    })
            
            # Sort by relevance score
            unified_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            print(f"   ✅ Returning {len(unified_results)} neural results")
            return unified_results
            
        except Exception as e:
            print(f"❌ Neural search error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def merge_results(
        keyword_results: List[Dict],
        neural_results: List[Dict],
        limit: int = 50
    ) -> List[Dict]:
        """Merge and deduplicate results from both search methods."""
        
        merged = {}
        
        for r in keyword_results:
            rid = r['id']
            merged[rid] = {
                **r,
                'keyword_score': r['relevance_score'],
                'neural_score': 0.0,
                'sources': ['keyword']
            }
        
        for r in neural_results:
            rid = r['id']
            if rid in merged:
                merged[rid]['neural_score'] = r['relevance_score']
                merged[rid]['sources'].append('neural')
                merged[rid]['source'] = 'both'
                merged[rid]['neural_context'] = r.get('neural_context')
            else:
                merged[rid] = {
                    **r,
                    'keyword_score': 0.0,
                    'neural_score': r['relevance_score'],
                    'sources': ['neural']
                }
        
        for rid, item in merged.items():
            # Weight keyword matches higher when present (they're more precise)
            both_boost = 0.3 if len(item['sources']) > 1 else 0.0
            keyword_weight = 0.6 if item['keyword_score'] > 0 else 0.4
            neural_weight = 1.0 - keyword_weight
            
            item['relevance_score'] = (
                item['keyword_score'] * keyword_weight +
                item['neural_score'] * neural_weight +
                both_boost
            )
            del item['keyword_score']
            del item['neural_score']
            del item['sources']
        
        results = list(merged.values())
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return results[:limit]
    
    @router.get("/unified")
    async def unified_search(
        q: str = Query(..., min_length=1, description="Search query"),
        limit: int = Query(50, le=100, description="Max results"),
        force_strategy: Optional[str] = Query(None, description="Force 'keyword', 'neural', or 'hybrid'")
    ):
        """Unified search that intelligently routes to the best search strategy."""
        start_time = time.time()
        
        # Analyze query intent
        intent = analyze_query_intent(q)
        
        # Allow override
        if force_strategy:
            try:
                intent.strategy = SearchStrategy(force_strategy)
                intent.signals.append(f"forced_strategy: {force_strategy}")
            except:
                pass
        
        print(f"\n{'='*60}")
        print(f"🔍 UNIFIED SEARCH: '{q}'")
        print(f"📊 Strategy: {intent.strategy.value} (confidence: {intent.confidence:.2f})")
        print(f"📋 Signals: {intent.signals}")
        print(f"🔑 Terms: {intent.search_terms}")
        print(f"🎯 Filters: {intent.extracted_filters}")
        if intent.is_low_quality:
            print(f"⚠️  LOW QUALITY QUERY - limiting search")
        print(f"{'='*60}")
        
        keyword_results = []
        neural_results = []
        final_results = []
        
        # Handle low-quality queries specially
        if intent.is_low_quality:
            # Return empty or minimal results for garbage queries
            elapsed_ms = (time.time() - start_time) * 1000
            return {
                "query": q,
                "strategy_used": "none",
                "intent_analysis": {
                    "strategy": intent.strategy.value,
                    "confidence": intent.confidence,
                    "signals": intent.signals,
                    "extracted_filters": intent.extracted_filters,
                    "search_terms": intent.search_terms,
                    "is_low_quality": True
                },
                "total_results": 0,
                "keyword_results": 0,
                "neural_results": 0,
                "merged_results": 0,
                "time_ms": elapsed_ms,
                "results": [],
                "message": "Query too short or contains no meaningful search terms. Please provide more specific terms."
            }
        
        # Execute based on strategy
        if intent.strategy == SearchStrategy.KEYWORD:
            keyword_results = await run_keyword_search(
                q, intent.search_terms, intent.extracted_filters, limit
            )
            final_results = keyword_results
            
        elif intent.strategy == SearchStrategy.NEURAL:
            neural_results = await run_neural_search(q, limit)
            if not neural_results:
                print("⚠️ Neural search returned no results, falling back to keyword")
                keyword_results = await run_keyword_search(
                    q, intent.search_terms, intent.extracted_filters, limit
                )
                final_results = keyword_results
            else:
                final_results = neural_results
            
        else:  # HYBRID
            keyword_results = await run_keyword_search(
                q, intent.search_terms, intent.extracted_filters, limit
            )
            neural_results = await run_neural_search(q, limit)
            final_results = merge_results(keyword_results, neural_results, limit)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        print(f"✅ Results: {len(keyword_results)} keyword, {len(neural_results)} neural, {len(final_results)} final")
        print(f"⏱️  Time: {elapsed_ms:.0f}ms")
        
        return {
            "query": q,
            "strategy_used": intent.strategy.value,
            "intent_analysis": {
                "strategy": intent.strategy.value,
                "confidence": intent.confidence,
                "signals": intent.signals,
                "extracted_filters": intent.extracted_filters,
                "search_terms": intent.search_terms
            },
            "total_results": len(final_results),
            "keyword_results": len(keyword_results),
            "neural_results": len(neural_results),
            "merged_results": len(final_results) if intent.strategy == SearchStrategy.HYBRID else 0,
            "time_ms": elapsed_ms,
            "results": final_results
        }
    
    @router.get("/analyze")
    async def analyze_query(q: str = Query(..., min_length=1)):
        """Analyze a query without executing search."""
        intent = analyze_query_intent(q)
        
        return {
            "query": q,
            "recommended_strategy": intent.strategy.value,
            "confidence": intent.confidence,
            "signals": intent.signals,
            "extracted_filters": intent.extracted_filters,
            "search_terms": intent.search_terms,
            "is_low_quality": intent.is_low_quality,
            "explanation": _get_strategy_explanation(intent)
        }
    
    def _get_strategy_explanation(intent: QueryIntent) -> str:
        """Generate human-readable explanation of routing decision"""
        if intent.is_low_quality:
            return "Query is too short or contains only common words. Please provide more specific search terms."
        elif intent.strategy == SearchStrategy.KEYWORD:
            if "short_query_identifier_or_name" in str(intent.signals):
                return "Using keyword search: query appears to be a name, identifier, or proper noun (1-2 words, capitalized)."
            elif "short_specific_query_forced_keyword" in str(intent.signals):
                return "Using keyword search: short specific query (1-2 words) - best suited for exact matching."
            else:
                return "Using keyword search: query contains exact terms, artifact patterns, or simple lookup language."
        elif intent.strategy == SearchStrategy.NEURAL:
            return "Using neural/semantic search: query is a question, uses conceptual language, or asks about related/similar items."
        else:
            return "Using hybrid search: query has mixed signals - running both keyword and neural search and merging results."
    
    return router