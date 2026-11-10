"""Corpus-backed retrieval providers for ProtoGraph (RAG, not SPARQL).

These make the document corpus + its RAG index the main retrieval source, so
search does NOT depend on GraphDB being up. GraphDB/SPARQL retrieval is parked.
"""
