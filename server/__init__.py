"""
Peter AI — Web Server package.

Submodules (added progressively):
  server.app          FastAPI application factory
  server.auth         JWT / Supabase token verification → UserContext
  server.peter_pool   Per-user PeterAgent cache
  server.ws           WebSocket event stream
  server.projection   Per-user PCA/UMAP 3-D projection
  server.routes.*     REST endpoint routers
"""
