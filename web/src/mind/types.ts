/**
 * Shared types for the 3-D mind view (mirrors memory/mind_view.py and the
 * event schema in core/events.py).
 */
export interface NodeStyle {
  tier: string;          // high | moderate | low | fading | curiosity | agent | skill | source
  scale: number;
  emissive: number;
  saturation: number;
  opacity: number;
}

export interface GraphNode {
  id: string;
  kind: string;          // entity | abstraction | curiosity | skill | agent | source
  label: string;
  status: string;
  pinned: boolean;
  access_count: number;
  usage_score: number;
  percentile: number | null;
  tier: string;
  style: NodeStyle;
  position: [number, number, number];
  collection: string | null;
  record_count: number;
  empty?: boolean;       // a topic/hub whose vector store holds no knowledge yet
  confidence: number;
}

/** Peter's own node id: drawn as the core sphere at the origin, never as a graph node. */
export const SELF_ID = 'self_peter';

export interface GraphEdge {
  id: string;
  src: string;           // may be SELF_ID (Peter's has_subject edges)
  dst: string;
  relation: string;
  weight?: number | null;   // cosine similarity of the two nodes' vectors (null: not a vector edge)
  primary?: boolean;        // the primary 'abstracts' parent of dst
}

/** A record-level vector link followed by recall (traverse event). */
export interface VectorLink {
  edge_id: string;
  from_node: string;
  to_node: string;
  from_record: string;
  to_record: string;
  similarity: number;
}

export interface GraphSnapshot {
  view: 'graph';
  layout?: 'graph' | 'tree' | 'cluster' | 'semantic';
  core_links?: string[];   // main subjects: Peter's stored has_subject edges
  self?: { id: string; position: [number, number, number] };
  cognition?: { semantic_ok: boolean };
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: { nodes: number; edges: number; by_kind: Record<string, number>; collections: number;
    semantic_ok?: boolean; empty_collections?: number };
}

export interface RecordPoint {
  id: string;
  document: string;
  namespace?: string;
  importance: number;
  kind: string;
  position: [number, number, number];
}

export interface NodeDetail {
  id: string;
  kind: string;
  label: string;
  status: string;
  pinned: boolean;
  usage: { score: number; percentile: number | null; access_count: number; last_accessed: number | null; created_at: number | null };
  tier: string;
  style: NodeStyle;
  confidence: number;
  collection: { name: string; record_count: number } | null;
  records: Array<{
    id: string; text: string; kind: string; confidence: number; trust: string;
    sources: Array<{ url: string; title?: string }>; origin_node?: string | null;
    embedding_preview: number[]; embedding_norm: number; embedding_dims: number;
  }>;
  sources: string[];
  aliases: string[];
  edges: Array<{ id: string; relation: string; direction: 'in' | 'out'; node_id: string; label: string; kind: string | null;
    weight?: number | null; primary?: boolean }>;
  empty?: boolean;
  curiosity: Array<{ id: string; question: string; status: string; priority: number }>;
}

/** Base hue per node kind (HSL hue in [0,1]). */
export const KIND_HUE: Record<string, number> = {
  entity:      0.60,   // blue
  abstraction: 0.78,   // violet
  curiosity:   0.13,   // amber
  skill:       0.33,   // green
  agent:       0.52,   // cyan
  source:      0.00,   // grey (saturation 0)
};

/** Colour per edge relation (hex). */
export const RELATION_HEX: Record<string, number> = {
  abstracts:     0xb18cff,
  bridges:       0xff9fd2,
  related_to:    0x7c9eff,
  curious_about: 0xffc857,
  cites:         0x5b6b8a,
  spawned_by:    0x3a4766,
  similar_to:    0x5fe3c0,   // drawn dashed: a neighbour by meaning, not a part-of link
  has_subject:   0x9fb8ff,   // Peter -> main subject
};
export const DEFAULT_EDGE_HEX = 0x3a4766;
