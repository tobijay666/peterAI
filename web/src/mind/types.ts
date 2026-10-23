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
  confidence: number;
}

export interface GraphEdge {
  id: string;
  src: string;
  dst: string;
  relation: string;
}

export interface GraphSnapshot {
  view: 'graph';
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: { nodes: number; edges: number; by_kind: Record<string, number>; collections: number };
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
  edges: Array<{ id: string; relation: string; direction: 'in' | 'out'; node_id: string; label: string; kind: string | null }>;
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
};
export const DEFAULT_EDGE_HEX = 0x3a4766;
