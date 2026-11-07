# Event Topics Reference

Generated from core/events.py.

- `task.compiled`
- `task.all_steps_completed`
- `task.failed`
- `outcome.recorded`
- `evolution.cycle_started`
- `evolution.cycle_completed`
- `skill.promoted`
- `skill.deprecated`
- `mind.turn_start  (agent)`
- `mind.turn_end  (agent)`
- `mind.llm_call  (agent, step)`
- `mind.tool_call  (agent, tool)`
- `mind.delegate  (agent, query)`
- `mind.context_injected  (agent, memory_ids, skill_hint)`
- `mind.store  (namespace, memory_id, kind)`
- `mind.recall  (namespace, memory_ids, count)`
- `mind.learned  (memory_id, namespace)`
- `mind.skill_created  (skill_id, trigger)`
- `mind.traverse  (query, path (node ids in visit order), edges (edge ids))`
- `mind.node_touched  (node_id, usage_score, tier, style)`
- `mind.gap_detected  (term, reason, missing_aspects)`
- `mind.research_start  (topic, queries)`
- `mind.research_step  (topic, url, status)`
- `mind.research_done  (topic, node_id, records, position, style)`
- `mind.research_answer  (question, answer (cited follow-up after background research))`
- `mind.node_spawned  (node_id, kind, label, collection, style, position)`
- `mind.edge_drawn  (edge_id, src, dst, relation)`
- `mind.curiosity_added  (id, question, topic_node_id, priority)`
- `mind.sleep_start  (reason, mode)`
- `mind.sleep_phase  (name)`
- `mind.embedding_migrated  (from_node, to_node, count)`
- `mind.node_fading  (node_id, usage_score, style)`
- `mind.node_pruned  (node_id, collection)`
- `mind.layout_updated  (moves [{node_id, position}] after the sleep-time refit)`
- `mind.sleep_end  (report)`
