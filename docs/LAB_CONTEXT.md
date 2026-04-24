## Lab context
1. Session 1: User nói “tôi thích Python, không thích Java” → agent ghi vào
Redis
2. Session 2 (new process): Agent load memory → proactively suggest
Python solution mà không cần hỏi lại
3. Session 3: Agent recall episode “user bị confused async/await” → tự
thêm explanation
4. So sánh: agent có memory vs không memory — response relevance,
user satisfaction

## Objectives
Mục tiêu: Build Multi-Memory Agent với LangGraph
Deliverable: Agent với full memory stack + benchmark report: so sánh agent
có/không memory trên 10 multi-turn conversations

## Instructions
1. Implement 4 memory backends: ConversationBufferMemory (short-term),
Redis (long-term), JSON episodic log, Chroma (semantic)
2. Build memory router: chọn memory type phù hợp dựa trên query intent —
user preference vs factual recall vs experience recall
3. Context window management: auto-trim khi gần limit, priority-based
eviction theo 4-level hierarchy
4. Benchmark: so sánh agent có/không memory trên 10 multi-turn
conversations — đo response relevance, context utilization, token efficiency