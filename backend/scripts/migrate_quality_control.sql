-- Memory Quality Control migration
-- Run once against the existing PostgreSQL database:
--   psql -d knowledge_agent -f scripts/migrate_quality_control.sql

-- 1. Add status column to archival_memory (backward-compatible: existing rows get 'active')
ALTER TABLE archival_memory ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'active';
CREATE INDEX IF NOT EXISTS idx_archival_memory_status ON archival_memory (status);

-- 2. Version history table — snapshots of archival_memory before UPDATE/DELETE
CREATE TABLE IF NOT EXISTS archival_versions (
    id TEXT PRIMARY KEY,
    archival_id TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(1024),
    version_number INT NOT NULL DEFAULT 1,
    change_type TEXT NOT NULL,       -- 'update' | 'delete' | 'rollback'
    changed_by TEXT DEFAULT 'memory_pipeline',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_archival_versions_archival_id
    ON archival_versions (archival_id, version_number DESC);
CREATE INDEX IF NOT EXISTS idx_archival_versions_created
    ON archival_versions (created_at DESC);

-- 3. Conflict review queue — low-confidence conflicts pending human review
CREATE TABLE IF NOT EXISTS conflict_reviews (
    id TEXT PRIMARY KEY,
    new_fact TEXT NOT NULL,
    existing_id TEXT NOT NULL,
    existing_content TEXT NOT NULL,
    similarity_score FLOAT NOT NULL,
    llm_decision TEXT,               -- 'update' | 'skip'
    llm_merged_text TEXT,
    llm_confidence FLOAT,            -- 0.0-1.0
    status TEXT NOT NULL DEFAULT 'pending',  -- 'pending' | 'approved' | 'rejected' | 'modified'
    resolution_text TEXT,
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conflict_reviews_status
    ON conflict_reviews (status, created_at DESC);
