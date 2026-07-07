-- 1. Enable pgvector for semantic search
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;

-- 2. Create the core pipeline tables
CREATE TABLE public.documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL,
    file_url TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE public.parsed_elements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE,
    element_type TEXT NOT NULL, 
    content TEXT NOT NULL,
    bounding_box JSONB, 
    confidence_score FLOAT,
    embedding vector(1536), 
    metadata JSONB DEFAULT '{}'::jsonb, 
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3. Enable Row Level Security (RLS)
ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.parsed_elements ENABLE ROW LEVEL SECURITY;

-- 4. Create RLS Policies
CREATE POLICY "Enable read access for authenticated users" ON public.documents FOR SELECT TO authenticated USING (true);
CREATE POLICY "Enable service role full access" ON public.documents FOR ALL TO service_role USING (true);
CREATE POLICY "Enable read access for authenticated users" ON public.parsed_elements FOR SELECT TO authenticated USING (true);
CREATE POLICY "Enable service role full access" ON public.parsed_elements FOR ALL TO service_role USING (true);

-- 5. Create an HNSW index
CREATE INDEX ON public.parsed_elements USING hnsw (embedding vector_cosine_ops);
