// Core domain types matching the Supabase schema

export type DocumentStatus = 'pending' | 'processing' | 'done' | 'error';

export interface Document {
  id: string;
  filename: string;
  file_url: string;
  status: DocumentStatus;
  created_at: string;
  updated_at: string;
}

export type ElementType =
  | 'title'
  | 'text'
  | 'image'
  | 'table'
  | 'list'
  | 'caption'
  | 'header'
  | 'footer';

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface ParsedElement {
  id: string;
  document_id: string;
  element_type: ElementType;
  content: string;
  bounding_box: BoundingBox | null;
  confidence_score: number | null;
  embedding: number[] | null;
  metadata: Record<string, unknown>;
  created_at: string;
}

export interface PaginatedResponse<T> {
  data: T[];
  count: number;
  page: number;
  page_size: number;
}
