import axios from 'axios';
import type { Document, ParsedElement, PaginatedResponse } from '../types';

const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
});

// ── Documents ────────────────────────────────────────────────────────────────

export const documentsApi = {
  list: (page = 1, pageSize = 20): Promise<PaginatedResponse<Document>> =>
    apiClient
      .get('/documents', { params: { page, page_size: pageSize } })
      .then((r) => r.data),

  get: (id: string): Promise<Document> =>
    apiClient.get(`/documents/${id}`).then((r) => r.data),

  upload: (file: File): Promise<Document> => {
    const form = new FormData();
    form.append('file', file);
    return apiClient
      .post('/documents/upload', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      .then((r) => r.data);
  },

  delete: (id: string): Promise<void> =>
    apiClient.delete(`/documents/${id}`).then(() => undefined),
};

// ── Parsed Elements ──────────────────────────────────────────────────────────

export const elementsApi = {
  listByDocument: (
    documentId: string,
    page = 1,
    pageSize = 50,
  ): Promise<PaginatedResponse<ParsedElement>> =>
    apiClient
      .get(`/documents/${documentId}/elements`, {
        params: { page, page_size: pageSize },
      })
      .then((r) => r.data),

  semanticSearch: (
    query: string,
    limit = 10,
  ): Promise<ParsedElement[]> =>
    apiClient
      .post('/elements/search', { query, limit })
      .then((r) => r.data),
};

// ── Health ───────────────────────────────────────────────────────────────────

export const healthApi = {
  check: (): Promise<{ status: string }> =>
    apiClient.get('/health').then((r) => r.data),
};

export default apiClient;
