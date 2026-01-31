import type { Product, ProductListResponse, SearchResponse } from '../types';

const API_BASE = '/api';

export async function getProducts(category?: string, search?: string, page?: number, pageSize?: number): Promise<ProductListResponse> {
  const params = new URLSearchParams();
  if (category && category !== 'All') params.append('category', category);
  if (search) params.append('search', search);
  if (page) params.append('page', page.toString());
  if (pageSize) params.append('pageSize', pageSize.toString());
  
  const url = `${API_BASE}/products${params.toString() ? '?' + params.toString() : ''}`;
  const response = await fetch(url);
  if (!response.ok) throw new Error('Failed to fetch products');
  
  const data = await response.json();
  
  // Handle both array response (search) and object response (list)
  if (Array.isArray(data)) {
    return { products: data, totalCount: data.length };
  }
  return data;
}

export async function getProductById(id: string): Promise<Product> {
  const response = await fetch(`${API_BASE}/products/${id}`);
  if (!response.ok) throw new Error('Product not found');
  return response.json();
}

export async function getCategories(): Promise<string[]> {
  const response = await fetch(`${API_BASE}/categories`);
  if (!response.ok) throw new Error('Failed to fetch categories');
  return response.json();
}

export async function semanticSearch(query: string, limit?: number): Promise<SearchResponse> {
  const response = await fetch(`${API_BASE}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, limit: limit ?? 20 }),
  });
  if (!response.ok) throw new Error('Search failed');
  return response.json();
}
