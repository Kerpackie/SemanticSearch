import { Product, SearchResponse } from '../types';

const API_BASE = '/api';

export async function getProducts(category?: string, search?: string): Promise<Product[]> {
  const params = new URLSearchParams();
  if (category && category !== 'All') params.append('category', category);
  if (search) params.append('search', search);
  
  const url = `${API_BASE}/products${params.toString() ? '?' + params.toString() : ''}`;
  const response = await fetch(url);
  if (!response.ok) throw new Error('Failed to fetch products');
  return response.json();
}

export async function getProductById(id: number): Promise<Product> {
  const response = await fetch(`${API_BASE}/products/${id}`);
  if (!response.ok) throw new Error('Product not found');
  return response.json();
}

export async function getCategories(): Promise<string[]> {
  const response = await fetch(`${API_BASE}/categories`);
  if (!response.ok) throw new Error('Failed to fetch categories');
  return response.json();
}

export async function semanticSearch(query: string): Promise<SearchResponse> {
  const response = await fetch(`${API_BASE}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  });
  if (!response.ok) throw new Error('Search failed');
  return response.json();
}
