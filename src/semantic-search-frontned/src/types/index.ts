// Product from BFF API - matches H&M data structure
export interface Product {
  articleId: string;
  productCode: number;
  name: string;
  description: string;
  productType: string;
  productGroupName: string;
  colourGroupName: string;
  colourMasterName: string;
  graphicalAppearance: string;
  department: string;
  indexName: string;
  indexGroupName: string;
  section: string;
  garmentGroup: string;
}

// Product list response with pagination info
export interface ProductListResponse {
  products: Product[];
  totalCount: number;
}

// Semantic search result
export interface SearchResult {
  id: string;
  name: string;
  description: string;
  score: number;
  rank: number;
  productGroup: string;
  colour: string;
  productType: string;
}

export interface SearchResponse {
  products: SearchResult[];
  processedQuery: string;
  totalResults: number;
}

// Helper to generate image URL from article ID
// Images should be placed in public/images/{article_id}.jpg
export function getProductImageUrl(articleId: string): string {
  return `/images/${articleId}.jpg`;
}

// Fallback placeholder image
export const PLACEHOLDER_IMAGE = 'https://via.placeholder.com/400x500?text=No+Image';
