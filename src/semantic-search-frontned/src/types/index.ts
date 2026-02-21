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
// H&M dataset stores images in subdirectories based on the first 3 digits of the article ID
// e.g. article 0108775015 → /images/010/0108775015.jpg
export function getProductImageUrl(articleId: string): string {
  const prefix = articleId.substring(0, 3);
  return `/images/${prefix}/${articleId}.jpg`;
}

// Fallback placeholder image (inline SVG – no external dependency)
export const PLACEHOLDER_IMAGE =
  "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='500' viewBox='0 0 400 500'%3E%3Crect width='400' height='500' fill='%23f0f0f0'/%3E%3Ctext x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' font-family='sans-serif' font-size='16' fill='%23aaa'%3ENo Image%3C/text%3E%3C/svg%3E";
