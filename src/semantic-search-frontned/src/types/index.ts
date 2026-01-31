export interface Product {
  id: number;
  name: string;
  description: string;
  price: number;
  category: string;
  imageUrl: string;
  colors: string[];
  sizes: string[];
  rating: number;
  reviewCount: number;
}

export interface SearchResponse {
  products: Product[];
  query: string;
  totalResults: number;
}
