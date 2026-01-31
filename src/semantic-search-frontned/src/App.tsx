import { useState, useEffect, useCallback } from 'react';
import { Header } from './components/Header';
import { SearchBar } from './components/SearchBar';
import { CategoryFilter } from './components/CategoryFilter';
import { ProductGrid } from './components/ProductGrid';
import { ProductModal } from './components/ProductModal';
import { getProducts, getCategories, semanticSearch } from './api/products';
import type {Product, SearchResult} from './types';
import './App.css';

// Convert search result to product format for display
function searchResultToProduct(result: SearchResult): Product {
  return {
    articleId: result.id,
    productCode: 0,
    name: result.name,
    description: result.description,
    productType: result.productType,
    productGroupName: result.productGroup,
    colourGroupName: result.colour,
    colourMasterName: result.colour,
    graphicalAppearance: '',
    department: '',
    indexName: '',
    indexGroupName: '',
    section: '',
    garmentGroup: '',
  };
}

function App() {
  const [products, setProducts] = useState<Product[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [categories, setCategories] = useState<string[]>(['All']);
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [searchQuery, setSearchQuery] = useState('');
  const [processedQuery, setProcessedQuery] = useState('');
  const [loading, setLoading] = useState(true);
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const pageSize = 24;

  // Load categories on mount
  useEffect(() => {
    getCategories()
      .then(setCategories)
      .catch(console.error);
  }, []);

  // Load products when category or page changes
  useEffect(() => {
    if (searchQuery) return; // Don't fetch if there's a search query
    
    setLoading(true);
    setError(null);
    
    getProducts(selectedCategory, undefined, page, pageSize)
      .then((response) => {
        setProducts(response.products);
        setTotalCount(response.totalCount);
        setProcessedQuery('');
      })
      .catch((err) => {
        setError('Failed to load products. Make sure the BFF server is running.');
        console.error(err);
      })
      .finally(() => setLoading(false));
  }, [selectedCategory, page, searchQuery]);

  const handleSearch = useCallback(async (query: string) => {
    setSearchQuery(query);
    setLoading(true);
    setError(null);
    setPage(1);
    
    try {
      if (query) {
        const response = await semanticSearch(query, 50);
        // Convert search results to product format
        const searchProducts = response.products.map(searchResultToProduct);
        setProducts(searchProducts);
        setTotalCount(response.totalResults);
        setProcessedQuery(response.processedQuery);
      } else {
        const response = await getProducts(selectedCategory, undefined, 1, pageSize);
        setProducts(response.products);
        setTotalCount(response.totalCount);
        setProcessedQuery('');
      }
    } catch (err) {
      setError('Search failed. Make sure the BFF server is running.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [selectedCategory]);

  const handleCategoryChange = useCallback((category: string) => {
    setSelectedCategory(category);
    setSearchQuery(''); // Clear search when changing category
    setPage(1);
  }, []);

  const totalPages = Math.ceil(totalCount / pageSize);

  return (
    <div className="app">
      <Header cartCount={0} />
      
      <main className="main-content">
        <section className="hero-section">
          <h1 className="hero-title">H&M Fashion Discovery</h1>
          <p className="hero-subtitle">
            Find the perfect outfit with our intelligent semantic search
          </p>
          <SearchBar 
            onSearch={handleSearch}
            placeholder="Try 'casual summer dress' or 'warm winter jacket'..."
          />
        </section>

        <section className="products-section">
          <div className="section-header">
            <h2 className="section-title">
              {searchQuery 
                ? `Search results for "${searchQuery}"${processedQuery && processedQuery !== searchQuery ? ` (interpreted as: "${processedQuery}")` : ''}` 
                : selectedCategory === 'All' 
                  ? 'All Products' 
                  : selectedCategory}
            </h2>
            <span className="product-count">
              {totalCount.toLocaleString()} {totalCount === 1 ? 'item' : 'items'}
              {!searchQuery && totalPages > 1 && ` • Page ${page} of ${totalPages}`}
            </span>
          </div>
          
          <CategoryFilter
            categories={categories}
            selectedCategory={selectedCategory}
            onCategoryChange={handleCategoryChange}
          />

          {error && (
            <div className="error-banner">
              <span>⚠️</span>
              {error}
            </div>
          )}
          
          <ProductGrid 
            products={products} 
            loading={loading}
            onProductClick={setSelectedProduct}
          />

          {/* Pagination - only show when not searching */}
          {!searchQuery && totalPages > 1 && (
            <div className="pagination">
              <button 
                className="pagination-button"
                disabled={page === 1}
                onClick={() => setPage(p => Math.max(1, p - 1))}
              >
                ← Previous
              </button>
              <span className="pagination-info">
                Page {page} of {totalPages}
              </span>
              <button 
                className="pagination-button"
                disabled={page === totalPages}
                onClick={() => setPage(p => Math.min(totalPages, p + 1))}
              >
                Next →
              </button>
            </div>
          )}
        </section>
      </main>

      <ProductModal 
        product={selectedProduct} 
        onClose={() => setSelectedProduct(null)} 
      />
    </div>
  );
}

export default App;
