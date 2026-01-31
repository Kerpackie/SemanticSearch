import { useState, useEffect, useCallback } from 'react';
import { Header } from './components/Header';
import { SearchBar } from './components/SearchBar';
import { CategoryFilter } from './components/CategoryFilter';
import { ProductGrid } from './components/ProductGrid';
import { ProductModal } from './components/ProductModal';
import { getProducts, getCategories, semanticSearch } from './api/products';
import { Product } from './types';
import './App.css';

function App() {
  const [products, setProducts] = useState<Product[]>([]);
  const [categories, setCategories] = useState<string[]>(['All']);
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(true);
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Load categories on mount
  useEffect(() => {
    getCategories()
      .then(setCategories)
      .catch(console.error);
  }, []);

  // Load products when category changes
  useEffect(() => {
    if (searchQuery) return; // Don't fetch if there's a search query
    
    setLoading(true);
    setError(null);
    
    getProducts(selectedCategory)
      .then(setProducts)
      .catch((err) => {
        setError('Failed to load products. Make sure the BFF server is running.');
        console.error(err);
      })
      .finally(() => setLoading(false));
  }, [selectedCategory, searchQuery]);

  const handleSearch = useCallback(async (query: string) => {
    setSearchQuery(query);
    setLoading(true);
    setError(null);
    
    try {
      if (query) {
        const response = await semanticSearch(query);
        setProducts(response.products);
      } else {
        const prods = await getProducts(selectedCategory);
        setProducts(prods);
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
  }, []);

  return (
    <div className="app">
      <Header cartCount={0} />
      
      <main className="main-content">
        <section className="hero-section">
          <h1 className="hero-title">Discover Your Style</h1>
          <p className="hero-subtitle">
            Find the perfect outfit with our intelligent search
          </p>
          <SearchBar 
            onSearch={handleSearch}
            placeholder="Try 'casual summer outfit' or 'formal business wear'..."
          />
        </section>

        <section className="products-section">
          <div className="section-header">
            <h2 className="section-title">
              {searchQuery 
                ? `Search results for "${searchQuery}"` 
                : selectedCategory === 'All' 
                  ? 'All Products' 
                  : selectedCategory}
            </h2>
            <span className="product-count">
              {products.length} {products.length === 1 ? 'item' : 'items'}
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
