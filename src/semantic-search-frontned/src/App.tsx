import { useState, useEffect, useCallback } from 'react';
import { LayoutGrid, Shirt, AlertTriangle, ChevronLeft, ChevronRight, Sparkles } from 'lucide-react';
import { Header } from './components/Header';
import { SearchBar } from './components/SearchBar';
import { CategoryFilter } from './components/CategoryFilter';
import { ProductGrid } from './components/ProductGrid';
import { ProductModal } from './components/ProductModal';
import { OutfitBuilder } from './components/OutfitBuilder';
import { UserSelector, TEST_USERS, type User } from './components/UserSelector';
import { getProducts, getCategories, semanticSearch, outfitSearch } from './api/products';
import type {Product, SearchResult} from './types';
import type { OutfitSlots } from './types/outfit';
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
  const [currentUser, setCurrentUser] = useState<User | null>(TEST_USERS[0]); // Start with Guest
  const [viewMode, setViewMode] = useState<'grid' | 'outfit'>('grid');
  const [outfitSlots, setOutfitSlots] = useState<OutfitSlots | null>(null);
  const [outfitName, setOutfitName] = useState<string | undefined>();
  const [styleDescription, setStyleDescription] = useState<string | undefined>();
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
        const customerId = currentUser?.id || undefined;
        
        if (viewMode === 'outfit') {
          // Use outfit search endpoint
          const response = await outfitSearch(query, customerId);
          setOutfitSlots(response.slots);
          setOutfitName(response.outfitName);
          setStyleDescription(response.styleDescription);
          setProcessedQuery(response.processedQuery);
          setTotalCount(response.totalResults);
          setProducts([]); // Clear grid products
        } else {
          // Use regular semantic search
          const response = await semanticSearch(query, 50, customerId);
          const searchProducts = response.products.map(searchResultToProduct);
          setProducts(searchProducts);
          setTotalCount(response.totalResults);
          setProcessedQuery(response.processedQuery);
          setOutfitSlots(null); // Clear outfit slots
        }
      } else {
        const response = await getProducts(selectedCategory, undefined, 1, pageSize);
        setProducts(response.products);
        setTotalCount(response.totalCount);
        setProcessedQuery('');
        setOutfitSlots(null);
      }
    } catch (err) {
      setError('Search failed. Make sure the BFF server is running.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [selectedCategory, currentUser, viewMode]);

  const handleCategoryChange = useCallback((category: string) => {
    setSelectedCategory(category);
    setSearchQuery(''); // Clear search when changing category
    setPage(1);
  }, []);

  const handleUserChange = useCallback((user: User) => {
    setCurrentUser(user);
    // Re-run search if there's an active search query
    if (searchQuery) {
      handleSearch(searchQuery);
    }
  }, [searchQuery, handleSearch]);

  const totalPages = Math.ceil(totalCount / pageSize);

  return (
    <div className="app">
      <Header cartCount={0} />
      
      <main className="main-content">
        <section className="hero-section">
          <div className="hero-header">
            <div>
              <h1 className="hero-title">H&M Fashion Discovery</h1>
              <p className="hero-subtitle">
                Find the perfect outfit with our intelligent semantic search
                {currentUser?.id && (
                  <span className="personalization-badge">
                    <Sparkles size={12} strokeWidth={2} />
                    Personalised for {currentUser.name}
                  </span>
                )}
              </p>
            </div>
            <UserSelector currentUser={currentUser} onUserChange={handleUserChange} />
          </div>
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
            <div className="section-controls">
              {searchQuery && (
                <div className="view-mode-toggle">
                  <button 
                    className={`toggle-btn ${viewMode === 'grid' ? 'active' : ''}`}
                    onClick={async () => {
                      if (viewMode !== 'grid') {
                        setViewMode('grid');
                        setLoading(true);
                        setError(null);
                        try {
                          const customerId = currentUser?.id || undefined;
                          const response = await semanticSearch(searchQuery, 50, customerId);
                          const searchProducts = response.products.map(searchResultToProduct);
                          setProducts(searchProducts);
                          setTotalCount(response.totalResults);
                          setProcessedQuery(response.processedQuery);
                          setOutfitSlots(null);
                          setOutfitName(undefined);
                          setStyleDescription(undefined);
                        } catch (err) {
                          setError('Search failed. Make sure the BFF server is running.');
                          console.error(err);
                        } finally {
                          setLoading(false);
                        }
                      }
                    }}
                  >
                    <LayoutGrid size={15} strokeWidth={2} />
                    Grid View
                  </button>
                  <button 
                    className={`toggle-btn ${viewMode === 'outfit' ? 'active' : ''}`}
                    onClick={async () => {
                      if (viewMode !== 'outfit') {
                        setViewMode('outfit');
                        setLoading(true);
                        setError(null);
                        try {
                          const customerId = currentUser?.id || undefined;
                          const response = await outfitSearch(searchQuery, customerId);
                          setOutfitSlots(response.slots);
                          setOutfitName(response.outfitName);
                          setStyleDescription(response.styleDescription);
                          setProcessedQuery(response.processedQuery);
                          setTotalCount(response.totalResults);
                          setProducts([]);
                        } catch (err) {
                          setError('Outfit search failed. Make sure the BFF server is running.');
                          console.error(err);
                        } finally {
                          setLoading(false);
                        }
                      }
                    }}
                  >
                    <Shirt size={15} strokeWidth={2} />
                    Outfit Builder
                  </button>
                </div>
              )}
              <span className="product-count">
                {totalCount.toLocaleString()} {totalCount === 1 ? 'item' : 'items'}
                {!searchQuery && totalPages > 1 && ` • Page ${page} of ${totalPages}`}
              </span>
            </div>
          </div>
          
          {!searchQuery && (
            <CategoryFilter
              categories={categories}
              selectedCategory={selectedCategory}
              onCategoryChange={handleCategoryChange}
            />
          )}

          {error && (
            <div className="error-banner">
              <AlertTriangle size={16} strokeWidth={2} />
              {error}
            </div>
          )}
          
          {viewMode === 'outfit' && searchQuery ? (
            loading ? (
              <div className="outfit-builder-loading">
                <div className="loading-spinner"></div>
                <p>Building your personalised outfit...</p>
              </div>
            ) : outfitSlots && Object.keys(outfitSlots).length > 0 ? (
              <OutfitBuilder 
                slots={outfitSlots}
                loading={false}
                outfitName={outfitName}
                styleDescription={styleDescription}
                onProductClick={() => {}}
              />
            ) : (
              <div className="outfit-builder-empty">
                <p>No outfit recommendations found. Try a different search query.</p>
              </div>
            )
          ) : (
            <>
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
                    <ChevronLeft size={16} strokeWidth={2} />
                    Previous
                  </button>
                  <span className="pagination-info">
                    Page {page} of {totalPages}
                  </span>
                  <button 
                    className="pagination-button"
                    disabled={page === totalPages}
                    onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                  >
                    Next
                    <ChevronRight size={16} strokeWidth={2} />
                  </button>
                </div>
              )}
            </>
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
