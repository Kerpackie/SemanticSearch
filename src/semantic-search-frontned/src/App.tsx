import { useState, useEffect, useCallback } from 'react';
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
                    ✨ Personalized for {currentUser.name}
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
                        } catch (err) {
                          setError('Search failed. Make sure the BFF server is running.');
                          console.error(err);
                        } finally {
                          setLoading(false);
                        }
                      }
                    }}
                  >
                    📋 Grid View
                  </button>
                  <button 
                    className={`toggle-btn ${viewMode === 'outfit' ? 'active' : ''}`}
                    onClick={async () => {
                      if (viewMode !== 'outfit') {
                        console.log('🎨 [OUTFIT BUILDER] Button clicked!');
                        console.log('🎨 [OUTFIT BUILDER] Current viewMode:', viewMode);
                        console.log('🎨 [OUTFIT BUILDER] Search query:', searchQuery);
                        console.log('🎨 [OUTFIT BUILDER] Customer ID:', currentUser?.id);
                        
                        setViewMode('outfit');
                        setLoading(true);
                        setError(null);
                        
                        try {
                          const customerId = currentUser?.id || undefined;
                          console.log('🎨 [OUTFIT BUILDER] Calling outfitSearch API...');
                          console.log('🎨 [OUTFIT BUILDER] Request params:', { query: searchQuery, customerId });
                          
                          const response = await outfitSearch(searchQuery, customerId);
                          
                          console.log('🎨 [OUTFIT BUILDER] API Response received:');
                          console.log('🎨 [OUTFIT BUILDER] - Total results:', response.totalResults);
                          console.log('🎨 [OUTFIT BUILDER] - Processed query:', response.processedQuery);
                          console.log('🎨 [OUTFIT BUILDER] - Slots:', response.slots);
                          console.log('🎨 [OUTFIT BUILDER] - Number of slots:', Object.keys(response.slots).length);
                          
                          if (Object.keys(response.slots).length === 0) {
                            console.warn('⚠️ [OUTFIT BUILDER] WARNING: No slots returned!');
                          } else {
                            Object.entries(response.slots).forEach(([slotName, slotData]) => {
                              console.log(`🎨 [OUTFIT BUILDER]   - ${slotName}: ${slotData.recommendations.length} items`);
                            });
                          }
                          
                          setOutfitSlots(response.slots);
                          setProcessedQuery(response.processedQuery);
                          setTotalCount(response.totalResults);
                          setProducts([]);
                          
                          console.log('🎨 [OUTFIT BUILDER] State updated successfully');
                        } catch (err) {
                          console.error('❌ [OUTFIT BUILDER] Error:', err);
                          setError('Outfit search failed. Make sure the BFF server is running.');
                          console.error(err);
                        } finally {
                          setLoading(false);
                          console.log('🎨 [OUTFIT BUILDER] Loading complete');
                        }
                      } else {
                        console.log('🎨 [OUTFIT BUILDER] Already in outfit mode, skipping');
                      }
                    }}
                  >
                    👔 Outfit Builder
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
              <span>⚠️</span>
              {error}
            </div>
          )}
          
          {viewMode === 'outfit' && searchQuery ? (
            (() => {
              console.log('[App] Outfit mode rendering decision:', {
                viewMode,
                searchQuery,
                loading,
                outfitSlots,
                outfitSlotsKeys: outfitSlots ? Object.keys(outfitSlots) : null
              });
              
              if (loading) {
                console.log('[App] -> Showing loading state');
                return (
                  <div className="outfit-builder-loading">
                    <div className="loading-spinner"></div>
                    <p>Building your personalized outfit...</p>
                  </div>
                );
              } else if (outfitSlots && Object.keys(outfitSlots).length > 0) {
                console.log('[App] -> Rendering OutfitBuilder with slots');
                return (
                  <OutfitBuilder 
                    slots={outfitSlots}
                    loading={false}
                    onProductClick={(productId) => {
                      console.log('Product clicked:', productId);
                    }}
                  />
                );
              } else {
                console.log('[App] -> Showing empty state');
                return (
                  <div className="outfit-builder-empty">
                    <p>No outfit recommendations found. Try a different search query.</p>
                  </div>
                );
              }
            })()
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
