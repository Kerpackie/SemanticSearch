import { Product } from '../types';
import './ProductModal.css';

interface ProductModalProps {
  product: Product | null;
  onClose: () => void;
}

export function ProductModal({ product, onClose }: ProductModalProps) {
  if (!product) return null;

  const renderStars = (rating: number) => {
    const stars = [];
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating % 1 >= 0.5;
    
    for (let i = 0; i < fullStars; i++) {
      stars.push(<span key={i} className="star filled">★</span>);
    }
    if (hasHalfStar) {
      stars.push(<span key="half" className="star half">★</span>);
    }
    for (let i = stars.length; i < 5; i++) {
      stars.push(<span key={i} className="star">★</span>);
    }
    return stars;
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>✕</button>
        
        <div className="modal-body">
          <div className="modal-image-section">
            <img src={product.imageUrl} alt={product.name} className="modal-image" />
          </div>
          
          <div className="modal-details">
            <span className="modal-category">{product.category}</span>
            <h2 className="modal-title">{product.name}</h2>
            
            <div className="modal-rating">
              <span className="stars">{renderStars(product.rating)}</span>
              <span className="rating-text">{product.rating} ({product.reviewCount} reviews)</span>
            </div>
            
            <p className="modal-description">{product.description}</p>
            
            <div className="modal-price">${product.price.toFixed(2)}</div>
            
            <div className="modal-option">
              <label>Color</label>
              <div className="option-buttons">
                {product.colors.map((color, i) => (
                  <button key={i} className={`option-button ${i === 0 ? 'selected' : ''}`}>
                    {color}
                  </button>
                ))}
              </div>
            </div>
            
            <div className="modal-option">
              <label>Size</label>
              <div className="option-buttons">
                {product.sizes.map((size, i) => (
                  <button key={i} className={`option-button ${i === 0 ? 'selected' : ''}`}>
                    {size}
                  </button>
                ))}
              </div>
            </div>
            
            <div className="modal-actions">
              <button className="add-to-cart-button">Add to Cart</button>
              <button className="wishlist-button">❤️</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
