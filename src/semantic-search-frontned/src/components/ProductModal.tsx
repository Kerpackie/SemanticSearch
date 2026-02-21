import { X, Heart, ShoppingCart } from 'lucide-react';
import {type Product, getProductImageUrl, PLACEHOLDER_IMAGE } from '../types';
import './ProductModal.css';
import { useState } from 'react';

interface ProductModalProps {
  product: Product | null;
  onClose: () => void;
}

export function ProductModal({ product, onClose }: ProductModalProps) {
  const [imageError, setImageError] = useState(false);
  
  if (!product) return null;

  const imageUrl = imageError ? PLACEHOLDER_IMAGE : getProductImageUrl(product.articleId);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose} aria-label="Close"><X size={20} strokeWidth={2} /></button>
        
        <div className="modal-body">
          <div className="modal-image-section">
            <img 
              src={imageUrl} 
              alt={product.name} 
              className="modal-image"
              onError={() => setImageError(true)}
            />
          </div>
          
          <div className="modal-details">
            <span className="modal-category">{product.productGroupName}</span>
            <h2 className="modal-title">{product.name}</h2>
            
            <div className="modal-meta-info">
              <span className="meta-badge product-type">{product.productType}</span>
              <span className="meta-badge garment-group">{product.garmentGroup}</span>
            </div>
            
            <p className="modal-description">{product.description || 'No description available'}</p>
            
            <div className="modal-attributes">
              <div className="attribute-row">
                <label>Colour</label>
                <span>{product.colourGroupName} ({product.colourMasterName})</span>
              </div>
              <div className="attribute-row">
                <label>Appearance</label>
                <span>{product.graphicalAppearance}</span>
              </div>
              <div className="attribute-row">
                <label>Department</label>
                <span>{product.department}</span>
              </div>
              <div className="attribute-row">
                <label>Section</label>
                <span>{product.section}</span>
              </div>
              <div className="attribute-row">
                <label>Index</label>
                <span>{product.indexName} - {product.indexGroupName}</span>
              </div>
              <div className="attribute-row">
                <label>Article ID</label>
                <span className="article-id">{product.articleId}</span>
              </div>
            </div>
            
            <div className="modal-actions">
              <button className="add-to-cart-button">
                <ShoppingCart size={18} strokeWidth={2} />
                Add to Cart
              </button>
              <button className="wishlist-button" aria-label="Add to wishlist">
                <Heart size={18} strokeWidth={1.75} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
