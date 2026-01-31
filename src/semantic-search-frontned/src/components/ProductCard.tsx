import {type Product, getProductImageUrl, PLACEHOLDER_IMAGE } from '../types';
import './ProductCard.css';
import { useState } from 'react';

interface ProductCardProps {
  product: Product;
  onClick?: (product: Product) => void;
}

export function ProductCard({ product, onClick }: ProductCardProps) {
  const [imageError, setImageError] = useState(false);
  const imageUrl = imageError ? PLACEHOLDER_IMAGE : getProductImageUrl(product.articleId);

  return (
    <div className="product-card" onClick={() => onClick?.(product)}>
      <div className="product-image-container">
        <img 
          src={imageUrl} 
          alt={product.name}
          className="product-image"
          loading="lazy"
          onError={() => setImageError(true)}
        />
        <div className="product-category-badge">{product.productGroupName}</div>
      </div>
      <div className="product-info">
        <h3 className="product-name">{product.name}</h3>
        <p className="product-description">{product.description || 'No description available'}</p>
        <div className="product-meta">
          <span className="product-type">{product.productType}</span>
          <span className="product-colour">{product.colourGroupName}</span>
        </div>
        <div className="product-details">
          <span className="product-department">{product.department}</span>
          <span className="product-section">{product.section}</span>
        </div>
      </div>
    </div>
  );
}
