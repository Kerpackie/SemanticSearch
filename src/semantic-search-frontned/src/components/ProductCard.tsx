import { Product } from '../types';
import './ProductCard.css';

interface ProductCardProps {
  product: Product;
  onClick?: (product: Product) => void;
}

export function ProductCard({ product, onClick }: ProductCardProps) {
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
    <div className="product-card" onClick={() => onClick?.(product)}>
      <div className="product-image-container">
        <img 
          src={product.imageUrl} 
          alt={product.name}
          className="product-image"
          loading="lazy"
        />
        <div className="product-category-badge">{product.category}</div>
      </div>
      <div className="product-info">
        <h3 className="product-name">{product.name}</h3>
        <p className="product-description">{product.description}</p>
        <div className="product-rating">
          <span className="stars">{renderStars(product.rating)}</span>
          <span className="review-count">({product.reviewCount})</span>
        </div>
        <div className="product-details">
          <span className="product-price">${product.price.toFixed(2)}</span>
          <span className="product-colors">{product.colors.length} colors</span>
        </div>
      </div>
    </div>
  );
}
