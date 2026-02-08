import { useState, useRef, useEffect } from 'react';
import { RecommendationCard } from './RecommendationCard';
import type { Recommendation } from '../types/outfit';
import './SlotCarousel.css';

interface SlotCarouselProps {
  recommendations: Recommendation[];
  onProductClick?: (productId: string) => void;
}

export function SlotCarousel({ recommendations, onProductClick }: SlotCarouselProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [startX, setStartX] = useState(0);
  const [scrollLeft, setScrollLeft] = useState(0);
  const carouselRef = useRef<HTMLDivElement>(null);

  console.log('[SlotCarousel] Rendering with', recommendations.length, 'recommendations');
  if (recommendations.length > 0) {
    console.log('[SlotCarousel] First recommendation:', recommendations[0]);
    console.log('[SlotCarousel] All recommendation IDs:', recommendations.map(r => r.id));
  }

  const visibleCards = 3; // Show 3 cards at a time on desktop
  const maxIndex = Math.max(0, recommendations.length - visibleCards);

  const handlePrevious = () => {
    setCurrentIndex(prev => Math.max(0, prev - 1));
  };

  const handleNext = () => {
    setCurrentIndex(prev => Math.min(maxIndex, prev + 1));
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setStartX(e.pageX - (carouselRef.current?.offsetLeft || 0));
    setScrollLeft(carouselRef.current?.scrollLeft || 0);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    e.preventDefault();
    const x = e.pageX - (carouselRef.current?.offsetLeft || 0);
    const walk = (x - startX) * 2;
    if (carouselRef.current) {
      carouselRef.current.scrollLeft = scrollLeft - walk;
    }
  };

  useEffect(() => {
    if (carouselRef.current) {
      const cardWidth = carouselRef.current.querySelector('.recommendation-card')?.clientWidth || 0;
      const gap = 16; // Match CSS gap
      carouselRef.current.scrollTo({
        left: currentIndex * (cardWidth + gap),
        behavior: 'smooth'
      });
    }
  }, [currentIndex]);

  return (
    <div className="slot-carousel">
      <button 
        className="carousel-nav carousel-nav-prev"
        onClick={handlePrevious}
        disabled={currentIndex === 0}
        aria-label="Previous items"
      >
        ‹
      </button>

      <div 
        className="carousel-track"
        ref={carouselRef}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onMouseMove={handleMouseMove}
      >
        {recommendations.map((recommendation, index) => (
          <RecommendationCard
            key={recommendation.id}
            recommendation={recommendation}
            rank={index + 1}
            onClick={() => onProductClick?.(recommendation.id)}
          />
        ))}
      </div>

      <button 
        className="carousel-nav carousel-nav-next"
        onClick={handleNext}
        disabled={currentIndex >= maxIndex}
        aria-label="Next items"
      >
        ›
      </button>

      <div className="carousel-indicators">
        {Array.from({ length: recommendations.length }, (_, i) => (
          <button
            key={i}
            className={`carousel-indicator ${i >= currentIndex && i < currentIndex + visibleCards ? 'active' : ''}`}
            onClick={() => setCurrentIndex(Math.min(i, maxIndex))}
            aria-label={`Go to item ${i + 1}`}
          />
        ))}
      </div>
    </div>
  );
}
