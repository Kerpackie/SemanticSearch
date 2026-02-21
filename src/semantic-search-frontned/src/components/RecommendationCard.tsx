import { Lightbulb, Palette, Tag } from 'lucide-react';
import type { Recommendation } from '../types/outfit';
import { getProductImageUrl, PLACEHOLDER_IMAGE } from '../types';
import './RecommendationCard.css';

interface RecommendationCardProps {
  recommendation: Recommendation;
  rank: number;
  onClick?: () => void;
}

export function RecommendationCard({ recommendation, rank, onClick }: RecommendationCardProps) {
  const { id, name, description, score, reasoning, metadata } = recommendation;

  return (
    <div className="recommendation-card" onClick={onClick}>
      <div className="recommendation-rank">
        <span className="rank-badge">#{rank}</span>
        <span className="score-badge">{(score * 100).toFixed(0)}%</span>
      </div>

      <div className="recommendation-image-container">
        <img 
          src={getProductImageUrl(id)}
          alt={name}
          className="recommendation-image"
          onError={(e) => {
            e.currentTarget.src = PLACEHOLDER_IMAGE;
          }}
        />
      </div>

      <div className="recommendation-content">
        <h4 className="recommendation-name">{name}</h4>
        {description && (
          <p className="recommendation-description">{description}</p>
        )}

        {reasoning && (
          <div className="recommendation-reasoning">
            <div className="reasoning-header">
              <Lightbulb size={14} strokeWidth={2} className="reasoning-icon" />
              <strong>Why this?</strong>
            </div>
            <p className="reasoning-text">{reasoning}</p>
          </div>
        )}

        {metadata && (
          <div className="recommendation-metadata">
            {metadata.colour && (
              <span className="metadata-tag">
                <Palette size={12} strokeWidth={2} className="tag-icon" />
                {metadata.colour}
              </span>
            )}
            {metadata.productType && (
              <span className="metadata-tag">
                <Tag size={12} strokeWidth={2} className="tag-icon" />
                {metadata.productType}
              </span>
            )}
          </div>
        )}
      </div>

      <div className="recommendation-overlay">
        <button className="view-details-btn">View Details</button>
      </div>
    </div>
  );
}
