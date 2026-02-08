import { SlotCarousel } from './SlotCarousel';
import type { SlotData, SlotType } from '../types/outfit';
import './OutfitSlot.css';

interface OutfitSlotProps {
  slotType: SlotType;
  slotData: SlotData;
  isExpanded: boolean;
  onToggle: () => void;
  onProductClick?: (productId: string) => void;
}

const SLOT_INFO: Record<SlotType, { icon: string; title: string; description: string }> = {
  upper_body: {
    icon: '👕',
    title: 'Upper Body',
    description: 'Tops, shirts, jackets, and sweaters'
  },
  lower_body: {
    icon: '👖',
    title: 'Lower Body',
    description: 'Pants, jeans, skirts, and shorts'
  },
  full_body: {
    icon: '👗',
    title: 'Full Body',
    description: 'Dresses, jumpsuits, and full outfits'
  },
  shoes: {
    icon: '👟',
    title: 'Footwear',
    description: 'Shoes, boots, sneakers, and sandals'
  },
  accessories: {
    icon: '👜',
    title: 'Accessories',
    description: 'Bags, belts, jewelry, and more'
  },
  underwear: {
    icon: '🩲',
    title: 'Underwear',
    description: 'Undergarments and intimate wear'
  },
  swimwear: {
    icon: '🩱',
    title: 'Swimwear',
    description: 'Swimsuits, bikinis, and beachwear'
  }
};

export function OutfitSlot({ slotType, slotData, isExpanded, onToggle, onProductClick }: OutfitSlotProps) {
  const info = SLOT_INFO[slotType];
  const itemCount = slotData.recommendations.length;

  return (
    <div className={`outfit-slot ${isExpanded ? 'expanded' : 'collapsed'}`}>
      <div className="slot-header" onClick={onToggle}>
        <div className="slot-header-left">
          <span className="slot-icon">{info.icon}</span>
          <div className="slot-info">
            <h3 className="slot-title">{info.title}</h3>
            <p className="slot-description">{info.description}</p>
          </div>
        </div>
        <div className="slot-header-right">
          <span className="slot-count">{itemCount} {itemCount === 1 ? 'item' : 'items'}</span>
          <span className={`slot-toggle-icon ${isExpanded ? 'expanded' : ''}`}>▼</span>
        </div>
      </div>

      {isExpanded && (
        <div className="slot-content">
          {slotData.reasoning && (
            <div className="slot-reasoning">
              <strong>Why this slot?</strong> {slotData.reasoning}
            </div>
          )}
          <SlotCarousel 
            recommendations={slotData.recommendations}
            onProductClick={onProductClick}
          />
        </div>
      )}

      {!isExpanded && (
        <div className="slot-preview">
          <div className="slot-preview-images">
            {slotData.recommendations.slice(0, 4).map((rec, idx) => (
              <div key={rec.id} className="slot-preview-image" style={{ zIndex: 4 - idx }}>
                <img 
                  src={`/images/${rec.id}.jpg`} 
                  alt={rec.name}
                  onError={(e) => {
                    e.currentTarget.src = 'https://via.placeholder.com/100x120?text=No+Image';
                  }}
                />
              </div>
            ))}
            {itemCount > 4 && (
              <div className="slot-preview-more">
                +{itemCount - 4}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
