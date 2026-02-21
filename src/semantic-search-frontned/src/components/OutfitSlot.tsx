import { Shirt, Footprints, Gem, ChevronDown, Wind, Waves, ShoppingBag } from 'lucide-react';
import { SlotCarousel } from './SlotCarousel';
import type { SlotData, SlotType } from '../types/outfit';
import { getProductImageUrl, PLACEHOLDER_IMAGE } from '../types';
import './OutfitSlot.css';

interface OutfitSlotProps {
  slotType: SlotType;
  slotData: SlotData;
  isExpanded: boolean;
  onToggle: () => void;
  onProductClick?: (productId: string) => void;
}

type SlotIconRecord = Record<SlotType, React.ReactNode>;

const SLOT_ICONS: SlotIconRecord = {
  upper_body: <Shirt size={22} strokeWidth={1.75} />,
  lower_body: <Wind size={22} strokeWidth={1.75} />,
  full_body: <ShoppingBag size={22} strokeWidth={1.75} />,
  shoes: <Footprints size={22} strokeWidth={1.75} />,
  accessories: <Gem size={22} strokeWidth={1.75} />,
  underwear: <Wind size={22} strokeWidth={1.75} />,
  swimwear: <Waves size={22} strokeWidth={1.75} />,
};

const SLOT_INFO: Record<SlotType, { title: string; description: string }> = {
  upper_body: { title: 'Upper Body', description: 'Tops, shirts, jackets & sweaters' },
  lower_body: { title: 'Lower Body', description: 'Pants, jeans, skirts & shorts' },
  full_body:  { title: 'Full Body',  description: 'Dresses, jumpsuits & full outfits' },
  shoes:      { title: 'Footwear',   description: 'Shoes, boots, sneakers & sandals' },
  accessories:{ title: 'Accessories',description: 'Bags, belts, jewellery & more' },
  underwear:  { title: 'Underwear',  description: 'Undergarments & intimate wear' },
  swimwear:   { title: 'Swimwear',   description: 'Swimsuits, bikinis & beachwear' },
};

export function OutfitSlot({ slotType, slotData, isExpanded, onToggle, onProductClick }: OutfitSlotProps) {
  const info = SLOT_INFO[slotType];
  const icon = SLOT_ICONS[slotType];
  const itemCount = slotData.recommendations.length;

  return (
    <div className={`outfit-slot ${isExpanded ? 'expanded' : 'collapsed'}`}>
      <div className="slot-header" onClick={onToggle}>
        <div className="slot-header-left">
          <span className="slot-icon">{icon}</span>
          <div className="slot-info">
            <h3 className="slot-title">{info.title}</h3>
            <p className="slot-description">{info.description}</p>
          </div>
        </div>
        <div className="slot-header-right">
          <span className="slot-count">{itemCount} {itemCount === 1 ? 'item' : 'items'}</span>
          <ChevronDown
            size={18}
            strokeWidth={2}
            className={`slot-toggle-icon ${isExpanded ? 'expanded' : ''}`}
          />
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
                  src={getProductImageUrl(rec.id)}
                  alt={rec.name}
                  onError={(e) => {
                    e.currentTarget.src = PLACEHOLDER_IMAGE;
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
