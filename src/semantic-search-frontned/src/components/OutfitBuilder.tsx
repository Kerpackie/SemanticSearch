import { useState } from 'react';
import { Sparkles, Lightbulb } from 'lucide-react';
import { OutfitSlot } from './OutfitSlot';
import type { OutfitSlots, SlotType } from '../types/outfit';
import './OutfitBuilder.css';

interface OutfitBuilderProps {
  slots: OutfitSlots;
  loading?: boolean;
  onProductClick?: (productId: string, slotType: SlotType) => void;
}

export function OutfitBuilder({ slots, loading, onProductClick }: OutfitBuilderProps) {
  const [selectedSlot, setSelectedSlot] = useState<SlotType | null>(null);

  // Organize slots in a specific order for the outfit builder
  const slotOrder: SlotType[] = [
    'upper_body',
    'lower_body', 
    'full_body',
    'shoes',
    'accessories',
    'underwear',
    'swimwear'
  ];

  const availableSlots = slots ? slotOrder.filter(slotType => 
    slots[slotType] && slots[slotType].recommendations && slots[slotType].recommendations.length > 0
  ) : [];

  if (loading) {
    return (
      <div className="outfit-builder-loading">
        <div className="loading-spinner"></div>
        <p>Building your personalised outfit...</p>
      </div>
    );
  }

  if (availableSlots.length === 0) {
    return (
      <div className="outfit-builder-empty">
        <p>No recommendations found. Try a different search query.</p>
      </div>
    );
  }

  return (
    <div className="outfit-builder">
      <div className="outfit-builder-header">
        <h2 className="outfit-builder-title">
          <Sparkles size={22} strokeWidth={1.75} className="outfit-builder-title-icon" />
          Your Personalised Outfit
        </h2>
        <p className="outfit-builder-subtitle">
          We've categorised {availableSlots.length} style {availableSlots.length === 1 ? 'slot' : 'slots'} for you.
          Scroll through each to find your perfect match!
        </p>
      </div>

      <div className="outfit-slots-container">
        {availableSlots.map((slotType) => (
          <OutfitSlot
            key={slotType}
            slotType={slotType}
            slotData={slots[slotType]}
            isExpanded={selectedSlot === slotType}
            onToggle={() => setSelectedSlot(selectedSlot === slotType ? null : slotType)}
            onProductClick={(productId) => onProductClick?.(productId, slotType)}
          />
        ))}
      </div>

      <div className="outfit-builder-footer">
        <Lightbulb size={16} strokeWidth={1.75} className="outfit-builder-hint-icon" />
        <p className="outfit-builder-hint">
          <strong>Tip:</strong> Each slot shows the top 10 re-ranked items based on your preferences and the search query.
        </p>
      </div>
    </div>
  );
}
