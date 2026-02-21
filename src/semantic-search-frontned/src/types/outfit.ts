// Outfit Builder Types

export type SlotType = 
  | 'upper_body'
  | 'lower_body'
  | 'full_body'
  | 'shoes'
  | 'accessories'
  | 'underwear'
  | 'swimwear';

export interface Recommendation {
  id: string;
  name: string;
  description: string;
  score: number;
  reasoning?: string; // Why this item was recommended for this slot
  metadata?: {
    colour?: string;
    productType?: string;
    productGroup?: string;
  };
}

export interface SlotData {
  slotType: SlotType;
  recommendations: Recommendation[];
  reasoning?: string; // Why this slot was created
}

export type OutfitSlots = {
  [K in SlotType]: SlotData;
};

export interface OutfitSearchResponse {
  slots: OutfitSlots;
  totalResults: number;
  processedQuery: string;
  outfitName?: string;       // GPT-generated outfit name e.g. "Casual Summer Look"
  styleDescription?: string; // GPT-generated style rationale
}
