import { Sparkles, Palette } from 'lucide-react';
import './OutfitExplanation.css';

interface OutfitExplanationProps {
  outfitName: string;
  styleDescription: string;
}

export function OutfitExplanation({ outfitName, styleDescription }: OutfitExplanationProps) {
  return (
    <div className="outfit-explanation">
      <div className="outfit-explanation-icon">
        <Sparkles size={28} strokeWidth={1.75} />
      </div>
      <div className="outfit-explanation-body">
        <div className="outfit-explanation-header">
          <Palette size={14} strokeWidth={2} className="outfit-explanation-palette" />
          <span className="outfit-explanation-label">AI Outfit Concept</span>
        </div>
        <h3 className="outfit-explanation-name">{outfitName}</h3>
        <p className="outfit-explanation-description">{styleDescription}</p>
      </div>
    </div>
  );
}

