import { ShoppingBag, Heart, ShoppingCart, User } from 'lucide-react';
import './Header.css';

interface HeaderProps {
  cartCount?: number;
}

export function Header({ cartCount = 0 }: HeaderProps) {
  return (
    <header className="header">
      <div className="header-content">
        <div className="logo">
          <ShoppingBag className="logo-icon" size={28} strokeWidth={1.75} />
          <span className="logo-text">StyleSearch</span>
        </div>
        <nav className="nav">
          <a href="#" className="nav-link">New Arrivals</a>
          <a href="#" className="nav-link">Collections</a>
          <a href="#" className="nav-link">Sale</a>
        </nav>
        <div className="header-actions">
          <button className="icon-button" aria-label="Wishlist">
            <Heart size={20} strokeWidth={1.75} />
          </button>
          <button className="icon-button cart-button" aria-label="Cart">
            <ShoppingCart size={20} strokeWidth={1.75} />
            {cartCount > 0 && <span className="cart-badge">{cartCount}</span>}
          </button>
          <button className="icon-button" aria-label="Account">
            <User size={20} strokeWidth={1.75} />
          </button>
        </div>
      </div>
    </header>
  );
}
