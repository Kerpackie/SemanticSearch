import './Header.css';

interface HeaderProps {
  cartCount?: number;
}

export function Header({ cartCount = 0 }: HeaderProps) {
  return (
    <header className="header">
      <div className="header-content">
        <div className="logo">
          <span className="logo-icon">👗</span>
          <span className="logo-text">StyleSearch</span>
        </div>
        <nav className="nav">
          <a href="#" className="nav-link">New Arrivals</a>
          <a href="#" className="nav-link">Collections</a>
          <a href="#" className="nav-link">Sale</a>
        </nav>
        <div className="header-actions">
          <button className="icon-button">❤️</button>
          <button className="icon-button cart-button">
            🛒
            {cartCount > 0 && <span className="cart-badge">{cartCount}</span>}
          </button>
          <button className="icon-button">👤</button>
        </div>
      </div>
    </header>
  );
}
