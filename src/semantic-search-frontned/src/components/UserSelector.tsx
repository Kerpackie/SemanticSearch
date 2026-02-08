import { useState } from 'react';
import './UserSelector.css';

export interface User {
  id: string;
  name: string;
  avatar: string;
}

// Hardcoded test users matching the recommender test data
export const TEST_USERS: User[] = [
  {
    id: '',
    name: 'Guest (No Personalization)',
    avatar: '👤',
  },
  {
    id: '00007d2de826758b65a93dd24ce629ed47051d84e43d377db969b74c6240134f',
    name: 'Emma Johnson',
    avatar: '👩',
  },
  {
    id: '0000ca64fcb140efb73420cb35a631c3edf41450e7827c3be3630c3f69e26969',
    name: 'Michael Chen',
    avatar: '👨',
  },
  {
    id: '0001f1ccb3ccaef1f88e34a4e4935f064a8b7deefca3f43277f6f4bc1cc9c455',
    name: 'Sarah Williams',
    avatar: '👱‍♀️',
  },
];

interface UserSelectorProps {
  currentUser: User | null;
  onUserChange: (user: User) => void;
}

export function UserSelector({ currentUser, onUserChange }: UserSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);

  const handleUserSelect = (user: User) => {
    onUserChange(user);
    setIsOpen(false);
  };

  return (
    <div className="user-selector">
      <button 
        className="user-selector-button"
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Select user"
      >
        <span className="user-avatar">{currentUser?.avatar || '👤'}</span>
        <span className="user-name">{currentUser?.name || 'Select User'}</span>
        <span className="dropdown-arrow">{isOpen ? '▲' : '▼'}</span>
      </button>

      {isOpen && (
        <>
          <div className="dropdown-backdrop" onClick={() => setIsOpen(false)} />
          <div className="user-dropdown">
            {TEST_USERS.map((user) => (
              <button
                key={user.id || 'guest'}
                className={`user-option ${currentUser?.id === user.id ? 'selected' : ''}`}
                onClick={() => handleUserSelect(user)}
              >
                <span className="user-avatar">{user.avatar}</span>
                <span className="user-name">{user.name}</span>
                {currentUser?.id === user.id && <span className="check-mark">✓</span>}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
