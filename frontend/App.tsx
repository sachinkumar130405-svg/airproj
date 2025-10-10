// frontend/src/App.tsx
import React from 'react';
import Dashboard from './components/Dashboard';

const App: React.FC = () => {
  return (
    <div style={{ background: '#1a1a2e', color: '#fff', minHeight: '100vh' }}>
      <header style={{ padding: '16px', textAlign: 'center' }}>
        <h1>Delhi AQI Predictor</h1>
      </header>
      <Dashboard />
    </div>
  );
};

export default App;