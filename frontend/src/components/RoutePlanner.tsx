// frontend/src/components/RoutePlanner.tsx
import React, { useState } from 'react';

const RoutePlanner: React.FC = () => {
  const [start, setStart] = useState('');
  const [end, setEnd] = useState('');
  const [route, setRoute] = useState([]);

  const handleRoute = async () => {
    const response = await fetch('/safe_route', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ start_text: start, end_text: end })
    });
    const data = await response.json();
    setRoute(data.route_segments);
    // Render on map (integrate with RouteMap)
  };

  return (
    <div>
      <input type="text" placeholder="Start address" value={start} onChange={e => setStart(e.target.value)} />
      <input type="text" placeholder="End address" value={end} onChange={e => setEnd(e.target.value)} />
      <button onClick={handleRoute}>Get Route</button>
    </div>
  );
};

export default RoutePlanner;