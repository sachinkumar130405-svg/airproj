// frontend/src/components/Dashboard.tsx
import React, { useState } from 'react';
import AQIPredictionChart from './AQIPredictionChart';
import HistoricalChart from './HistoricalChart';
import RoutePlanner from './RoutePlanner';
import RegionHighlight from './RegionHighlight';

const Dashboard: React.FC = () => {
  const [selectedLocation, setSelectedLocation] = useState('Connaught Place');
  const [forecast, setForecast] = useState([]);
  const [historical, setHistorical] = useState([]);

  const handlePredict = async () => {
    const response = await fetch('/predict_text', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ location_text: selectedLocation, forecast_hours: 24 })
    });
    const data = await response.json();
    setForecast(data.forecasts);
    // Fetch historical (assume endpoint /history/location)
    const histResponse = await fetch(`/history?location=${selectedLocation}`);
    setHistorical(await histResponse.json());
  };

  return (
    <div style={{ padding: '16px' }}>
      <input type="text" value={selectedLocation} onChange={e => setSelectedLocation(e.target.value)} />
      <button onClick={handlePredict}>Predict</button>
      <button onClick={() => navigator.geolocation.getCurrentPosition(pos => setSelectedLocation(`${pos.coords.latitude},${pos.coords.longitude}`))}>Use GPS</button>
      <AQIPredictionChart forecast={forecast} />
      <HistoricalChart historical={historical} />
      <RoutePlanner />
      <RegionHighlight />
    </div>
  );
};

export default Dashboard;