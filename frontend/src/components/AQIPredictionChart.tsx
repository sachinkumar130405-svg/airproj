// frontend/src/components/AQIPredictionChart.tsx
import React from 'react';
import { Chart as ChartJS, LineElement, CategoryScale, LinearScale, Tooltip, Legend } from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(LineElement, CategoryScale, LinearScale, Tooltip, Legend);

interface Props {
  forecast: any[];
}

const AQIPredictionChart: React.FC<Props> = ({ forecast }) => {
  const labels = forecast.map(f => f.ts);
  const data = forecast.map(f => f.predicted_aqi);

  return (
    <Line data={{
      labels,
      datasets: [{ label: 'AQI Forecast', data, borderColor: '#4b9be1' }]
    }} />
  );
};

export default AQIPredictionChart;